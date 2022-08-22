import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

import linear_attention_transformer
import matplotlib.pyplot as plt
import numpy as np
import performer_pytorch
import torch
import torch.utils.data
import tqdm
import x_transformers

import dataset_compact
import dataset_flat
import music_x_transformers
import representation_compact
import representation_mmm
import representation_remi
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "muse"),
        required=True,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        default=50,
        type=int,
        help="number of samples to generate",
    )
    # Data
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="whether to shuffle the test data",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    # Model
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        help="sequence length to generate (default to max sequence length)",
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j", "--jobs", default=1, type=int, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def save_pianoroll(filename, music, size=None, **kwargs):
    """Save the piano roll to file."""
    music.show_pianoroll(track_label="program", **kwargs)
    if size is not None:
        plt.gcf().set_size_inches(size)
    plt.savefig(filename)
    plt.close()


def save_result(
    filename, data, sample_dir, encoding, vocabulary, representation
):
    """Save the results in multiple formats."""
    # Save as a numpy array
    np.save(sample_dir / "npy" / f"{filename}.npy", data)

    # Save as a CSV file
    representation.save_csv_codes(sample_dir / "csv" / f"{filename}.csv", data)

    # Save as a TXT file
    representation.save_txt(
        sample_dir / "txt" / f"{filename}.txt", data, vocabulary
    )

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding, vocabulary)

    # Save as a MusPy JSON file
    music.save(sample_dir / "json" / f"{filename}.json")

    # Save as a piano roll
    save_pianoroll(
        sample_dir / "png" / f"{filename}.png", music, (20, 5), preset="frame"
    )

    # Save as a MIDI file
    music.write(sample_dir / "mid" / f"{filename}.mid")

    # Save as a WAV file
    music.write(
        sample_dir / "wav" / f"{filename}.wav",
        options="-o synth.polyphony=4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav" / f"{filename}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3" / f"{filename}.mp3")]
    )

    # Trim the music
    music.trim(music.resolution * 64)

    # Save the trimmed version as a piano roll
    save_pianoroll(
        sample_dir / "png-trimmed" / f"{filename}.png", music, (10, 5)
    )

    # Save as a WAV file
    music.write(
        sample_dir / "wav-trimmed" / f"{filename}.wav",
        options="-o synth.polyphony=4096",
    )

    # Save also as a MP3 file
    subprocess.check_output(
        ["ffmpeg", "-loglevel", "error", "-y", "-i"]
        + [str(sample_dir / "wav-trimmed" / f"{filename}.wav")]
        + ["-b:a", "192k"]
        + [str(sample_dir / "mp3-trimmed" / f"{filename}.mp3")]
    )


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = pathlib.Path(
                f"data/{args.dataset}/processed/test-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "generate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'generate-args.json'}")
    utils.save_args(args.out_dir / "generate-args.json", args)

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")

    # Set variables
    if args.seq_len is None:
        args.seq_len = train_args["max_seq_len"]
    if train_args["representation"] == "compact":
        representation = representation_compact
        dataset = dataset_compact
    elif train_args["representation"] == "mmm":
        representation = representation_mmm
        dataset = dataset_flat
    elif train_args["representation"] == "remi":
        representation = representation_remi
        dataset = dataset_flat

    # Make sure the sample directory exists
    sample_dir = args.out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "npy").mkdir(exist_ok=True)
    (sample_dir / "csv").mkdir(exist_ok=True)
    (sample_dir / "txt").mkdir(exist_ok=True)
    (sample_dir / "json").mkdir(exist_ok=True)
    (sample_dir / "png").mkdir(exist_ok=True)
    (sample_dir / "mid").mkdir(exist_ok=True)
    (sample_dir / "wav").mkdir(exist_ok=True)
    (sample_dir / "mp3").mkdir(exist_ok=True)
    (sample_dir / "png-trimmed").mkdir(exist_ok=True)
    (sample_dir / "wav-trimmed").mkdir(exist_ok=True)
    (sample_dir / "mp3-trimmed").mkdir(exist_ok=True)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Get representation
    if train_args["representation"] == "mmm":
        representation = representation_mmm
    elif train_args["representation"] == "remi":
        representation = representation_remi
    else:
        raise ValueError(
            f"Unknown representation: {train_args['representation']}"
        )

    # Load the encoding
    encoding = representation.get_encoding()

    # Load the indexer
    indexer = representation.Indexer(encoding["event_code_map"])

    # Get the vocabulary
    vocabulary = encoding["code_event_map"]

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    test_dataset = dataset.MusicDataset(
        args.names,
        args.in_dir,
        encoding=encoding,
        indexer=indexer,
        encode_fn=representation.encode_notes,
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        use_csv=args.use_csv,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=args.shuffle,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.debug(f"Creating model...")
    if train_args["model"] == "compact":
        model = music_x_transformers.MusicTransformerWrapper(
            encoding=encoding,
            max_seq_len=train_args["max_seq_len"],
            max_beat=train_args["max_beat"],
            use_abs_pos_emb=train_args["abs_pos_emb"],
            emb_dropout=train_args["dropout"],
            attn_layers=music_x_transformers.Decoder(
                dim=train_args["dim"],
                depth=train_args["layers"],
                heads=train_args["heads"],
                rotary_pos_emb=train_args["rel_pos_emb"],
                attn_dropout=train_args["dropout"],
                ff_dropout=train_args["dropout"],
            ),
        )
        model = music_x_transformers.MusicAutoregressiveWrapper(
            model, encoding=encoding
        )
    elif train_args["model"] == "transformer":
        model = x_transformers.TransformerWrapper(
            num_tokens=len(indexer),
            max_seq_len=train_args["max_seq_len"],
            use_abs_pos_emb=train_args["abs_pos_emb"],
            emb_dropout=train_args["dropout"],
            attn_layers=x_transformers.Decoder(
                dim=train_args["dim"],
                depth=train_args["layers"],
                heads=train_args["heads"],
                rotary_pos_emb=train_args["rel_pos_emb"],
                attn_dropout=train_args["dropout"],
                ff_dropout=train_args["dropout"],
            ),
        )
        model = x_transformers.AutoregressiveWrapper(model)
    elif train_args["model"] == "performer":
        model = performer_pytorch.PerformerLM(
            num_tokens=len(indexer),
            max_seq_len=train_args["max_seq_len"],
            dim=train_args["dim"],
            depth=train_args["layers"],
            heads=train_args["heads"],
            causal=True,
            nb_features=256,
            feature_redraw_interval=1000,
            generalized_attention=False,
            kernel_fn=torch.nn.ReLU(),
            reversible=True,
            ff_chunks=10,
            use_scalenorm=False,
            use_rezero=False,
            ff_glu=True,
            emb_dropout=train_args["dropout"],
            ff_dropout=train_args["dropout"],
            attn_dropout=train_args["dropout"],
            local_attn_heads=4,
            local_window_size=256,
            rotary_position_emb=train_args["rel_pos_emb"],
            shift_tokens=True,
        )
        model = performer_pytorch.AutoregressiveWrapper(model)
    elif train_args["model"] == "linear":
        model = linear_attention_transformer.LinearAttentionTransformerLM(
            num_tokens=len(indexer),
            dim=train_args["dim"],
            heads=train_args["heads"],
            depth=train_args["layers"],
            max_seq_len=train_args["max_seq_len"],
            causal=True,
            ff_dropout=train_args["dropout"],
            attn_dropout=train_args["dropout"],
            attn_layer_dropout=train_args["dropout"],
            blindspot_size=64,
            n_local_attn_heads=train_args["heads"],
            local_attn_window_size=128,
            reversible=True,
            ff_chunks=2,
            ff_glu=True,
            attend_axially=False,
            shift_tokens=True,
            use_rotary_emb=train_args["rel_pos_emb"],
        )
        model = linear_attention_transformer.AutoregressiveWrapper(model)
    model = model.to(device)

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Get special tokens
    sos = indexer["start-of-song"]
    eos = indexer["end-of-song"]

    # Get the logits filter function
    if args.filter == "top_k":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_k
    elif args.filter == "top_p":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_p
    elif args.filter == "top_a":
        filter_logits_fn = x_transformers.autoregressive_wrapper.top_a
    else:
        raise ValueError("Unknown logits filter.")

    # Iterate over the dataset
    with torch.no_grad():
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(args.n_samples), ncols=80):
            batch = next(data_iter)

            # ------------
            # Ground truth
            # ------------
            truth_np = batch["seq"][0].numpy()
            save_result(
                f"{i}_truth",
                truth_np,
                sample_dir,
                encoding,
                vocabulary,
                representation,
            )

            # ------------------------
            # Unconditioned generation
            # ------------------------

            if train_args["model"] == "compact":
                # Get output start tokens
                tgt_start = torch.zeros(
                    (1, 1, 7), dtype=torch.long, device=device
                )
                tgt_start[:, 0, 0] = sos

                # Generate new samples
                generated = model.generate(
                    tgt_start,
                    args.seq_len,
                    eos_token=eos,
                    temperature=args.temperature,
                    filter_logits_fn=args.filter,
                    filter_thres=args.filter_threshold,
                    monotonicity_dim=("type", "beat"),
                )
            else:
                # Get output start tokens
                tgt_start = torch.zeros(
                    (1, 1), dtype=torch.long, device=device
                )
                tgt_start[:, 0] = sos

                # Generate new samples
                generated = model.generate(
                    tgt_start,
                    args.seq_len,
                    eos_token=eos,
                    temperature=args.temperature,
                    filter_logits_fn=filter_logits_fn,
                    filter_thres=args.filter_threshold,
                )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            # Save the results
            save_result(
                f"{i}_unconditioned",
                generated_np[0],
                sample_dir,
                encoding,
                vocabulary,
                representation,
            )


if __name__ == "__main__":
    main()
