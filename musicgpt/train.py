import argparse
import logging
import pathlib
import pprint
import shutil
import sys

import linear_attention_transformer
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
    parser.add_argument(
        "-r",
        "--representation",
        choices=("compact", "mmm", "remi"),
        default="remi",
        help="representation key",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=("compact", "transformer", "performer", "linear"),
        default="linear",
        help="model type",
    )
    parser.add_argument(
        "-t", "--train_names", type=pathlib.Path, help="training names"
    )
    parser.add_argument(
        "-v", "--valid_names", type=pathlib.Path, help="validation names"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=16,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--grad_acc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to accumulate gradients to increase the batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug_pitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to randomly transpose the pitches",
    )
    parser.add_argument(
        "--aug_beat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to randomly select a starting beat",
    )
    # Model
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=64,
        type=int,
        help="maximum number of beats",
    )
    parser.add_argument("--dim", default=768, type=int, help="model dimension")
    parser.add_argument(
        "-l", "--layers", default=12, type=int, help="number of layers"
    )
    parser.add_argument(
        "--heads", default=12, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout rate"
    )
    parser.add_argument(
        "--abs_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use absolute positional embedding",
    )
    parser.add_argument(
        "--rel_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use relative positional embedding",
    )
    # Training
    parser.add_argument(
        "--steps",
        default=500000,
        type=int,
        help="number of steps",
    )
    parser.add_argument(
        "--valid_steps",
        default=10000,
        type=int,
        help="validation frequency",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "-e",
        "--early_stopping_tolerance",
        default=10,
        type=int,
        help="number of extra validation rounds before early stopping",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0005,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        default=5000,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "--lr_decay_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "--grad_norm_clip",
        default=1.0,
        type=float,
        help="gradient norm clipping",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="weight decay",
    )
    # Others
    parser.add_argument(
        "-g", "--gpu", nargs="+", type=int, help="gpu number(s)"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.train_names is None:
            args.train_names = pathlib.Path(
                f"data/{args.dataset}/processed/train-names.txt"
            )
        if args.valid_names is None:
            args.valid_names = pathlib.Path(
                f"data/{args.dataset}/processed/valid-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_baseline_{args.dataset}")
    if args.representation == "compact":
        representation = representation_compact
        dataset = dataset_compact
        assert args.model == "compact"
    elif args.representation == "mmm":
        representation = representation_mmm
        dataset = dataset_flat
        assert args.model != "compact"
    elif args.representation == "remi":
        representation = representation_remi
        dataset = dataset_flat
        assert args.model != "compact"

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        level=logging.DEBUG + 10 * args.quiet,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "train.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'train-args.json'}")
    utils.save_args(args.out_dir / "train-args.json", args)

    # # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu[0]}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.get_encoding()

    # Load the indexer
    kwargs = {}
    if args.representation in ("remi", "mmm"):
        indexer = representation.Indexer(encoding["event_code_map"])
        kwargs["indexer"] = indexer
        kwargs["encode_fn"] = representation.encode_notes

    # Create the datasets
    logging.debug(f"Creating the datasets...")
    train_dataset = dataset.MusicDataset(
        args.train_names,
        args.in_dir,
        encoding=encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_pitch_augmentation=args.aug_pitch,
        use_beat_augmentation=args.aug_beat,
        use_csv=args.use_csv,
        **kwargs,
    )
    valid_dataset = dataset.MusicDataset(
        args.valid_names,
        args.in_dir,
        encoding=encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
        **kwargs,
    )

    # Create the datasets
    logging.debug(f"Creating the data loader...")
    batch_size = max(args.batch_size // 8, 8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )
    logging.debug(f"Using batch size: {batch_size}")
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        args.batch_size,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.debug(f"Creating model...")
    if args.model == "compact":
        model = music_x_transformers.MusicTransformerWrapper(
            encoding=encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_abs_pos_emb=args.abs_pos_emb,
            emb_dropout=args.dropout,
            attn_layers=music_x_transformers.Decoder(
                dim=args.dim,
                depth=args.layers,
                heads=args.heads,
                rotary_pos_emb=args.rel_pos_emb,
                attn_dropout=args.dropout,
                ff_dropout=args.dropout,
            ),
        )
        model = music_x_transformers.MusicAutoregressiveWrapper(
            model, encoding=encoding
        )
    elif args.model == "transformer":
        model = x_transformers.TransformerWrapper(
            num_tokens=len(indexer),
            max_seq_len=args.max_seq_len,
            use_abs_pos_emb=args.abs_pos_emb,
            emb_dropout=args.dropout,
            attn_layers=x_transformers.Decoder(
                dim=args.dim,
                depth=args.layers,
                heads=args.heads,
                rotary_pos_emb=args.rel_pos_emb,
                attn_dropout=args.dropout,
                ff_dropout=args.dropout,
            ),
        )
        model = x_transformers.AutoregressiveWrapper(model)
    elif args.model == "performer":
        assert not (args.abs_pos_emb and args.rel_pos_emb)
        model = performer_pytorch.PerformerLM(
            num_tokens=len(indexer),
            max_seq_len=args.max_seq_len,
            dim=args.dim,
            depth=args.layers,
            heads=args.heads,
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
            emb_dropout=args.dropout,
            ff_dropout=args.dropout,
            attn_dropout=args.dropout,
            local_attn_heads=4,
            local_window_size=256,
            rotary_position_emb=args.rel_pos_emb,
            shift_tokens=True,
        )
        model = performer_pytorch.AutoregressiveWrapper(model)
    elif args.model == "linear":
        assert not (args.abs_pos_emb and args.rel_pos_emb)
        model = linear_attention_transformer.LinearAttentionTransformerLM(
            num_tokens=len(indexer),
            dim=args.dim,
            heads=args.heads,
            depth=args.layers,
            max_seq_len=args.max_seq_len,
            causal=True,
            ff_dropout=args.dropout,
            attn_dropout=args.dropout,
            attn_layer_dropout=args.dropout,
            blindspot_size=64,
            n_local_attn_heads=args.heads,
            local_attn_window_size=128,
            reversible=True,
            ff_chunks=2,
            ff_glu=True,
            attend_axially=False,
            shift_tokens=True,
            use_rotary_emb=args.rel_pos_emb,
        )
        model = linear_attention_transformer.AutoregressiveWrapper(model)
    if args.gpu is not None and len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu).to(device)
    else:
        model = model.to(device)

    # Summarize the model
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"Number of parameters: {n_parameters}")
    logging.info(f"Number of trainable parameters: {n_trainables}")

    # Create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            args.lr_warmup_steps,
            args.lr_decay_steps,
            args.lr_decay_multiplier,
        ),
    )

    # Create a file to record losses
    loss_csv = open(args.out_dir / "loss.csv", "w")
    loss_csv.write("step,train_loss,valid_loss\n")

    # Initialize variables
    step = 0
    min_val_loss = float("inf")
    if not args.early_stopping:
        count_early_stopping = 0

    # Iterate for the specified number of steps
    train_iterator = iter(train_loader)
    epoch = 0
    grad_acc_steps = 1

    # Wrap with a try-except block to handle keyboard interrupt
    try:
        while step < args.steps:

            # Training
            logging.debug(f"Training...")
            model.train()
            recent_losses = []

            # Clear gradients
            optimizer.zero_grad()

            # Set batch size
            if args.batch_size >= 16 and epoch in (1, 5, 10):
                if epoch == 1:
                    batch_size = max(args.batch_size // 4, 8)
                elif epoch == 5:
                    batch_size = max(args.batch_size // 2, 8)
                elif epoch == 10:
                    batch_size = args.batch_size
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size,
                    shuffle=True,
                    num_workers=args.jobs,
                    collate_fn=dataset.MusicDataset.collate,
                )
                train_iterator = iter(train_loader)
                logging.debug(f"Using batch size: {batch_size}")

            # Accumulate gradients to further increase the batch size
            if args.grad_acc:
                if epoch == 15:
                    grad_acc_steps = 2
                elif epoch == 20:
                    grad_acc_steps = 4

            # Training loop
            pbar = tqdm.tqdm(total=args.valid_steps, ncols=120)
            for local_step in range(args.valid_steps * grad_acc_steps):
                # Get next batch
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    # Reinitialize dataset iterator
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                # Get input and output pair
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)

                # Update the model parameters
                if args.model in ("compact", "transformer"):
                    loss = model(seq, mask=mask)
                else:
                    loss = model(seq, mask=mask, return_loss=True)
                if args.gpu is not None and len(args.gpu) > 1:
                    loss = loss.mean()

                loss = loss / grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_norm_clip
                )
                if (local_step + 1) % grad_acc_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    pbar.update(1)

                # Compute the moving average of the loss
                recent_losses.append(float(loss) * grad_acc_steps)
                if len(recent_losses) > 10:
                    del recent_losses[0]
                train_loss = np.mean(recent_losses)
                pbar.set_postfix(loss=f"{train_loss:8.4f}")
            pbar.close()

            # Release GPU memory right away
            del seq, mask

            # Validation
            logging.debug(f"Validating...")
            model.eval()
            with torch.no_grad():
                total_loss = 0
                count = 0
                for batch in valid_loader:
                    # Get input and output pair
                    seq = batch["seq"].to(device)
                    mask = batch["mask"].to(device)

                    # Pass through the model
                    if args.model in ("compact", "transformer"):
                        loss = model(seq, mask=mask)
                    else:
                        loss = model(seq, mask=mask, return_loss=True)
                    if args.gpu is not None and len(args.gpu) > 1:
                        loss = loss.mean()

                    # Accumulate validation loss
                    count += len(batch)
                    total_loss += len(batch) * float(loss)
            val_loss = total_loss / count
            logging.info(f"Validation loss: {val_loss:.4f}")

            # Release GPU memory right away
            del seq, mask

            # Write losses to file
            loss_csv.write(f"{step},{train_loss},{val_loss}\n")

            # Save the model
            checkpoint_filename = checkpoint_dir / f"model_{step}.pt"
            torch.save(model.state_dict(), checkpoint_filename)
            logging.info(f"Saved the model to: {checkpoint_filename}")

            # Copy the model if it is the best model so far
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                shutil.copyfile(
                    checkpoint_filename,
                    checkpoint_dir / "best_model.pt",
                )
                # Reset the early stopping counter if we found a better model
                if args.early_stopping:
                    count_early_stopping = 0
            elif args.early_stopping:
                # Increment the early stopping counter if no improvement is found
                count_early_stopping += 1

            # Early stopping
            if (
                not args.early_stopping
                and count_early_stopping > args.early_stopping_tolerance
            ):
                logging.info(
                    "Stopped the training for no improvements in "
                    f"{args.early_stopping_tolerance} rounds."
                )
                break

            epoch += 1

    except KeyboardInterrupt:
        logging.info("Detected KeyboardInterrupt and stopped the training!")

    finally:
        # Save the optimizer states
        optimizer_filename = checkpoint_dir / f"optimizer_{step}.pt"
        torch.save(optimizer.state_dict(), optimizer_filename)
        logging.info(f"Saved the optimizer state to: {optimizer_filename}")

        # Save the scheduler states
        scheduler_filename = checkpoint_dir / f"scheduler_{step}.pt"
        torch.save(scheduler.state_dict(), scheduler_filename)
        logging.info(f"Saved the scheduler state to: {scheduler_filename}")

        # Close the file
        loss_csv.close()


if __name__ == "__main__":
    main()
