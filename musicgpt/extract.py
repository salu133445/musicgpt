"""Extract note sequences from music."""
import argparse
import logging
import pathlib
import pprint
import sys

import joblib
import muspy
import numpy as np
import tqdm

import representation_utils
import utils


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract note sequences from music"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd", "muse"),
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
        "-r",
        "--resolution",
        default=12,
        type=int,
        help="number of time steps per quarter note",
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="number of jobs",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def extract(name, in_dir, out_dir, resolution):
    """Encode a note sequence into the representation."""
    # Load the score
    music = muspy.load(in_dir / f"{name}.json")

    # Encode the score
    notes = representation_utils.extract_notes(music, resolution)

    # Filter out bad files
    if len(notes) < 50:
        return

    # Set start beat to zero
    notes[:, 0] = notes[:, 0] - notes[0, 0]

    # Make sure output directory exists
    (out_dir / name).parent.mkdir(exist_ok=True, parents=True)

    # Save the notes as a CSV file
    representation_utils.save_csv_notes(
        (out_dir / name).with_suffix(".csv"), notes
    )

    # Save the notes as a NPY file
    np.save((out_dir / name).with_suffix(".npy"), notes)

    return name


@utils.ignore_exceptions
def extract_ignore_exceptions(name, in_dir, out_dir, resolution):
    """Encode a note sequence into machine-learning friendly codes,
    ignoring all exceptions."""
    return extract(name, in_dir, out_dir, resolution)


def process(name, in_dir, out_dir, resolution, ignore_exceptions=True):
    """Wrapper for multiprocessing."""
    if ignore_exceptions:
        return extract_ignore_exceptions(name, in_dir, out_dir, resolution)
    return extract(name, in_dir, out_dir, resolution)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Set default arguments
    if args.dataset is not None:
        if args.names is None:
            args.names = args.names or pathlib.Path(
                f"data/{args.dataset}/processed/json-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/json/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(
                f"data/{args.dataset}/processed/notes/"
            )

    # Make sure output directory exists
    args.out_dir.mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Get filenames
    logging.info("Loading names...")
    names = utils.load_txt(args.names)

    # Iterate over names
    logging.info("Iterating over names...")
    extracted_names = []
    if args.jobs == 1:
        for name in (pbar := tqdm.tqdm(names, ncols=120)):
            pbar.set_postfix_str(name)
            result = process(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.ignore_exceptions,
            )
            if result is not None:
                extracted_names.append(result)
    else:
        results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
            joblib.delayed(process)(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.ignore_exceptions,
            )
            for name in names
        )
        extracted_names = [result for result in results if result is not None]
    logging.info(
        f"Extracted {len(extracted_names)} out of {len(names)} files."
    )

    # Save successfully encoded names
    out_filename = args.out_dir.parent / "names.txt"
    utils.save_txt(out_filename, extracted_names)
    logging.info(f"Saved the extracted filenames to: {out_filename}")


if __name__ == "__main__":
    main()
