# Music Generative Pretrained Transformer

We aim to scale up music transformer models using the largest symbolic music dataset available.

## Prerequisites

### Set up development environment

We recommend using Conda. You can create the environment with the following command.

```sh
conda env create -f environment.yml
```

## Preprocessing

### Download the MuseScore dataset

For copyright concern, please download the MuseScore data yourselves. You may find this repository helpful (https://github.com/Xmader/musescore-dataset).

### Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/muse/muse -type f -name *.mscz | cut -c 16- > data/muse/original-names.txt
```

### Convert the data

Convert the MSCZ files into MusPy files for processing.

```sh
python convert_muse.py
```

> Note: You may enable multiprocessing via the `-j {JOBS}` option. For example, `python convert_muse.py -j 10` will run the script with 10 jobs.

### Extract the note list

Extract a list of notes from the MusPy JSON files.

```sh
python extract.py -d muse
```

### Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python split.py -d muse
```

## Training

Train a Music GPT model.

- Absolute positional embedding (APE):

  `python musicgpt/train.py -d muse -o exp/muse/ape -g 0`

- Relative positional embedding (RPE):
muse
  `python musicgpt/train.py -d muse -o exp/muse/rpe --no-abs_pos_emb --rel_pos_emb -g 0`

- No positional embedding (NPE):

  `python musicgpt/train.py -d muse -o exp/muse/npe --no-abs_pos_emb --no-rel_pos_emb -g 0`

> Please run `python musicgpt/train.py -h` to see additional options.

## Evaluation

Evaluate the trained model.

```sh
python musicgpt/evaluate.py -d muse -o exp/muse/ape -ns 100 -g 0
```

> Please run `python musicgpt/evaluate.py -h` to see additional options.

## Generation (inference)

Generate new samples using a trained model.

```sh
python musicgpt/generate.py -d muse -o exp/muse/ape -g 0
```

> Please run `python musicgpt/generate.py -h` to see additional options.
