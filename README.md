# Play Fair for epic-kitchens-100: Frame Attribution in Video Models

This repo introduces an attribution method for explaining action recognition models. Such models fuse information from multiple frames within a video, through score aggregation or relational reasoning. 

We break down a model’s class score into the sum of contributions from each frame, fairly. Our method adapts an axiomatic solution to fair reward distribution in cooperative games, known as the Shapley value, for elements in a variable-length sequence, which we call the Element Shapley Value (ESV). Critically, we propose a tractable approximation of ESV that scales linearly with the number of frames in the sequence.

# Setup

## Environment

You will always need to set your PYTHONPATH to include the `src` folder. We provide an `.envrc` for use with [`direnv`](https://direnv.net/) which will automatically do that for you when you `cd` into the project directory. Alternatively just run:

```bash
$ export PYTHONPATH=$PWD/src 
```

Create a conda environment with the environment file, you will also always need to activate this environment to use the correct packages

```bash
$ conda env create -n epic-100 -f environment.yml
$ conda activate epic-100
```

Alternatively, just add it to the `.envrc` file which will run it automatically:

```bash
$ echo conda activate epic-100 >> .envrc
$ direnv allow
```

You will also need to install a version of ffmpeg with vp9 support, we suggest using the static builds provided by John Van Sickle:

```bash
$ wget "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz"
$ tar -xvf "ffmpeg-git-amd64-static.tar.xz"
$ mkdir -p bin
$ mv ffmpeg-git-*-amd64-static/{ffmpeg,ffprobe} bin
```

## Data

We store our files in the [`gulpio2`](https://github.com/willprice/GulpIO2) format.

1. Download (p01 frames?) [epic-kitchens-100](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m)

can download easier with the download script we provide

```bash
$ cd datasets
$ bash ./download_p01_frames.sh
```
2. Gulp the dataset (we supply a labels pkl for just p01) (RGB P01 FRAMES ONLY)

```bash
$ python src/scripts/gulp_data \
    /path/to/rgb/frames (datasets/epic-100/p01-frames) \
    datasets/epic-100/gulp \
    p01.pkl (or datasets/epic-100/labels/p01.pkl) \
    rgb
```

If you need to write the gulp directory to somewhere other than the path specified in the command above, make sure to symlink it afterwards to datasets/epic-100/gulp/train so the configuration files don't need to be updated.

```bash
$ ln -s /path/to/gulp/directory datasets/epic-100/gulp/train
```

## Models

We provide TRN model pretrained on the training set of EPIC-KITCHENS-100

```bash
$ cd checkpoints
$ bash ./download_checkpoints.sh
```

check that it has downloaded:

```bash
$ tree -h
.
├── [ 150]  download.sh
└── [103M]  trn_rgb.ckpt

0 directories, 2 files
```

# Extracting Features

As computing ESVs is expensive, we work with temporal models that operate over features. We can run these in a reasonable amount of time depending on the number of frames and whether approximate methods are used.

We provide a script to extract per-frame features, saving them to a PKL file. Extract these features using the TRN checkpoint

```bash
$ python src/scripts/extract_features.py \
    /path/to/gulp/directory \
    checkpoints/trn_rgb.ckpt \
    datasets/epic-100/features/p01_features.pkl
```

optionally you can change the number of workers for the PyTorch DataLoader with the `--num_workers` argument. If this script failes at any point then you can simply rerun and it will continue from where it crashed.

# Training MTRN models

We train an MLP classifier for each 1-8 frame inputs which passes the concatenated frame features as inputs through two fully connected layers to give predictions for the verb / noun classes.

We use tensorboard to log extensive training results which you can view either throughout the training or afterwards at any point, run

```bash
$ tensorboard --logdir=datasets/epic-100/runs --reload_multifile True
```

In this example we train the verbs and the nouns separately although the framework is available to train both in the same neural network. For the selected learning rate `3e-4` and batch size `512` the best testing accuracy was observed when training for 200 epochs. You can run the scripts one by one

```bash
$ python src/scripts/train_mtrn.py \
    datasets/epic-100/features/p01_features.pkl \
    datasets/epic-100/models/ \
    --type "verb"

$ python src/scripts/train_mtrn.py \
    datasets/epic-100/features/p01_features.pkl \
    datasets/epic-100/models/ \
    --type "noun"
```

Alternatively we provide a script that will run this automatically for you with an argument for the epochs

```bash
$ bash ./train_verb_noun.sh 200
```

## Additional arguments

`--val_features_pkl` If you want to train / test on two distinct frame feature sets rather than using a train/test split

`--train-test-split` Specify a train/test split between 0 and 1, default 0.3

`--min-frames` Minimum number of frames to train models for, default 1

`--max-frames` Maximum number of frames to train models for, default 8 (these two arguments can also be used in case of a training crash)

`--epoch` How many iterations to train the models for, default 200

`--batch-size` The size of the mini batches to feed to the model at a time, default 512

# Computing ESVs

We compute ESVs using a collection of models, each which operate over a fixed-length input (e.g. TRN).

## Computing class priors

Regardless of whether your model supports variable-length inputs or not, we need to compute the class priors to use in the computation of the ESVs. We provide a script that does this by computing the emprical class frequency for both verb and nouns in p01.

```bash
$ python src/scripts/compute_verb_class_priors.py \
    p01.pkl \
    datasets/epic-100/labels/verb_class_priors.csv

$ python src/scripts/compute_noun_class_priors.py \
    p01.pkl \
    datasets/epic-100/labels/noun_class_priors.csv
```

## Models supporting a fixed-length input (e.g. TRN)

For models that don't support a variable-length

This is implemented in the OnlineShapleyAttributor class.

We provide an example of how to do this for TRN, as the basic variant only supports a fixed-length input. (make sure you've set up your environment, downloaded and prepped the dataset, downloaded the models and extracted the features first)

```bash
$ python src/scripts/compute_esvs.py \
    datasets/epic-100/features/p01_features.pkl \
    datasets/epic-100/models/
    datasets/epic-100/labels/verb_class_priors.csv \
    datasets/epic-100/labels/verb_class_priors.csv \
    datasets/epic-100/esvs/mtrn-esv-n_frames=8.pkl \
    --sample_n_frames 8
```

## Visualisation

We provide a dashboard to investigate model behaviour when we vary how many frames are fed to the model. This dashboard is powered by multiple sets of results produced by the `compute_esvs.py` script

First compute ESVs for 1-8 frame inputs:

```bash
$ for n in $(seq 1 8); do
    python src/scripts/compute_esvs.py \
        datasets/epic-100/features/p01_features.pkl \
        datasets/epic-100/models/
        datasets/epic-100/labels/verb_class_priors.csv \
        datasets/epic-100/labels/verb_class_priors.csv \
        datasets/epic-100/esvs/mtrn-esv-n_frames=$n.pkl \
        --sample_n_frames $n
  done
```

Then we can collate them

```bash
$ python src/scripts/collate_esvs.py
    --dataset "Epic Kitchens 100 (P01)"
    --model "MTRN"
    datasets/epic-100/esvs/mtrn-esv-n_frames={1..8}.pkl \
    datasets/epic-100/esvs/mtrn-esv-min_frames=1-max_frames=8.pkl
```

before we can run the dashboard, we need to dump out the videos from the gulp directory as webm files (since we gulp the files, it alters the FPS). Watch out that you don't end up using the conda bundled ffmpeg which doesn't support VP9 encoding if you replace `.bin/ffmpeg` with `ffmpeg`, check which you are using by running `which ffmpeg`
