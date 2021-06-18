# Play Fair for epic-kitchens-100: Frame Attribution in Video Models

This repo extends the attribution method from [play-fair](https://github.com/willprice/play-fair) for explaining action recognition models with the epic-kitchens-100 dataset. Such models fuse information from multiple frames within a video, through score aggregation or relational reasoning. 

We break down a model’s class score into the sum of contributions from each frame, fairly. Our method adapts an axiomatic solution to fair reward distribution in cooperative games, known as the Shapley value, for elements in a variable-length sequence, which we call the Element Shapley Value (ESV). Critically, we propose a tractable approximation of ESV that scales linearly with the number of frames in the sequence.

If you want to explore further follow the set up guide below, extract features from the backbone models, and compute ESVs yourself.

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

Alternatively, just add it to the `.envrc` file which will run it automatically (be careful that you have set the source of your conda installation either in your .bashrc file or also by adding it to the top of the `.envrc` file)

```bash
$ echo 'conda activate epic-100' | cat - .envrc > temp && mv temp .envrc
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

1. Download p01 frames from [epic-kitchens-100](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m)

    You can either do this manually or use the included script (the script uses the [epic-downloader](https://github.com/epic-kitchens/epic-kitchens-download-scripts) which downloads the `.tar` files from the direct download link, download speeds for this may be (very) slow depending on the region, in which case we recommend to download using Academic Torrents, find out more [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts#download-speed))

    ```bash
    $ cd datasets
    $ bash ./download_p01_frames.sh
    ```
2. Extract the frames, if you downloaded the frames using the script above, then simply run
    ```bash
    $ cd datasets
    $ bash ./extract_p01_frames.sh
    ```
    
    or if you downloaded them externally then simply run the same script with the directory of the `/P01` folder as an argument
    ```bash
    $ cd datasets
    $ bash ./extract_p01_frames.sh <path-to-downloaded-rgb-frames>/P01/
    ```
    
    once extracted, make sure that you have them all extracted correctly:
    
    ```bash
    $ cd <path-to-rgb-frames>/P01
    $ tree -hd
    .
    ├── [3.9M]  P01_01
    ├── [1.1M]  P01_02
    ├── [272K]  P01_03
    ├── [256K]  P01_04
    ├── [3.0M]  P01_05
    ├── [1.1M]  P01_06
    ├── [412K]  P01_07
    ├── [248K]  P01_08
    ├── [8.2M]  P01_09
    ├── [320K]  P01_10
    ├── [3.6M]  P01_101
    ├── [520K]  P01_102
    ├── [364K]  P01_103
    ├── [500K]  P01_104
    ├── [4.1M]  P01_105
    ├── [1.4M]  P01_106
    ├── [252K]  P01_107
    ├── [268K]  P01_108
    ├── [10.0M]  P01_109
    ├── [1.2M]  P01_11
    ├── [456K]  P01_12
    ├── [240K]  P01_13
    ├── [3.2M]  P01_14
    ├── [2.1M]  P01_15
    ├── [460K]  P01_16
    ├── [2.6M]  P01_17
    ├── [8.4M]  P01_18
    └── [1.1M]  P01_19

    28 directories
    ```
4. Gulp the dataset (we supply a labels pkl for just p01) (RGB P01 FRAMES ONLY)

    ```bash
    $ python src/scripts/gulp_data \
        /path/to/rgb/frames (datasets/epic-100/frames) \
        datasets/epic-100/gulp/train \
        datasets/epic-100/labels/p01.pkl \
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

| Argument | Description | Default |
| - | - | - |
| `--val_features_pkl` | If you want to train / test on two distinct frame feature sets rather than using a train/test split | `None` |
| `--train-test-split` | Specify a train/test split between 0 and 1 | 0.3 |
| `--min-frames` | Minimum number of frames to train models for | 1 |
| `--max-frames` | Maximum number of frames to train models for (these two arguments can also be used in case of a training crash) | 8 |
| `--epoch`| How many iterations to train the models for | 200 |
| `--batch-size` | The size of the mini batches to feed to the model at a time | 512 |

# Computing ESVs

We compute ESVs using a collection of models, each which operate over a fixed-length input (e.g. TRN).

## Computing class priors

Regardless of whether your model supports variable-length inputs or not, we need to compute the class priors to use in the computation of the ESVs. We provide a script that does this by computing the emprical class frequency for both verb and nouns in p01.

```bash
$ python src/scripts/compute_verb_class_priors.py \
    datasets/epic-100/labels/p01.pkl \
    datasets/epic-100/labels/verb_class_priors.csv

$ python src/scripts/compute_noun_class_priors.py \
    datasets/epic-100/labels/p01.pkl \
    datasets/epic-100/labels/noun_class_priors.csv
```

## Models supporting a fixed-length input (e.g. TRN)

For models that don't support a variable-length input, we propose a way of ensembling a collection of fixed-length input models inot a new meta-model which we can compute ESVs for. To make this explanation more concrete, we now describe the process in detail for TRN. 

To start with, we train multiple TRN models for 1, 2, ..., n frames separately. By training these models separately we ensure that they are capable of acting alone (this also has the nice benefit of improving performance over joint training in our experience!). At inference time, we compute all possible subsampled variants of the input video we wish to classify and pass each of these through the corresponding single scale model. We aggregate scores for verbs/nouns so that each scale is given equal weighting in the final result.

This is implemented in the [`OnlineShapleyAttributor`](src/attribution/online_shapley_value_attributor.py#L17) class taken from [play-fair](https://github.com/willprice/play-fair).

We provide an example of how to do this for TRN, as the basic variant only supports a fixed-length input. (make sure you've set up your environment, downloaded and prepped the dataset, downloaded the models, extracted the features and trained for 1 .. n verb and noun models first)

```bash
$ python src/scripts/compute_esvs.py \
    datasets/epic-100/features/p01_features.pkl \
    datasets/epic-100/models/ (path where all models for all frames for verbs and nouns are located) \
    datasets/epic-100/labels/verb_class_priors.csv \
    datasets/epic-100/labels/noun_class_priors.csv \
    datasets/epic-100/esvs/mtrn-esv-n_frames=8.pkl \
    --sample_n_frames 8
```

# Visualisation

We provide a dashboard to investigate model behaviour when we vary how many frames are fed to the model. This dashboard is powered by multiple sets of results produced by the `compute_esvs.py` script

## Computing multi-frame ESVs

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

## Dumping video files

before we can run the dashboard, we need to dump out the videos from the gulp directory as webm files (since we gulp the files, it alters the FPS). Watch out that you don't end up using the conda bundled ffmpeg which doesn't support VP9 encoding if you replace `.bin/ffmpeg` with `ffmpeg`, check which you are using by running `which ffmpeg`

```bash
$ mkdir datasets/epic-100/video_frames
$ python src/scripts/dump_frames_from_gulp_dir.py \
    datasets/epic-100/gulp/train \
    datasets/epic-100/video_frames

$ for dir in datasets/epic-100/video_frames; do
    if [[ -f "$dir/frame_000000.jpg" && ! -f "$dir.webm" ]]; then
        ./bin/ffmpeg \
        -r 8 \
        -i "$dir/frame_%06d.jpg" \
        -c:v vp9 \
        -row-mt 1 \
        -auto-alt-ref 1 \
        -speed 4 \
        -b:v 200k \
        "$dir.webm" -y
    fi
  done

$ mkdir datasets/epic-100/video_frames/videos
$ mv datasets/epic-100/video_frames/*.webm datasets/epic-100/video_frames/videos
```

## Extracting verb-noun links

while [play-fair](https://github.com/willprice/play-fair) for Something-Something-v2 only predicts a single class label, we are predicting a verb and a noun label separately. To make the dashboard easier to use we have to extract action sequence instances for all verb/noun combinations:

```bash
$ python src/scripts/extract_vert_noun_links.py \
    datasets/epic-100/gulp/train \
    datasets/epic-100/labels/verb_noun.pkl \
    datasets/epic-100/EPIC_100_verb_classes.csv \
    datasets/epic-100/EPIC_100_noun_classes.csv
    
$ python src/scripts/extract_vert_noun_links.py \
    datasets/epic-100/gulp/train \
    datasets/epic-100/labels/verb_noun_classes.pkl \
    datasets/epic-100/EPIC_100_verb_classes.csv \
    datasets/epic-100/EPIC_100_noun_classes.csv \
    --classes True
    
$ python src/scripts/extract_vert_noun_links.py \
    datasets/epic-100/gulp/train \
    datasets/epic-100/labels/verb_noun_classes_narration.pkl \
    datasets/epic-100/EPIC_100_verb_classes.csv \
    datasets/epic-100/EPIC_100_noun_classes.csv \
    --classes True \
    --narration-id True
```

## Running the dashboard

Now we can run the dashboard

```bash
$ python src/apps/esv_dashboard/visualise_esvs.py \
    mtrn-esv-min_n_frames\=1-max_n_frames\=8-epoch=200.pkl \
    datasets/epic-100/video_frames \
    datasets/epic/labels/EPIC_100_verb_classes.csv \
    datasets/epic/labels/EPIC_100_noun_classes.csv \
    datasets/epic/labels/
```

alternatively if you trained on different number of epochs or dumped the video frames to a different directory you can run the dashboard using the script:

```bash
$ bash ./dashboard.sh <epochs> <path-to-dumped-video-frames>
```
