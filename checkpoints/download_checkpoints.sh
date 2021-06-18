#!/usr/bin/env bash
set -eux

download() {
  wget "$1" -O "$2"
}

download "https://www.dropbox.com/s/l1cs7kozz3f03r4/trn_rgb.ckpt?dl=1" trn_rgb.ckpt
