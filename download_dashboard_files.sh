#!/usr/bin/env bash
set -eux

download() {
  wget "$1" -O "$2"
}

download "https://drive.google.com/file/d/1f6DO6UxH1OqW8s6Efdgne81yqRhkULNR/view?usp=sharing" dashboard_files.tar
tar -xvf dashboard_files.tar
rm dashboard_files.tar
