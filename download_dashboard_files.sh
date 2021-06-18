#!/usr/bin/env bash

wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1f6DO6UxH1OqW8s6Efdgne81yqRhkULNR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1f6DO6UxH1OqW8s6Efdgne81yqRhkULNR" -O dashboard_files.tar && rm -rf cookies.txt
tar -xvf dashboard_files.tar
rm dashboard_files.tar
