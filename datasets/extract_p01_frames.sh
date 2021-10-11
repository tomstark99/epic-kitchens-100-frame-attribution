#!/bin/bash

if [ -z "$1" ]
then
    cd epic-100/frames/P01
else
    cd $1
fi
for f in *.tar; do
    echo extracting $f...
    d="$f"
    d_arr=(${d//./ })
    d_name=${d_arr[0]}
    mkdir "$d_name"
    (tar -xf "$f" -C "$d_name")
done
rm *.tar
echo DONE
