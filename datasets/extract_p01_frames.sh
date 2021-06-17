#!/bin/bash

cd epic-100/frames/P01
for f in *.tar; do 
    d="$f"
    d_arr=(${d//./ })
    d_name=${d_arr[0]}
    mkdir "$d_name"
    (tar -xf "$f" -C "$d_name")
done
rm *.tar
cd ../../..
