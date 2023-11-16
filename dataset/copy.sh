#!/bin/bash

src_dir=~/dev/CPP_FILES
target_dir=~/dev/vector-database-benchmark/dataset/cpp

while [ "$(ls -A $src_dir)" ]; do
    cd $src_dir

    # we can't upload everything at once, because of github's commit size limit
    for file in $(ls | head -1000)
    do
        mv "$file" "$target_dir" 
    done

    cd $target_dir
    git add .
    git commit -m "update"
    git push
done
