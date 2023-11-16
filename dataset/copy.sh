#!/bin/bash

src_dir=~/dev/PYTHON_FILES
target_dir=~/dev/vector-database-benchmark/dataset/python

while [ "$(ls -A $src_dir)" ]; do
    cd $src_dir

    # we can't upload everything at once, because of github's commit size limit
    for file in $(ls | head -5000)
    do
        mv "$file" "$target_dir"
    done

    cd $target_dir
    git add .
    git commit -m "update"
    git push
done
