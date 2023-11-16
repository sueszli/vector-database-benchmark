#!/bin/bash

while IFS= read -r line
do
    echo "cloning $line"
    git clone "$line" &
done < "repositories.csv"

wait
