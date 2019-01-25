#!/bin/bash

set -e

if test -z "$OUTPUT_DIR"
then
    # $OUTPUT_DIR=${1:-"./data"}
    OUTPUT_DIR="./data"
else
    OUTPUT_DIR=$OUTPUT_DIR
fi

echo "Downloading to ${OUTPUT_DIR}."

if [ -d "${OUTPUT_DIR}/small_parallel_enja" ]
then
    # echo "Exists"
    echo "Already exists"
else
    # echo "Does not exist"
    git clone https://github.com/odashi/small_parallel_enja.git "${OUTPUT_DIR}/small_parallel_enja"
fi
