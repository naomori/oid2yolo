#!/bin/bash

function make_train_tarballs() {
    train_tarballs=()
    for i in {0..9}; do
        train_tarballs+=("train_${i}.tar.gz")
    done
    for i in {a..f}; do
        train_tarballs+=("train_${i}.tar.gz")
    done
}

make_train_tarballs
test_val_tarball=(
    "test.tar.gz"
    "val.tar.gz"
)
tarballs=(${train_tarballs[@]} ${test_val_tarball[@]})

tarball_dir=$1
for tarball in "${tarballs[@]}"; do
    tar -zxf "${tarball_dir}/${tarball}"
done
