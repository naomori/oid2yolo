#!/bin/bash

img=640
batch=16
epoch_num=50
model=yolov5x
resume=true
weights=./weights/yolov5x-last-250.pt


function get_train_cmd() {
    cmd="python3 train.py "
    cmd+=" --img ${img} --batch ${batch} --epochs ${epoch_num}"
    cmd+=" --data ./config/fdlpd_colab.yaml"
    cmd+=" --cfg ./models/${model}.yaml"
    cmd+=" --cache-images"
    if [ ${resume} ]; then
        cmd+=" --resume ${weights} --weights ${weights}"
    else
        cmd+=" --weights ${weights}"
    fi
    echo "${cmd}"
    return
}

function get_latest_exp_dir() {
    find_dir=$1
    latest_file=$(find "${find_dir}" -type f -name "last*.pt" -print0 | xargs -0 ls -t | head -n 1)
    latest_dir=${latest_file%/*}
    parent_dir=${latest_dir%/*}
    echo "${parent_dir}"
    return
}

cmd=$(get_train_cmd)
${cmd}

exp_dir=$(get_latest_exp_dir "./runs")
date=$(date -I)

rsync -ahv --progress "${exp_dir}" "/s3/runs-${date}-${model}-epoch${epoch_num}+250"

