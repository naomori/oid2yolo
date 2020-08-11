#!/bin/sh

img=640
batch=32
epoch_num=10
model=yolov5m
weight=yolov5m.pt

python3 train.py --img ${img} --batch ${batch} --epochs ${epoch_num} \
    --data ./config/fdlpd_colab.yaml \
    --cfg ./models/${model}.yaml \
    --weights ./weights/${weight} \
    --cache-images

date=`date -I`
rsync -ahv --progress ./runs /s3/runs-${date}-${model}-epoch${epoch_num}
