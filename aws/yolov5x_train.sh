#!/bin/sh

img=640
batch=16
epoch_num=50
model=yolov5x
weight=yolov5x-last-50.pt

python3 train.py --img ${img} --batch ${batch} --epochs ${epoch_num} \
    --data ./config/fdlpd_colab.yaml \
    --cfg ./models/${model}.yaml \
    --weights ./weights/${weight} \
    --cache-images

date=`date -I`
rsync -ahv --progress ./runs /s3/runs-${date}-${model}-epoch100
