#!/bin/sh

img=640
batch=32
epoch_num=10
oid2yolo_dir="/workspace/oid2yolo"
work_dir="/workspace/yolov5"

python train.py --img ${img} --batch ${batch} --epochs ${epoch_num} \
	--data ${oid2yolo_dir}/config/fdlpd_yolov5.yaml \
	--cfg ${work_dir}/models/yolov5s.yaml \
	--weights ${work_dir}/weights/yolov5s.pt \
	--cache-images

