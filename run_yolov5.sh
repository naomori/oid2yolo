#!/bin/bash
docker run --detach \
	-p 8888:8888 \
	-p 6006:6006 \
	--privileged \
	--gpus all \
	--shm-size=1g --ulimit memlock=-1 \
	-it \
    -v /home/naomori/PycharmProjects/oid2yolo:/workspace/oid2yolo \
    -v /home/naomori/PycharmProjects/yolov5:/workspace/yolov5 \
    -v /loft/open_images_dataset_v6:/workspace/oid \
	--hostname yolov5 \
	--name yolov5 \
    ultralytics/yolov5:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --port=6006 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.notebook_dir='/workspace'
