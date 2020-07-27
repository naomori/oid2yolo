#!/bin/bash
docker run --detach \
	-p 8888:8888 \
	-p 6006:6006 \
	--privileged \
	--gpus all \
	--shm-size=1g --ulimit memlock=-1 \
	-it \
    -v /home/naomori/PycharmProjects/oid2yolo_proj:/workspace/oid2yolo_proj \
    -v /home/naomori/PycharmProjects/open_images_dataset_v6:/workspace/open_images_dataset_v6 \
	--hostname yolov5 \
	--name yolov5 \
    ultralytics/yolov5:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password=''
