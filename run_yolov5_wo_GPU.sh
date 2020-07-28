#!/bin/bash
docker run --detach \
	--privileged \
	--shm-size=1g --ulimit memlock=-1 \
	-it \
    -v /home/naomori/PycharmProjects/oid2yolo_proj:/workspace/oid2yolo_proj \
    -v /home/naomori/PycharmProjects/open_images_dataset_v6:/workspace/open_images_dataset_v6 \
    -v /home/naomori/PycharmProjects/yolov5:/workspace/yolov5 \
	--hostname oid2yolo \
	--name oid2yolo \
    ultralytics/yolov5:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.notebook_dir='/workspace'
