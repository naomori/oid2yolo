#!/bin/sh
apt install vim
apt install htop

rsync -ahv --progress /s3/yolo-nc2/test.tar.gz /content/yolov5/data/
rsync -ahv --progress /s3/yolo-nc2/val.tar.gz /content/yolov5/data/
rsync -ahv --progress /s3/yolo-nc2/train_*.tar.gz /content/yolov5/data/
chmod 660 /content/yolov5/data/*.tar.gz
tar -C /content/yolov5/data/ -zxf /content/yolov5/data/test.tar.gz
tar -C /content/yolov5/data/ -zxf /content/yolov5/data/val.tar.gz
tar -C /content/yolov5/data/ -zxf /content/yolov5/data/train_*.tar.gz

rsync -ahv --progress /s3/weights/yolov5s.pt /content/yolov5/weights/
rsync -ahv --progress /s3/weights/last.pt /content/yolov5/weights/
chmod 660 /content/yolov5/weights/*.pt

rsync -ahv --progress /s3/models/yolov5s.yaml /content/yolov5/models/
chmod 660 /content/yolov5/models/yolov5s.yaml

mkdir -p /content/yolov5/config
rsync -ahv --progress /s3/config/fdlpd_colab.yaml /content/yolov5/config
chmod 660 /content/yolov5/config/fdlpd_colab.yaml
