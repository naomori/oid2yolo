#!/bin/bash
apt install vim
apt install htop
apt install tmux


cp /s3/scripts/.tmux.conf "${HOME}/.tmux.conf"
chmod 755 "${HOME}/.tmux.conf"

chmod 755 /content/yolov5/scripts/*.sh
( cd /content/yolov5/ || /bin/bash /content/yolov5/scripts/download_weights.sh )

rsync -ahv --progress /s3/runs-2020-08-05-epoch300/runs/exp0/weights/last.pt            /content/yolov5/weights/yolov5s-last-300.pt
rsync -ahv --progress /s3/runs-2020-08-10-yolov5m-epoch100/runs/exp0/weights/last.pt    /content/yolov5/weights/yolov5m-last-100.pt
rsync -ahv --progress /s3/runs-2020-08-11-yolov5m-epoch100+100/exp0/weights/last.pt     /content/yolov5/weights/yolov5m-last-200.pt
rsync -ahv --progress /s3/runs-2020-08-12-yolov5m-epoch100+100+100/exp0/weights/last.pt /content/yolov5/weights/yolov5m-last-300.pt
rsync -ahv --progress /s3/runs-yolov5x-2020-08-09-epoch50/runs/exp0/weights/last.pt     /content/yolov5/weights/yolov5x-last-50.pt

chmod 664 /content/yolov5/weights/*.pt

( cd /content/yolov5/models/ || /bin/bash /content/yolov5/scripts/change_nc.sh )

mkdir -p /content/yolov5/config
rsync -ahv --progress /s3/config/fdlpd_colab.yaml /content/yolov5/config
chmod 664 /content/yolov5/config/fdlpd_colab.yaml
