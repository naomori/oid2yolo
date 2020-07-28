#!/bin/sh

epoch_num=500

python3 train.py --img 640 --batch 16 --epochs ${epoch_num} \
	--data ./config/fdlpd_colab.yaml \
	--cfg ./models/yolov5s.yaml \
	--weights ./weights/last.pt \
	--cache-images

date=`date -I`
rsync -ahv --progress ./runs /s3/runs-${date}-epoch${epoch_num}
