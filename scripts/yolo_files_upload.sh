aws --profile=moritan --region=ap-northeast-1 s3 cp yolov5s.pt s3://open-images-dataset-yolo-format/weights/
aws --profile=moritan --region=ap-northeast-1 s3 cp last.pt s3://open-images-dataset-yolo-format/weights/

aws --profile=moritan --region=ap-northeast-1 s3 cp yolov5s.yaml s3://open-images-dataset-yolo-format/models/
aws --profile=moritan --region=ap-northeast-1 s3 cp fdlpd_colab.yaml s3://open-images-dataset-yolo-format/config/

aws --profile=moritan --region=ap-northeast-1 s3 cp prepare_training.sh s3://open-images-dataset-yolo-format/scripts/
aws --profile=moritan --region=ap-northeast-1 s3 cp yolo_train.sh s3://open-images-dataset-yolo-format/scripts/

