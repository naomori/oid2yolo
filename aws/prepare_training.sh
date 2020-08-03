#!/bin/bash
apt install vim
apt install htop


function extract_tarball() {
    tarball=$1
    rsync -ahv --progress /s3/yolo-nc2/${tarball} /content/yolov5/data/
    chmod 660 /content/yolov5/data/${tarball}
    tar -C /content/yolov5/data/ -zxf /content/yolov5/data/${tarball}
    rm -f /content/yolov5/data/${tarball}
}

function make_train_tarballs() {
    train_tarballs=()
    for i in {0..9}; do
        train_tarballs+=("train_${i}.tar.gz")
    done
    for i in {a..f}; do
        train_tarballs+=("train_${i}.tar.gz")
    done
}

make_train_tarballs
test_val_tarball=(
    "test.tar.gz"
    "val.tar.gz"
)
tarballs=(${train_tarballs[@]} ${test_val_tarball[@]})

for tarball in "${tarballs[@]}"; do
    extract_tarball ${tarball}
done

rsync -ahv --progress /s3/weights/yolov5s.pt /content/yolov5/weights/
rsync -ahv --progress /s3/weights/last.pt /content/yolov5/weights/
chmod 660 /content/yolov5/weights/*.pt

rsync -ahv --progress /s3/models/yolov5s.yaml /content/yolov5/models/
chmod 660 /content/yolov5/models/yolov5s.yaml

mkdir -p /content/yolov5/config
rsync -ahv --progress /s3/config/fdlpd_colab.yaml /content/yolov5/config
chmod 660 /content/yolov5/config/fdlpd_colab.yaml
