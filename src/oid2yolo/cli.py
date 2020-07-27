import time
import argparse
import logging
from oid2yolo.api import Oid2yolo
from functools import wraps


def measure_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}: {elapsed_time}[sec]")
        return result
    return wrapper


@measure_elapsed_time
def oid2yolo_initialize(args):
    return Oid2yolo(args.oid, args.yolo)


@measure_elapsed_time
def oid2yolo_shrink_bboxes(oid2yolo_obj):
    return oid2yolo_obj.shrink_bboxes()


@measure_elapsed_time
def oid2yolo_extract_images(oid2yolo_obj):
    num_of_extract_images = oid2yolo_obj.extract_images()
    print(f"a number of extracted images: {num_of_extract_images}")
    return num_of_extract_images


@measure_elapsed_time
def oid2yolo_create_annotations(oid2yolo_obj):
    file_num, anno_num = oid2yolo_obj.create_annotations()
    print(f"generate files: {file_num}, annotations: {anno_num}")
    return file_num


@measure_elapsed_time
def oid2yolo_split_validation(oid2yolo_obj, val_prop):
    train_num, val_num = oid2yolo_obj.split_validation(val_prop)
    print(f"train:{train_num}, validation:{val_num}")
    return train_num + val_num


@measure_elapsed_time
def oid2yolo_draw_bboxes(oid2yolo_obj, data_type):
    image_num = oid2yolo_obj.draw_bboxes(data_type)
    print(f"image files: {image_num}")
    return image_num


@measure_elapsed_time
def execute_command(oid2yolo_obj, command, args):
    if command == "convert":
        oid2yolo_shrink_bboxes(oid2yolo_obj)
        if args.skip_extract_images is False:
            oid2yolo_extract_images(oid2yolo_obj)
        oid2yolo_create_annotations(oid2yolo_obj)
    elif command == "split":
        oid2yolo_split_validation(oid2yolo_obj, args.val_prop)
    elif command == "confirm":
        oid2yolo_draw_bboxes(oid2yolo_obj, args.data_type)
    else:
        print(f"unknown command:{command}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--oid', default='./config/oid.json', help='OpenImagesDataset config file')
    p.add_argument('-y', '--yolo', default='./config/yolo.yaml.', help='YOLO format config file')
    p.add_argument('-c', '--command', default='convert')
    p.add_argument('-s', '--skip_extract_images', action='store_true', help='skip extract images')
    p.add_argument('-v', '--val_prop', default=0.3, help='proportion of validation')
    p.add_argument('-t', '--data_type', default='all', help='data type:{train,val,test} for confirm command')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    oid2yolo_obj = oid2yolo_initialize(args)
    execute_command(oid2yolo_obj, args.command, args)
    del oid2yolo_obj


if __name__ == '__main__':
    main()
