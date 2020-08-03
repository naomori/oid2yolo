import time
import argparse
import logging
from oid2yolo.api import Oid2yolo
from functools import wraps


def measure_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        print(f"{func.__name__}: {elapsed_time}[sec]")
        return result
    return wrapper


@measure_elapsed_time
def oid2yolo_initialize(args):
    return Oid2yolo(args.oid, args.yolo)


@measure_elapsed_time
def oid2yolo_shrink_bbox_all(oid2yolo_obj, overwrite=False):
    return oid2yolo_obj.shrink_bbox_all(overwrite)


@measure_elapsed_time
def oid2yolo_extract_images(oid2yolo_obj, dry_run=False):
    num_of_extract_images = oid2yolo_obj.extract_images(dry_run)
    print(f"a number of extracted images: {num_of_extract_images}")
    return num_of_extract_images


@measure_elapsed_time
def oid2yolo_create_annotations(oid2yolo_obj, dry_run=False):
    file_num, annotation_num = oid2yolo_obj.create_annotations(dry_run)
    print(f"generate files: {file_num}, annotations: {annotation_num}")
    return file_num


@measure_elapsed_time
def oid2yolo_split_dataset(oid2yolo_obj, split_prop, dry_run=False):
    dataset, df_move = oid2yolo_obj.split_dataset(split_prop, dry_run)
    if dataset is None or df_move is None:
        return 0
    print(f"[train] before:{dataset['train']['before']}, after:{dataset['train']['after']}")
    print(f"[validation] before:{dataset['val']['before']}, after:{dataset['val']['after']}")
    print(f"[move] num:{len(df_move.index.unique())}")
    print(f"{df_move['class_name'].value_counts()}")
    print(f"{df_move['class_name'].value_counts(normalize=True)}")
    return len(df_move.index.unique())


@measure_elapsed_time
def oid2yolo_draw_bbox_all(oid2yolo_obj, data_type):
    image_num = oid2yolo_obj.draw_bbox_all(data_type)
    print(f"image files: {image_num}")
    return image_num


@measure_elapsed_time
def execute_command(oid2yolo_obj, command, args):
    if command == "shrink":
        oid2yolo_shrink_bbox_all(oid2yolo_obj, overwrite=True)
    elif command == "convert":
        oid2yolo_shrink_bbox_all(oid2yolo_obj, overwrite=args.overwrite_oid)
        if args.skip_extract_images is False:
            oid2yolo_extract_images(oid2yolo_obj, args.dry_run)
        oid2yolo_create_annotations(oid2yolo_obj, args.dry_run)
    elif command == "split":
        oid2yolo_split_dataset(oid2yolo_obj, [float(prop) for prop in args.split_prop.split(':')], args.dry_run)
    elif command == "confirm":
        oid2yolo_draw_bbox_all(oid2yolo_obj, args.data_type)
    else:
        print(f"unknown command:{command}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--oid', default='./config/oid.json', help='OpenImagesDataset config file')
    p.add_argument('-y', '--yolo', default='./config/yolo.yaml.', help='YOLO format config file')
    p.add_argument('-c', '--command', default='convert', help='shrink, convert, split, or confirm')
    p.add_argument('-d', '--dry_run', action='store_true', help='dry-run')
    p.add_argument('-s', '--skip_extract_images', action='store_true', help='skip extract images')
    p.add_argument('-w', '--overwrite_oid', action='store_true', help='overwrite oid')
    p.add_argument('-p', '--split_prop', default="0.8:0.2", help='split train and validation set')
    p.add_argument('-t', '--data_type', default='all', help='data type:{train,val,test} for confirm command')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    oid2yolo_obj = oid2yolo_initialize(args)
    execute_command(oid2yolo_obj, args.command, args)
    del oid2yolo_obj


if __name__ == '__main__':
    main()
