import os
import sys
import tempfile
import glob
import tarfile
import re
import shutil
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from more_itertools import chunked

import pandas as pd


def split_last_df_image(df):
    df_image = df.groupby("ImageID")
    last_image_id = df.tail(1)["ImageID"]
    last_group = df_image.get_group(*last_image_id)
    last_group_len = len(last_group)
    if last_group_len < len(df):
        df.drop(df.tail(last_group_len).index, inplace=True)
    df_all = df
    df_rest = last_group
    return df_all, df_rest


def convert_label_filter(class_filter, dataset_type, df_all_classes):
    label_filter = []
    if dataset_type == "train" or dataset_type == "val":
        filter_type = "train_val"
    else:
        filter_type = "test"
    for classes in class_filter[filter_type]:
        labels = []
        for class_name in classes:
            df = df_all_classes[df_all_classes["ClassName"] == class_name]
            labels.append(*df["LabelName"])
        label_filter.append(labels)
    return label_filter


def image_filter_func(df, label_filter, required_labels):
    tf = False
    for filter_entry in label_filter:
        tf = set(filter_entry).issubset(list(df["LabelName"]))
        if tf is True:
            break
    if tf is True:
        return df[df['LabelName'].isin(required_labels)]
    return None


def apply_image_filter(df_all, label_filter, df_labels):
    return df_all.groupby("ImageID", as_index=False).apply(image_filter_func,
                                                           label_filter=label_filter,
                                                           required_labels=df_labels.values)


def select_from_oid_bbox_with_class_names(oid_bbox_csv, label_filter, df_classes, overwrite=False):
    oid_bbox_csv_base, oid_bbox_csv_ext = os.path.splitext(oid_bbox_csv)
    new_oid_bbox_csv = oid_bbox_csv_base + "_selected" + oid_bbox_csv_ext
    if os.path.exists(new_oid_bbox_csv) is True and overwrite is False:
        return new_oid_bbox_csv

    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        df_rest = None
        need_columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
        for index, df_bbox in enumerate(pd.read_csv(oid_bbox_csv, header=0, usecols=need_columns, chunksize=50000)):
            df, df_last = split_last_df_image(df_bbox)
            if df_rest is not None:
                df = pd.concat([df_rest, df])
            df_rest = df_last
            if index == 0:
                df = apply_image_filter(df, label_filter, df_classes["LabelName"])
                df.to_csv(new_oid_bbox_csv, header=True, index=False)
            else:
                future = executor.submit(apply_image_filter, df, label_filter, df_classes["LabelName"])
                futures.append(future)

    concurrent.futures.wait(futures, timeout=None)
    df_list = [future.result() for future in concurrent.futures.as_completed(futures)]
    [df.to_csv(new_oid_bbox_csv, header=False, index=False, mode='a') for df in df_list]
    return new_oid_bbox_csv


def extract_selected_images(tarball, image_files, extract_dir, dry_run):
    with tarfile.open(tarball, mode='r') as tar:
        members = [m for m in tar.getmembers() if os.path.basename(m.name) in image_files]
        if dry_run is False:
            with tempfile.TemporaryDirectory() as tmpdir:
                tar.extractall(path=tmpdir, members=members)
                [shutil.move(f"{tmpdir}/{m.name}", f"{extract_dir}/{os.path.basename(m.name)}") for m in members]
        [image_files.remove(os.path.basename(m.name)) for m in members]
        len_members = len(members)
    return len_members


def extract_images_from_tarballs_with_image_id(image_ids, tarballs_path, extract_dir, dry_run):
    image_files = [image_id + ".jpg" for image_id in image_ids]
    if dry_run is False:
        os.makedirs(extract_dir, exist_ok=True)
    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for tarball in glob.glob(tarballs_path):
            future = executor.submit(extract_selected_images, tarball, image_files, extract_dir, dry_run)
            futures.append(future)
    concurrent.futures.wait(futures, timeout=None)
    num_of_extract_images = sum([future.result() for future in concurrent.futures.as_completed(futures)])
    return num_of_extract_images


def convert_path_of_images_to_labels(images_dir):
    labels_path = [re.sub('.(jpg|jpeg|png|JPG|JPEG|PNG)$', '.txt', p).replace('images/', 'labels/')
                   for p in glob.glob(images_dir + '/**', recursive=True)
                   if re.search('.(jpg|jpeg|png|JPG|JPEG|PNG)$', p)]
    return labels_path


def convert_label_name_to_class_idx(label_name, df_classes):
    try:
        return df_classes[df_classes['LabelName'] == label_name].index[0]
    except Exception as e:
        print(f"Exception occurred at convert_label_name_to_class_idx({label_name})")
        print(e, file=sys.stderr)
        return -1


def convert_bbox_from_oid_to_yolo(df_image, df_classes):
    bbox_columns = ['class_index', 'x_center', 'y_center', 'width', 'height']
    class_index_list = []
    x_center_list = []
    y_center_list = []
    width_list = []
    height_list = []
    for row in df_image.itertuples(index=False):
        class_index = convert_label_name_to_class_idx(row.LabelName, df_classes)
        class_index_list.append(int(class_index))
        x_center_list.append(row.XMin + (row.XMax-row.XMin)/2)
        y_center_list.append(row.YMin + (row.YMax-row.YMin)/2)
        width_list.append((row.XMax-row.XMin))
        height_list.append((row.YMax-row.YMin))
    df_yolo = pd.DataFrame({'class_index': class_index_list,
                            'x_center': x_center_list, 'y_center': y_center_list,
                            'width': width_list, 'height': height_list},
                           columns=bbox_columns)
    return df_yolo


def create_yolo_annotation(labels_path, df_oid, df_classes, dry_run):
    total_len = 0
    for label_path in labels_path:
        image_id = os.path.splitext(os.path.basename(label_path))[0]
        df_image = df_oid[df_oid['ImageID'] == image_id]
        df_yolo = convert_bbox_from_oid_to_yolo(df_image, df_classes)
        if dry_run is False:
            df_yolo.to_csv(label_path, header=False, index=False, sep=' ')
        total_len += len(df_yolo)
    return total_len


def create_yolo_annotations(labels_all_path, oid_annotations_csv, df_classes, dry_run=False):
    df_oid = pd.read_csv(oid_annotations_csv, header=0)
    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for labels_path in chunked(labels_all_path, 500):
            future = executor.submit(create_yolo_annotation, labels_path, df_oid, df_classes, dry_run)
            futures.append(future)
    concurrent.futures.wait(futures, timeout=None)
    annotations_len = sum([future.result() for future in concurrent.futures.as_completed(futures)])
    return len(labels_all_path), annotations_len


def yolo_labels_to_dataframe(labels_path):
    df_all = pd.DataFrame(data=None)
    for label_path in labels_path:
        image_id = os.path.splitext(os.path.basename(label_path))[0]
        df = pd.read_csv(label_path, header=None, delimiter=' ',
                         names=['class_index', 'x_center', 'y_center', 'width', 'height'])
        df['image_id'] = image_id
        df = df.set_index('image_id')
        df_all = pd.concat([df_all, df])
    return df_all


def yolo_annotation_to_dataframe(labels_dir):
    labels_all_path = [p for p in glob.glob(labels_dir + '/**', recursive=True) if re.search('.txt', p)]
    futures = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for labels_path in chunked(labels_all_path, 500):
            future = executor.submit(yolo_labels_to_dataframe, labels_path)
            futures.append(future)
    concurrent.futures.wait(futures, timeout=None)
    df_list = [future.result() for future in concurrent.futures.as_completed(futures)]
    df_all = pd.DataFrame(data=None)
    for df in df_list:
        df_all = pd.concat([df_all, df])
    return df_all
