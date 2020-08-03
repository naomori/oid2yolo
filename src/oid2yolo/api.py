"""Main API for oid2yolo project."""
import os
import glob
import re
import shutil
import random
import cv2
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from oid2yolo.config import parse_oid, parse_yolo
from oid2yolo.annotations import convert_label_filter
from oid2yolo.annotations import select_from_oid_bbox_with_class_names
from oid2yolo.annotations import extract_images_from_tarballs_with_image_id
from oid2yolo.annotations import convert_path_of_images_to_labels
from oid2yolo.annotations import create_yolo_annotations
from oid2yolo.annotations import yolo_labels_to_dataframe


class Oid2yolo:

    def __init__(self, oid_json, yolo_yaml):
        self.oid_obj = parse_oid(oid_json)
        self.yolo_obj = parse_yolo(yolo_yaml)
        df_all_classes = pd.read_csv(self.oid_obj["class"], names=['LabelName', 'ClassName'])
        class_names = self.yolo_obj['names'][:self.yolo_obj['nc']]
        df_classes = df_all_classes[df_all_classes['ClassName'].isin(class_names)].copy()
        df_classes['order'] = df_classes['ClassName'].apply(lambda x: class_names.index(x))
        df_classes.sort_values('order', inplace=True)
        df_classes.reset_index(drop=True, inplace=True)
        df_classes.drop(columns='order', inplace=True)
        self.df_classes = df_classes
        self.df_all_classes = df_all_classes
        self.annotation = {'train': None, 'val': None, 'test': None}
        self.image_filter = self.oid_obj["image_filter"]

    def multiprocess_annotations(self, func, func_arg):
        futures = []
        with ProcessPoolExecutor(max_workers=len(self.annotation)) as executor:
            for dataset_type in self.annotation.keys():
                future = executor.submit(func, dataset_type, func_arg)
                futures.append(future)
        return futures

    def shrink_bbox(self, dataset_type, overwrite=False):
        label_filter = convert_label_filter(self.image_filter, dataset_type, self.df_all_classes)
        annotation = select_from_oid_bbox_with_class_names(self.oid_obj['bbox'][dataset_type], label_filter,
                                                           self.df_classes, overwrite)
        return dataset_type, annotation

    def shrink_bbox_all(self, overwrite=False):
        futures = self.multiprocess_annotations(self.shrink_bbox, overwrite)
        concurrent.futures.wait(futures, timeout=None)
        result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        for result in result_list:
            self.annotation[result[0]] = result[1]
        return True

    def extract_images_with_type(self, dataset_type, dry_run=False):
        df_annotation = pd.read_csv(self.annotation[dataset_type], header=0)
        image_ids = df_annotation['ImageID'].unique()
        tarballs_path = self.oid_obj['images'][dataset_type]
        extract_dir = self.yolo_obj[dataset_type]
        num_of_images = extract_images_from_tarballs_with_image_id(image_ids, tarballs_path, extract_dir, dry_run)
        return num_of_images

    def extract_images(self, dry_run=False):
        futures = self.multiprocess_annotations(self.extract_images_with_type, dry_run)
        concurrent.futures.wait(futures, timeout=None)
        num_of_extract_images = sum([future.result() for future in concurrent.futures.as_completed(futures)])
        return num_of_extract_images

    def create_yolo_annotations(self, dataset_type, dry_run=False):
        labels_all_path = convert_path_of_images_to_labels(self.yolo_obj[dataset_type])
        if dry_run is False:
            os.makedirs(os.path.dirname(labels_all_path[0]), exist_ok=True)
        file_num, annotation_num = create_yolo_annotations(labels_all_path, self.annotation[dataset_type],
                                                           self.df_classes, dry_run)
        return file_num, annotation_num

    def create_annotations(self, dry_run=False):
        file_total_num = 0
        annotation_total_num = 0
        futures = self.multiprocess_annotations(self.create_yolo_annotations, dry_run)
        concurrent.futures.wait(futures, timeout=None)
        result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        for result in result_list:
            file_total_num += result[0]
            annotation_total_num += result[1]
        return file_total_num, annotation_total_num

    def split_dataset(self, split_prop, dry_run=False):
        def get_images_info(key):
            dir_name = self.yolo_obj[key]
            path_name = [p for p in glob.glob(dir_name + '/**', recursive=True)
                         if re.search('.(jpg|jpeg|png|JPG|JPEG|PNG)$', p)]
            image_num = len(path_name)
            return {'before': image_num, 'dir': dir_name, 'path': path_name}

        split_dataset = ['train', 'val']
        dataset = {d_set: get_images_info(d_set) for d_set in split_dataset}
        total_num = sum([dataset[d_set]['before'] for d_set in split_dataset])
        train_prop, val_prop = split_prop
        train_fix_num = int(total_num * train_prop)
        val_fix_num = total_num - train_fix_num
        dataset['train']['after'] = train_fix_num
        dataset['train']['diff'] = train_fix_num - dataset['train']['before']
        dataset['val']['after'] = val_fix_num
        dataset['val']['diff'] = val_fix_num - dataset['val']['before']

        src_list = [dataset[key] for key in split_dataset if dataset[key]['diff'] < 0]
        dst_list = [dataset[key] for key in split_dataset if dataset[key]['diff'] > 0]
        if len(src_list) == 0 or len(dst_list) == 0:
            return None, None

        move_label_list = []
        random.seed(0)
        for src in src_list:
            for src_image in random.sample(src['path'], abs(src['diff'])):
                dst = random.choice([d for d in dst_list if d['diff'] > 0])
                src_label = re.sub('.(jpg|jpeg|png|JPG|JPEG|PNG)$', '.txt', src_image).replace('images', 'labels')
                if dry_run is False:
                    shutil.move(src_image, dst['dir'])
                    dst_path = shutil.move(src_label, dst['dir'].replace('images', 'labels'))
                    move_label_list.append(dst_path)
                else:
                    move_label_list.append(src_label)
                dst['diff'] -= 1

        df_move = yolo_labels_to_dataframe(move_label_list)
        df_move['class_name'] = df_move['class_index'].agg(lambda x: self.df_classes.loc[x, "ClassName"])

        return dataset, df_move

    def draw_bbox(self, image_path):
        bbox_columns = ['class_index', 'x_center', 'y_center', 'width', 'height']
        label_path = re.sub('.(jpg|jpeg|png|JPG|JPEG|PNG)$', '.txt', image_path).replace('images', 'labels')
        df_bbox = pd.read_csv(label_path, delimiter=' ', names=bbox_columns)
        image = cv2.imread(image_path)
        y, x = image.shape[:2]
        for row in df_bbox.itertuples():
            class_name = self.df_classes["ClassName"][row.class_index]
            x_min = int(x * (row.x_center - (row.width / 2)))
            x_max = int(x * (row.x_center + row.width / 2))
            y_min = int(y * (row.y_center - (row.height / 2)))
            y_max = int(y * (row.y_center + row.height / 2))
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255), thickness=2)
            cv2.putText(image, class_name, (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        bbox_image_path = image_path.replace('images', 'images_w_bbox')
        os.makedirs(os.path.dirname(bbox_image_path), exist_ok=True)
        cv2.imwrite(bbox_image_path, image)

    def draw_bbox_all(self, data_type):
        num_image_paths = 0
        for dataset_type in self.annotation.keys():
            if data_type is not "all" and data_type is not dataset_type:
                continue
            images_dir = self.yolo_obj[dataset_type]
            images_path = [p for p in glob.glob(images_dir + '/**', recursive=True)
                           if re.search('.(jpg|jpeg|png|JPG|JPEG|PNG)$', p)]
            for image_path in images_path:
                self.draw_bbox(image_path)
            num_image_paths += len(images_path)
        return num_image_paths
