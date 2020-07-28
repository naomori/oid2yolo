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
from oid2yolo.annotations import extract_from_oid_bbox_with_class_names
from oid2yolo.annotations import extract_images_from_tarballs_with_image_id
from oid2yolo.annotations import convert_path_of_images_to_labels
from oid2yolo.annotations import create_yolo_annotations


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

    def multiprocess_annotations(self, func):
        futures = []
        with ProcessPoolExecutor(max_workers=len(self.annotation)) as executor:
            for anno_type in self.annotation.keys():
                future = executor.submit(func, anno_type)
                futures.append(future)
        return futures

    def shrink_bbox(self, anno_type):
        label_filter = convert_label_filter(self.image_filter, self.df_all_classes)
        annotation = extract_from_oid_bbox_with_class_names(self.oid_obj['bbox'][anno_type],
                                                            label_filter, self.df_classes)
        return anno_type, annotation

    def shrink_bboxes(self):
        futures = self.multiprocess_annotations(self.shrink_bbox)
        concurrent.futures.wait(futures, timeout=None)
        result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        for result in result_list:
            self.annotation[result[0]] = result[1]
        return True

    def extract_images_with_type(self, anno_type):
        df_anno = pd.read_csv(self.annotation[anno_type], header=0)
        image_ids = df_anno['ImageID'].unique()
        tarballs_path = self.oid_obj['images'][anno_type]
        extract_dir = self.yolo_obj[anno_type]
        num_of_images = extract_images_from_tarballs_with_image_id(image_ids, tarballs_path, extract_dir)
        return num_of_images

    def extract_images(self):
        futures = self.multiprocess_annotations(self.extract_images_with_type)
        concurrent.futures.wait(futures, timeout=None)
        num_of_extract_images = sum([future.result() for future in concurrent.futures.as_completed(futures)])
        return num_of_extract_images

    def create_yolo_annotations(self, annotation_type):
        labels_path = convert_path_of_images_to_labels(self.yolo_obj[annotation_type])
        file_num, annotation_num = create_yolo_annotations(labels_path, self.annotation[annotation_type],
                                                           self.df_classes)
        return file_num, annotation_num

    def create_annotations(self):
        file_total_num = 0
        annotation_total_num = 0
        futures = self.multiprocess_annotations(self.create_yolo_annotations)
        concurrent.futures.wait(futures, timeout=None)
        result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        for result in result_list:
            file_total_num += result[0]
            annotation_total_num += result[1]
        return file_total_num, annotation_total_num

    def split_validation(self, val_prop):
        train_images_dir = self.yolo_obj['train']
        train_images_path = [p for p in glob.glob(train_images_dir + '/**', recursive=True)
                             if re.search('.(jpg|jpeg|png|JPG|JPEG|PNG)$', p)]
        train_num = len(train_images_path)

        val_images_dir = self.yolo_obj['val']
        val_images_path = [p for p in glob.glob(val_images_dir + '/**', recursive=True)
                           if re.search('.(jpg|jpeg|png|JPG|JPEG|PNG)$', p)]
        val_num = len(val_images_path)

        total_num = train_num + val_num
        if total_num * val_prop < val_num:
            diff_num = val_num - int(total_num * val_prop)
            src_path = val_images_path
            dst_image_dir = train_images_dir
        elif total_num * val_prop > val_num:
            diff_num = int(total_num * val_prop) - val_num
            src_path = train_images_path
            dst_image_dir = val_images_dir
        else:
            print(f"no change")
            return len(train_images_path), len(val_images_path)

        random.seed(0)
        for src_image in random.sample(src_path, diff_num):
            src_label = re.sub('.(jpg|jpeg|png|JPG|JPEG|PNG)$', '.txt', src_image).replace('images', 'labels')
            dst_label_dir = dst_image_dir.replace('images', 'labels')
            shutil.move(src_image, dst_image_dir)
            shutil.move(src_label, dst_label_dir)

        if total_num * val_prop < val_num:
            return train_num+diff_num, val_num-diff_num
        else:
            return train_num-diff_num, val_num+diff_num

    def draw_bbox(self, image_path):
        bbox_columns = ['class_index', 'x_center', 'y_center', 'width', 'height']
        label_path = re.sub('.(jpg|jpeg|png|JPG|JPEG|PNG)$', '.txt', image_path).replace('images', 'labels')
        df_bboxes = pd.read_csv(label_path, delimiter=' ', names=bbox_columns)
        image = cv2.imread(image_path)
        y, x = image.shape[:2]
        for row in df_bboxes.itertuples():
            class_name = self.df_classes["ClassName"][row.class_index]
            x_min = int(x * (row.x_center - (row.width / 2)))
            x_max = int(x * (row.x_center + row.width / 2))
            y_min = int(y * (row.y_center - (row.height / 2)))
            y_max = int(y * (row.y_center + row.height / 2))
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255), thickness=2)
            cv2.putText(image, class_name, (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        bbox_image_path = image_path.replace('images', 'images_w_bboxes')
        os.makedirs(os.path.dirname(bbox_image_path), exist_ok=True)
        cv2.imwrite(bbox_image_path, image)

    def draw_bboxes(self, data_type):
        num_image_paths = 0
        for anno_type in self.annotation.keys():
            if data_type is not "all" and data_type is not anno_type:
                continue
            images_dir = self.yolo_obj[anno_type]
            images_path = [p for p in glob.glob(images_dir + '/**', recursive=True)
                           if re.search('.(jpg|jpeg|png|JPG|JPEG|PNG)$', p)]
            for image_path in images_path:
                self.draw_bbox(image_path)
            num_image_paths += len(images_path)
        return num_image_paths
