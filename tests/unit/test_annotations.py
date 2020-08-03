import os
import glob
import shutil
import pandas as pd
from oid2yolo.annotations import split_last_df_image
from oid2yolo.annotations import convert_label_filter
from oid2yolo.annotations import apply_image_filter
from oid2yolo.annotations import select_from_oid_bbox_with_class_names
from oid2yolo.annotations import extract_selected_images
from oid2yolo.annotations import extract_images_from_tarballs_with_image_id
from oid2yolo.annotations import convert_path_of_images_to_labels
from oid2yolo.annotations import convert_label_name_to_class_idx
from oid2yolo.annotations import create_yolo_annotation
from oid2yolo.annotations import create_yolo_annotations

oid_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../oid')
oid_bbox_csv = oid_path + '/validation-annotations-bbox.csv'

oid_images_tarball = oid_path + '/val/validation.tar.gz'
image_files = ['val/f68b8956e765994b.jpg',
               'val/7a9a879301438d31.jpg',
               'val/166b07b41546ecf6.jpg',
               'val/52557c5def8add62.jpg',
               'val/d5288976319d7301.jpg',
               'val/66155125e42978fe.jpg',
               'val/a26fa53e85db9d4a.jpg',
               'val/d1608d8475fa6b99.jpg',
               'val/9dbcd789f6a946e6.jpg',
               'val/c3eca54610a3d11c.jpg']
oid_class_csv = oid_path + '/class-descriptions-boxable.csv'
class_names = ['Human face', 'Human head', 'Vehicle registration plate']
df_all_classes = pd.read_csv(oid_class_csv, names=['LabelName', 'ClassName'])
df_classes = df_all_classes[df_all_classes['ClassName'].isin(class_names)].copy()
df_classes['order'] = df_classes['ClassName'].apply(lambda x: class_names.index(x))
df_classes.sort_values('order', inplace=True)
df_classes.reset_index(drop=True, inplace=True)
df_classes.drop(columns='order', inplace=True)
extract_dir = oid_path + '/val'
image_filter = [
    ["Vehicle registration plate"],
    ["Car", "Human face"],
    ["Car", "Human head"],
    ["Vehicle", "Human face"],
    ["Vehicle", "Human head"]
]


def test_split_by_imageid():
    need_columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
    reader = pd.read_csv(oid_bbox_csv, header=0, usecols=need_columns, chunksize=10000)
    df = reader.get_chunk(1000)
    df_all, df_rest = split_last_df_image(df)
    assert len(df_all) > 0
    assert len(df_rest) == 10


def test_convert_label_filter():
    label_filter = convert_label_filter(image_filter, df_all_classes)
    assert label_filter[0] == ['/m/01jfm_']
    assert label_filter[1] == ['/m/0k4j', '/m/0dzct']
    assert label_filter[2] == ['/m/0k4j', '/m/04hgtk']
    assert label_filter[3] == ['/m/07yv9', '/m/0dzct']
    assert label_filter[4] == ['/m/07yv9', '/m/04hgtk']


def test_apply_image_filter():
    need_columns = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
    reader = pd.read_csv(oid_bbox_csv, header=0, usecols=need_columns, chunksize=10000)
    df = reader.get_chunk(10000)
    df_all, df_rest = split_last_df_image(df)
    label_filter = convert_label_filter(image_filter, df_all_classes)
    df = apply_image_filter(df_all, label_filter, df_classes["LabelName"])

    labels = set([label for label_list in label_filter for label in label_list])
    labels = labels - set(df_classes["LabelName"].values)
    df_none = df[df['LabelName'].isin(labels)]
    assert df_none.empty is True


def test_extract_from_oid_bbox_with_class_names():
    label_filter = convert_label_filter(image_filter, df_all_classes)
    new_bbox_csv = select_from_oid_bbox_with_class_names(oid_bbox_csv, label_filter, df_classes)
    assert os.path.isfile(new_bbox_csv)
    df_anno = pd.read_csv(new_bbox_csv, header=0)
    for column in ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]:
        assert column in df_anno.columns
    os.remove(new_bbox_csv)


def test_extract_selected_images():
    base_image_files = [os.path.basename(image_file) for image_file in image_files]
    len_image_files = len(base_image_files)
    num_of_images = extract_selected_images(oid_images_tarball, base_image_files, extract_dir)
    assert num_of_images == len_image_files


def test_extract_images_from_tarballs_with_image_id():
    image_ids = [os.path.splitext(os.path.basename(image_file))[0] for image_file in image_files]
    num_of_images = extract_images_from_tarballs_with_image_id(image_ids, oid_images_tarball, extract_dir)
    glob_image_files = glob.glob(extract_dir + "/**/*.jpg", recursive=True)
    assert num_of_images == len(glob_image_files)


def test_convert_path_of_images_to_labels():
    yolo_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../yolo')
    images_dir = yolo_path + '/images/train/'
    labels_dir = yolo_path + '/labels/train/'
    result_paths = convert_path_of_images_to_labels(images_dir)
    for result_path in result_paths:
        assert labels_dir in result_path


def test_convert_label_name_to_class_idx():
    label_name = '/m/01jfm_'
    class_idx = convert_label_name_to_class_idx(label_name, df_classes)
    assert class_idx == class_names.index('Vehicle registration plate')


def test_create_yolo_annotation():
    yolo_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../yolo')
    images_dir = yolo_path + '/images/val/'
    labels_path = convert_path_of_images_to_labels(images_dir)
    label_path = labels_path[0]
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    label_filter = convert_label_filter(image_filter, df_all_classes)
    new_bbox_csv = select_from_oid_bbox_with_class_names(oid_bbox_csv, label_filter, df_classes)
    df_oid_annotations = pd.read_csv(new_bbox_csv, header=0)
    image_id = "01877d79c356fad3"
    df_image = df_oid_annotations[df_oid_annotations['ImageID'] == image_id]

    num_of_annotation = create_yolo_annotation(label_path, df_image, df_classes)
    assert num_of_annotation > 0


def test_create_yolo_annotations():
    label_filter = convert_label_filter(image_filter, df_all_classes)
    new_bbox_csv = select_from_oid_bbox_with_class_names(oid_bbox_csv, label_filter, df_classes)
    yolo_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../yolo')
    images_dir = yolo_path + '/images/val/'
    labels_dir = yolo_path + '/labels/val/'
    labels_path = convert_path_of_images_to_labels(images_dir)
    create_yolo_annotations(labels_path, new_bbox_csv, df_classes)
    for label_path in labels_path:
        assert os.path.isfile(label_path)
    shutil.rmtree(labels_dir)
