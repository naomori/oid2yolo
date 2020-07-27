import os
from oid2yolo.api import Oid2yolo

config_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../config')
oid_path = config_path + '/oid_ok.json'
yolo_path = config_path + '/fdlpd_ok.yaml'
labels_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../yolo/labels')


def test_create_instance():
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    assert oid2yolo_obj


def test_shrink_bboxes():
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    assert oid2yolo_obj.shrink_bboxes() is True
    assert os.path.isfile(oid2yolo_obj.annotation['train'])
    assert os.path.isfile(oid2yolo_obj.annotation['val'])
    assert os.path.isfile(oid2yolo_obj.annotation['test'])


def test_extract_images_with_type():
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    oid2yolo_obj.shrink_bboxes()
    num_of_images = oid2yolo_obj.extract_images_with_type('train')
    assert num_of_images > 0
    num_of_images = oid2yolo_obj.extract_images_with_type('val')
    assert num_of_images > 0
    num_of_images = oid2yolo_obj.extract_images_with_type('test')
    assert num_of_images > 0


def test_extract_images():
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    oid2yolo_obj.shrink_bboxes()
    num_of_images = oid2yolo_obj.extract_images()
    assert num_of_images > 0


def test_create_annotations():
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    oid2yolo_obj.shrink_bboxes()
    oid2yolo_obj.extract_images()
    file_num, anno_num = oid2yolo_obj.create_annotations()
    assert file_num > 0
    assert anno_num > 0


def test_draw_bbox():
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    image_path = "/workspace/oid2yolo_proj/yolo/images/val/1945f91b6f5311ed.jpg"
    oid2yolo_obj.draw_bbox(image_path)
