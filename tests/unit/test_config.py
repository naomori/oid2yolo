import os
from oid2yolo.config import parse_oid, parse_yolo


pwd_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../config')


def test_ok_parse_oid():
    oid_path = config_path + '/oid_ok.json'
    oid_obj = parse_oid(oid_path)
    assert isinstance(oid_obj, dict)
    assert oid_obj['images']['train'] == "/workspace/oid2yolo_proj/tests/oid/train/*.tar.gz"
    assert oid_obj['images']['val'] == "/workspace/oid2yolo_proj/tests/oid/val/*.tar.gz"
    assert oid_obj['images']['test'] == "/workspace/oid2yolo_proj/tests/oid/test/*.tar.gz"
    assert oid_obj['bbox']['train'] == "/workspace/oid2yolo_proj/tests/oid/train-annotations-bbox.csv"
    assert oid_obj['bbox']['val'] == "/workspace/oid2yolo_proj/tests/oid/validation-annotations-bbox.csv"
    assert oid_obj['bbox']['test'] == "/workspace/oid2yolo_proj/tests/oid/test-annotations-bbox.csv"
    assert oid_obj['class'] == "/workspace/oid2yolo_proj/tests/oid/class-descriptions-boxable.csv"
    assert oid_obj['image_filter'][0] == ["Vehicle registration plate"]
    assert oid_obj['image_filter'][1] == ["Car", "Human face"]
    assert oid_obj['image_filter'][2] == ["Car", "Human head"]
    assert oid_obj['image_filter'][3] == ["Vehicle", "Human face"]
    assert oid_obj['image_filter'][4] == ["Vehicle", "Human head"]


def test_ok_parse_yolo():
    yolo_path = config_path + '/fdlpd_ok.yaml'
    yolo_obj = parse_yolo(yolo_path)
    assert isinstance(yolo_obj, dict)
    assert yolo_obj['train'] == "/workspace/oid2yolo_proj/tests/yolo/images/train"
    assert yolo_obj['val'] == "/workspace/oid2yolo_proj/tests/yolo/images/val"
    assert yolo_obj['test'] == "/workspace/oid2yolo_proj/tests/yolo/images/test"
    assert yolo_obj['nc'] == 3
    assert yolo_obj['names'] == ['Human face', 'Human head', 'Vehicle registration plate']
