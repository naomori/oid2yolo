import os
from oid2yolo.config import parse_oid, parse_yolo


config_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../config')


def test_ok_parse_oid():
    oid_path = config_path + '/oid_ok.json'
    oid_obj = parse_oid(oid_path)
    assert isinstance(oid_obj, dict)
    assert oid_obj['images']['train'] == '../oid/train/*.tar.gz'
    assert oid_obj['images']['val'] == '../oid/val/*.tar.gz'
    assert oid_obj['images']['test'] == '../oid/test/*.tar.gz'
    assert oid_obj['bbox']['train'] == "../oid/oidv6-train-annotations-bbox.csv"
    assert oid_obj['bbox']['val'] == "../oid/validation-annotations-bbox.csv"
    assert oid_obj['bbox']['test'] == "../oid/test-annotations-bbox.csv"
    assert oid_obj['class'] == "../oid/class-descriptions-boxable.csv"


def test_ok_parse_yolo():
    yolo_path = config_path + '/fdlpd_ok.yaml'
    yolo_obj = parse_yolo(yolo_path)
    assert isinstance(yolo_obj, dict)
    assert yolo_obj['train'] == "../yolo/images/train"
    assert yolo_obj['val'] == "../yolo/images/val"
    assert yolo_obj['test'] == "../yolo/images/test"
    assert yolo_obj['nc'] == 3
    assert yolo_obj['names'] == ['Human face', 'Human head', 'Vehicle registration plate']
