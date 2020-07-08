import os
from oid2yolo.api import Oid2yolo

config_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../config')


def test_asdict():
    oid_path = config_path + '/oid_ok.json'
    yolo_path = config_path + '/fdlpd_ok.yaml'
    oid2yolo_obj = Oid2yolo(oid_path, yolo_path)
    assert oid2yolo_obj
