"""Main API for oid2yolo project."""
from oid2yolo.config import parse_oid, parse_yolo


class Oid2yolo:

    def __init__(self, oid_json, yolo_yaml):
        self.oid_obj = parse_oid(oid_json)
        self.yolo_obj = parse_yolo(yolo_yaml)
