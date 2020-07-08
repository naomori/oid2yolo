"""Main API for oid2yolo project."""

from collections import namedtuple


# Oid2yolo element types : [oid_json: str, yolo_yaml: str]
Oid2yolo = namedtuple('Oid2yolo', ['oid_json', 'yolo_yaml'])
Oid2yolo.__new__.__defaults__ = (None, None)


# custom exceptions
class Oid2yoloException(Exception):
    """A tasks error has occurred."""


def


def extract_classes_from_oid(csv_file):
    extracted_csv_file = ""
    return extracted_csv_file