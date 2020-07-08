import sys
import json
import yaml


def check_oid(oid_obj):
    t1_keys = ['images', 'bbox', 'class']
    t2_keys = ['train', 'val', 'test']

    for t1_key in t1_keys:
        if t1_key not in oid_obj.keys():
            print(f"{t1_key} not in oid", file=sys.stderr)
            return False
        if t1_key == 'class':
            continue
        for t2_key in t2_keys:
            if t2_key not in oid_obj[t1_key].keys():
                print(f"{t2_key} not in {t1_key}", file=sys.stderr)
                return False
    return True


def parse_oid(oid_file):
    try:
        with open(oid_file) as f:
            oid_obj = json.load(f)
        if not check_oid(oid_obj):
            raise ValueError("parse error in json")
        return oid_obj
    except Exception as e:
        print('Exception occurred at parse_oid()')
        print(e, file=sys.stderr)
        sys.exit(1)


def check_yolo(yolo_obj):
    t1_keys = ['train', 'val', 'test', 'nc', 'names']

    for t1_key in t1_keys:
        if t1_key not in yolo_obj.keys():
            print(f"{t1_key} not in yolo", file=sys.stderr)
            return False
    if int(yolo_obj['nc']) > len(yolo_obj['names']):
        print(f"nc:{yolo_obj['nc']} is over length:{len(yolo_obj['names'])} of names", file=sys.stderr)
        return False
    return True


def parse_yolo(yolo_file):
    try:
        with open(yolo_file) as f:
            yolo_obj = yaml.safe_load(f)
        if not check_yolo(yolo_obj):
            raise ValueError("parse error in yaml")
        return yolo_obj
    except Exception as e:
        print('Exception occurred at parse_yolo()')
        print(e, file=sys.stderr)
        sys.exit(1)
