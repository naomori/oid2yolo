from oid2yolo import Oid2yolo


def test_asdict():
    """_asdict() should return a dictionary."""
    t_oid2yolo = Oid2yolo('oid.json', 'yolo.yaml')
    t_dict = t_oid2yolo._asdict()
    expected = {'oid_json': 'oid.json',
                'yolo_yaml': 'yolo.yaml'}
    assert t_dict == expected

