"""Minimal setup file for oid2yolo project."""

from setuptools import setup, find_packages

setup(
    name='oid2yolo',
    version='0.1.0',
    license='BSD License',
    description='Convert Open Image Dataset to YOLO format',

    author='Naoki Morita',
    author_email='naoki.morita@gmail.com',
    url='https://github.com/naomori/yolov5_fdlpd.git',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    install_requires=['PyYAML'],

    entry_points={
        'console_scripts': [
            'oid2yolo = oid2yolo.cli:oid2yolo_cli',
        ]
    },
)
