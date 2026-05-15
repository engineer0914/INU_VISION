from setuptools import setup
import os
from glob import glob

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),
        (
            os.path.join('share', package_name, 'yolo_models', 'train1_05_04', 'weights'),
            glob('yolo_models/train1_05_04/weights/*')
        ),
        (
            os.path.join('share', package_name, 'yolo_models', '0128_train', 'weights'),
            glob('yolo_models/0128_train/weights/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orinagx',
    maintainer_email='orinagx@example.com',
    description='YOLO depth pose estimator package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_3d_node = vision.yolo_3d_node:main',
        ],
    },
)
