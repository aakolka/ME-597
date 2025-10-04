from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'task_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aryaman Akolkar',
    maintainer_email='aakolka@purdue.edu',
    description='Basic Publisher and Subscriber',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
        'talker = task_1.basic_pub:main',
        'listener = task_1.basic_sub:main',
        ],
    },
)
