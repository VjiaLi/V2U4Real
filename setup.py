# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modified by: WeiJia Li <vjiali@stu.xmu.edu.cn>
# License: MIT

from os.path import dirname, realpath
from setuptools import setup, find_packages
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]


setup(
    name='V2U4Real',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/VjiaLi/V2U4Real.git',
    license='MIT',
    author='Weijia Li',
    author_email='vjiali@stu.xmu.edu.cn',
    description='An opensource pytorch framework for vehicle-to-uav cooperative detection',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
