'''Installs cpp_patcher'''
from os import path
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        "npy_patcher",
        sorted(glob("src/*.cpp")),
        include_dirs=['./']
    ),
]


version = '1.0.0'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='npy-patcher',
    version=version,
    description='C++ based NumPy N-Dimesional patch extraction.',
    author='Matthew Lyon',
    author_email='matthewlyon18@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    ext_modules=ext_modules
)
