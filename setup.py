from setuptools import setup, find_packages

setup(
    name = 'statstools',
    version = '0.1.0',
    author = 'jung-benjamin',
    packages = find_packages(include = ['statstools', 'statstools.*'])
    )
