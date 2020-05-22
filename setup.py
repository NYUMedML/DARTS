from setuptools import setup, find_packages
from os.path import join as pjoin
from DARTS._version import __version__

requirements= open('requirements.txt').read().split()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='DARTS',
    version=__version__,
    description='DenseUnet-based Automatic Rapid brain Segmentation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='NYUMedML',
    author_email='Razavian Lab',
    url='https://github.com/NYUMedML/DARTS',
    license='GNU General Public License v3.0',
    install_requires=requirements,
    packages=find_packages(exclude=('archive','plots')),
    package_data={'models': [pjoin('models','*')]},
    scripts= ['DARTS/perform_pred.py'],
    python_requires='>=3')

