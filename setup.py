from setuptools import setup, find_packages

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as f: requirements = f.read().splitlines()

with open(path.join(here, 'README.md'), encoding='utf-8') as f: readme = f.read()

setup(
     name='sand',
     version='1.0.0',
     author="Medoid AI",
     description="Sand is a Python module for designing, running and tracking Machine Learning experiments / tasks for binary classification problems. It is built on top of scikit-learn framework.",
     long_description=readme,
     platforms=['any'],
     long_description_content_type='text/markdown',
     url="https://github.com/medoidai/sand",
     python_requires='>=3.6',
     install_requires=requirements,
     packages=find_packages(),
     classifiers=[
         "Programming Language :: Python",
         'Intended Audience :: Developers',
         "Operating System :: OS Independent",
         'License :: OSI Approved',
         'Topic :: Scientific/Engineering'
     ]
)
