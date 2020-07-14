from setuptools import setup, find_packages

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as f: requirements = [o.strip() for o in f]

with open(path.join(here, 'README.md'), encoding='utf-8') as f: readme = f.read()

setup(
     name='sand',
     version='1.0.0',
     license='MIT',
     author="Medoid AI",
     description="Sand is a Python module for designing, running and tracking Machine Learning experiments / tasks. It is built on top of scikit-learn framework.",
     long_description=readme,
     platforms=['any'],
     long_description_content_type='text/markdown',
     url="https://github.com/medoidai/sand",
     python_requires='>=3.6',
     install_requires=requirements,
     packages=find_packages(),
     classifiers=[
         "Development Status :: 5 - Production/Stable",
         "Intended Audience :: Developers",
         "Intended Audience :: Education",
         "Intended Audience :: Science/Research",
         "Operating System :: OS Independent",
         "License :: OSI Approved :: MIT License",
         "Programming Language :: Python",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Topic :: Scientific/Engineering",
         "Topic :: Scientific/Engineering :: Artificial Intelligence"]
)
