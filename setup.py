from setuptools import setup, find_packages

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f: readme = f.read()

setup(
     name='skrobot',
     version='1.0.13',
     license='MIT',
     author="Medoid AI",
     author_email="info@medoid.ai",
     description="skrobot is a Python module for designing, running and tracking Machine Learning experiments / tasks. It is built on top of scikit-learn framework.",
     long_description=readme,
     platforms=['any'],
     download_url='https://github.com/medoidai/skrobot/archive/1.0.13.tar.gz',
     long_description_content_type='text/markdown',
     url="https://github.com/medoidai/skrobot",
     python_requires='>=3.6',
     install_requires=['featuretools==0.23.0',
                       'joblib==1.0.0',
                       'matplotlib==3.3.3',
                       'numpy==1.19.4',
                       'numpyencoder==0.3.0',
                       'pandas==1.2.0',
                       'plotly==4.14.1',
                       'scikit-learn==0.24.0',
                       'scikit-plot==0.3.7',
                       'stringcase==1.2.0'],
     packages=find_packages(),
     classifiers=[
         "Development Status :: 5 - Production/Stable",
         "Intended Audience :: Developers",
         "Intended Audience :: Education",
         "Intended Audience :: Science/Research",
         "Operating System :: OS Independent",
         "License :: OSI Approved :: MIT License",
         "Programming Language :: Python",
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "Topic :: Scientific/Engineering",
         "Topic :: Scientific/Engineering :: Artificial Intelligence",
         "Environment :: Console"]
)
