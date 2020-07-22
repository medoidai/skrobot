from setuptools import setup, find_packages

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f: readme = f.read()

setup(
     name='skrobot',
     version='1.0.4',
     license='MIT',
     author="Medoid AI",
     author_email="info@medoid.ai",
     description="skrobot is a Python module for designing, running and tracking Machine Learning experiments / tasks. It is built on top of scikit-learn framework.",
     long_description=readme,
     platforms=['any'],
     download_url='https://github.com/medoidai/skrobot/archive/1.0.4.tar.gz',
     long_description_content_type='text/markdown',
     url="https://github.com/medoidai/skrobot",
     python_requires='>=3.6',
     install_requires=['scikit-learn==0.21.3',
                       'joblib==0.13.2',
                       'numpy==1.16.4',
                       'pandas==0.24.2',
                       'numpyencoder==0.1.0',
                       'plotly==4.3.0',
                       'stringcase==1.2.0',
                       'scikit-plot==0.3.7',
                       'matplotlib==3.1.0'],
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
         "Topic :: Scientific/Engineering",
         "Topic :: Scientific/Engineering :: Artificial Intelligence",
         "Environment :: Console"]
)
