<div align="center"><img src="https://github.com/medoidai/skrobot/raw/master/static/skrobot-logo.png" width="250px;" /></div>

-----------------

# skrobot's Documentation

## How to setup the documentation environment?

Make sure you have [Python](https://www.python.org/), [Git](https://git-scm.com/) and [virtualenv](https://pypi.org/project/virtualenv/) installed.

### Clone the project's repository

```sh
$ git clone https://github.com/medoidai/skrobot.git
```

### Create virtual environment and install dependencies

```sh
$ virtualenv -p python docs/venv
$ source docs/venv/bin/activate
$ pip install -r docs/requirements.txt
```

### Install skrobot

```sh
$ python setup.py install
```

## Generate Sphinx documentation

### HTML documentation

```sh
$ cd docs
$ make html
```

**Thank you!**