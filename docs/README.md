<div align="center"><img src="https://github.com/medoidai/skrobot/raw/master/static/skrobot-logo.png" width="250px;" /></div>

-----------------

# skrobot Documentation

## How to setup the documentation environment?

Make sure you have [Python](https://www.python.org/), [Git](https://git-scm.com/) and [virtualenv](https://pypi.org/project/virtualenv/) installed.

### Clone the project's repository

```sh
$ git clone https://github.com/medoidai/skrobot.git
```

### Create virtual environment and install dependencies

```sh
$ cd skrobot/docs
$ virtualenv -p python venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Generate HTML documentation

```sh
make html
```

**Thank you!**