How do I install it?
====================

PyPI
----

.. code:: sh

   pip install skrobot

Graphviz
--------

If you want to export feature computation graphs using the argument ``export_feature_graphs`` in :class:`.DeepFeatureSynthesisTask` class, you need to install Graphviz.

Conda users:

.. code:: sh

    conda install python-graphviz

GNU/Linux:

.. code:: sh

    sudo apt-get install graphviz
    pip install graphviz

Mac OS:

.. code:: sh

    brew install graphviz
    pip install graphviz

Windows:

.. code:: sh

    conda install python-graphviz

Development Version
-------------------

The skrobot version on PyPI may always be one step behind; you can install the latest development version from the GitHub repository by executing

.. code:: sh

    pip install git+git://github.com/medoidai/skrobot.git

Or, you can clone the GitHub repository and install skrobot from your local drive via

.. code:: sh

    python setup.py install