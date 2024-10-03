Introduction...

Installation
============
Installation can then be done via pip::

    pip install mlfinpy


For the sake of best practice, it is good to do this with a dependency manager. I suggest you
set yourself up with `poetry <https://github.com/sdispater/poetry>`_, then within a new poetry project
run:

.. code-block:: text

    poetry add mlfinpy

.. note::
    If any of these methods don't work, please `raise an issue
    <https://github.com/baobach/mlfinpy/issues>`_ with the 'packaging' label on GitHub



For developers
--------------

If you are planning on using Mlfinpy as a starting template for significant
modifications, it probably makes sense to clone the repository and to just use the
source code

.. code-block:: text

    git clone https://github.com/baobach/mlfinpy

Alternatively, if you still want the convenience of a global ``from mlfinpy import x``,
you should try

.. code-block:: text

    pip install -e git+https://github.com/baobach/mlfinpy.git

A Quick Example
===============
Example of using the package

Contents
========

.. toctree::
   :maxdepth: 2

   UserGuide
   FinancialDataStructure
   FeaturesEngineering
   Labelling
   Sampling

.. toctree::
   :maxdepth: 1
   :caption: Other information
   
   FAQ
   Roadmap
   Contributing
   About

Project principles and design decisions
=======================================

- It should be easy to swap out individual components of the optimization process
  with the user's proprietary improvements.
- Usability is everything: it is better to be self-explanatory than consistent.
- There is no point in portfolio optimization unless it can be practically
  applied to real asset prices.
- Everything that has been implemented should be tested.
- Inline documentation is good: dedicated (separate) documentation is better.
  The two are not mutually exclusive.
- Formatting should never get in the way of good code: because of this,
  I have deferred **all** formatting decisions to `Black
  <https://github.com/ambv/black>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
