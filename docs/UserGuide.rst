.. _user-guide:

##########
User Guide
##########

Data Preparation
================

In order to utilize the bar sampling methods presented below, our data must first be formatted properly.
Many data vendors will let you choose the format of your raw tick data files. We want to only focus on the following
3 columns: ``date_time``, ``price``, ``volume``. The reason for this is to minimise the size of the csv files and the
amount of time when reading in the files.

First import your tick data.

.. code-block:: python

   # Required Imports
   import numpy as np
   import pandas as pd

   data = pd.read_csv('data.csv')

The provided data is sourced from TickData LLC which provides software called TickWrite, to aid in the formatting of saved files.
This allows us to save csv files in the format date_time, price, volume. (If you don't use TickWrite then make sure to
pre-format your files)

Make sure to first do some pre-processing and then save the data to a ``.csv`` file or provide a ``pandas.DataFrame`` input.

.. code-block:: python

   # Don't convert to datetime here, it will take forever to convert
   # on account of the sheer size of tick data files.
   date_time = data['Date'] + ' ' + data['Time']
   new_data = pd.concat([date_time, data['Price'], data['Volume']], axis=1)
   new_data.columns = ['date', 'price', 'volume']


Initially, your instinct may be to pass an in-memory DataFrame object but the truth is when you're running the function
in production, your raw tick data csv files will be way too large to hold in memory. We used the subset 2011 to 2019 and
it was more than 25 gigs. It is for this reason that the mlfinpy package suggests using a file path to read the raw data
files from disk.

.. code-block:: python

   # Save to csv
   new_data.to_csv('FILE_PATH', index=False)

In this guide, we can use the provided sample ``tick_data`` using the ``dataset`` module:

.. code-block:: python

   from mlfinpy.datasets import (load_tick_sample, load_stock_prices, load_dollar_bar_sample)

   # Load sample tick data
   tick_df = load_tick_sample()

Transform Data
===============

The first implemented module is ``mlfinpy.data_struture`` module. The main idea is to transform the unstructure tick data to
a more structured data format such as ``tick_bars``, ``volume_bars``, ``dollar_bars``, etc. By doing so, we can restore the
normality in the return distribution of the asset. This is a crutial part to create a high predictive power ML model.

The ``data_structure`` module has several data structures to choose from. As recommended in the literature, we will use the
dollar bar data structure to transform the raw tick data since it is the most stable structure.

.. code-block:: python

   from mlfinpy.data_structure import standard_bars

   # Dollar Bars with threshold $50,000 per bar
   dollar = standard_bars.get_dollar_bars(tick_df, threshold=50_000)

The detail on how to use the ``data_strucutre`` module is here :ref:`data-structure`.

Fix-width Window Fracdiff (FFD)
===============================

Making time series stationary often requires stationary data transformations, such as integer differentiation. Transform the
data to create a *stationary* series can come with a cost of losing it's **memory**. The most important characteristic of a
financial timeseries is lost and the data is no longer hold predictive power.

According to Marcos Lopez de Prado: “If the features are not stationary we cannot map the new observation to a large
number of known examples”. The method proposed by Marcos Lopez de Prado aims to make data stationary while preserving as much
memory as possible, as it’s the memory part that has predictive power.

Fractionally differentiated features approach allows differentiating a time series to the point where the series is stationary,
but not over differencing such that we lose all predictive power.

.. code-block:: python

   from mlfinpy.util.frac_diff import frac_diff_ffd, plot_min_ffd

   # Deriving the fractionally differentiated features
   dollar_ffd = frac_diff_ffd(dollar.close, 0.5)

   # Plotting the graph to find the minimum d
   # Make sure the input dataframe has a 'close' column
   plot_min_ffd(dollar)

In the making process...

CUSUM Filter
============
Apply CUSUM filter to the FFD we obtained in the previous step.

In the making process...

Triple Barrier Method
=====================
Using triple barrier method with target = 2 * daily volatility and vertical barrier of 5 days.

In the making process...

Sample With Sequential Booststrap
===================================
Using the sequential bootstrap method and determine sample weights, return weight.

In the making process...

Train The Classifier
====================
Fit bagging classifiers of decision trees.

In the making process...
