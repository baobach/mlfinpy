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

Transform Data
===============

The first implemented module is ``mlfinpy.data_struture`` module. The main idea is to transform the unstructure tick data to
a more structured data format such as ``tick_bars``, ``volume_bars``, ``dollar_bars``, etc. By doing so, we can restore the
normality in the return distribution of the asset. This is a crutial part to create a high predictive power ML model.

In the making process...

Fix-width Window Fracdiff (FFD)
===============================
Timeseries has memory, transformation such as returns, log-returns, etc. trying to find stationarity in the timeseries. 
This will strip away the memory which can have strong predictive power. This trade off is a dilemma.
Fix-width Window Fracdiff (FFD) approach shows that there is no need to give up all of the memory in order to gain stationarity.

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

