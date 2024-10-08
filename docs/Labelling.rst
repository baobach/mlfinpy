.. _labelling:

##############
Data Labelling
##############

The simplest approach to supvervised financial machine learning is to aim to predict the price of an instrument at some fixed horison.
But due to the nature of the data, this goal is practically unachievable. Another approach is to focus on the classification problem,
i.e. predict discretized returns. Classification problem with finite set of possible labels has a greater chan to obtain predictive
power than regression problem where the pool of outcome is virtually infinite.

This module implemented both the commonly used as well as a couple of interesting techniques for labeling financial data. The vast
majority of our users make use of the following labeling schemes (in a classification setting):

    * Raw Returns
    * Fixed Horizon
    * Triple-Barrier and Meta-labeling
    * Trend Scanning

.. _labelling-raw_returns:

Raw Returns
===========

Labeling data by raw returns is the most simple and basic method of labeling financial data for machine learning. Raw returns can
be calculated either on a simple or logarithmic basis. Using returns rather than prices is usually preferred for financial time series
data because returns are usually stationary, unlike prices. This means that returns across different assets, or the same asset
at different times, can be directly compared with each other. The same cannot be said of price differences, since the magnitude of the
price change is highly dependent on the starting price, which varies with time.

The simple return for an observation with
price :math:`p_t` at time :math:`t` relative to its price at time :math:`t-1` is as follows:

.. math::
    r_t = \frac{p_{t}}{p_{t-1}} - 1

And the logarithmic return is:

.. math::
    r_t = log(p_t) - log(p_{t-1})

The label :math:`L_t` is simply equal to :math:`r_t`, or to the sign of :math:`r_t`, if binary labeling is desired.

 .. math::
     \begin{equation}
     \begin{split}
       L_{t} = \begin{cases}
       -1 &\ \text{if} \ \ r_t < 0\\
       0 &\ \text{if} \ \ r_t = 0\\
       1 &\ \text{if} \ \ r_t > 0
       \end{cases}
     \end{split}
     \end{equation}

If desired, the user can specify a resampling period to apply to the price data prior to calculating returns. The user
can also lag the returns to make them forward-looking.

The following shows the distribution of logarithmic daily returns on Microsoft stock during the time period between January
2010 and May 2020.

.. figure:: media/Raw_returns_distribution.png
   :scale: 90 %
   :align: center
   :figclass: align-center
   :alt: raw returns image

   Distribution of logarithmic returns on MSFT.

Implementation
--------------

.. py:currentmodule:: mlfinpy.labeling.raw_return
.. automodule:: mlfinpy.labeling.raw_return
   :members:

Example
-------
Below is an example on how to use the raw returns labeling method.

.. code-block:: python

    import pandas as pd
    from mlfinpy.labeling import raw_return

    # Import price data
    data = pd.read_csv('../Sample-Data/stock_prices.csv', index_col='Date', parse_dates=True)

    # Create labels numerically based on simple returns
    returns = raw_returns(prices=data, lag=True)

    # Create labels categorically based on logarithmic returns
    returns = raw_returns(prices=data, binary=True, logarithmic=True, lag=True)

    # Create labels categorically on weekly data with forward looking log returns.
    returns = raw_returns(prices=data, binary=True, logarithmic=True, resample_by='W', lag=True)

|

------------------------------------

|

.. _labeling-fixed_time_horizon:

Fixed Time Horizon
==================

Fixed horizon labels is a classification labeling technique used in the following paper: `Dixon, M., Klabjan, D. and
Bang, J., 2016. Classification-based Financial Markets Prediction using Deep Neural Networks. <https://arxiv.org/abs/1603.08604>`_

Fixed time horizon is a common method used in labeling financial data, usually applied on time bars. The rate of return relative
to :math:`t_0` over time horizon :math:`h`, assuming that returns are lagged, is calculated as follows (M.L. de Prado, Advances in
Financial Machine Learning, 2018):

.. math::
    r_{t0,t1} = \frac{p_{t1}}{p_{t0}} - 1

Where :math:`t_1` is the time bar index after a fixed horizon has passed, and :math:`p_{t0}, p_{t1}`
are prices at times :math:`t_0, t_1`. This method assigns a label based on comparison of rate of return to a threshold :math:`\tau`

 .. math::
     \begin{equation}
     \begin{split}
       L_{t0, t1} = \begin{cases}
       -1 &\ \text{if} \ \ r_{t0, t1} < -\tau\\
       0 &\ \text{if} \ \ -\tau \leq r_{t0, t1} \leq \tau\\
       1 &\ \text{if} \ \ r_{t0, t1} > \tau
       \end{cases}
     \end{split}
     \end{equation}

To avoid overlapping return windows, rather than specifying :math:`h`, the user is given the option of resampling the returns to
get the desired return period. Possible inputs for the resample period can be found `here.
<https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
Optionally, returns can be standardized by scaling by the mean and standard deviation of a rolling window. If threshold is a pd.Series,
**threshold.index and prices.index must match**; otherwise labels will fail to be returned. If resampling
is used, the threshold must match the index of prices after resampling. This is to avoid the user being forced to manually fill
in thresholds.

The following shows the distribution of labels for standardized returns on closing prices of SPY in the time period from Jan 2008 to July 2016
using a 20-day rolling window for the standard deviation.

.. figure:: media/fixed_horizon_labels_example.png
   :scale: 100 %
   :align: center
   :figclass: align-center
   :alt: fixed horizon example

   Distribution of labels on standardized returns on closing prices of SPY.

Though time bars are the most common format for financial data, there can be potential problems with over-reliance on time bars. Time
bars exhibit high seasonality, as trading behavior may be quite different at the open or close versus midday; thus it will not be
informative to apply the same threshold on a non-uniform distribution. Solutions include applying the fixed horizon method to tick or
volume bars instead of time bars, using data sampled at the same time every day (e.g. closing prices) or inputting a dynamic threshold
as a pd.Series corresponding to the timestamps in the dataset. However, the fixed horizon method will always fail to capture information
about the path of the prices [Lopez de Prado, 2018].

.. tip::
   **Underlying Literature**

   The following sources describe this method in more detail:

   - **Advances in Financial Machine Learning, Chapter 3.2** *by* Marcos Lopez de Prado (p. 43-44).
   - **Machine Learning for Asset Managers, Chapter 5.2** *by* Marcos Lopez de Prado (p. 65-66).


Implementation
--------------

.. py:currentmodule:: mlfinpy.labeling.fixed_time_horizon
.. automodule:: mlfinpy.labeling.fixed_time_horizon
   :members:

Example
-------
Below is an example on how to use the Fixed Horizon labeling technique on real data.

.. code-block::

    import pandas as pd
    import numpy as np

    from mlfinpy.labeling import fixed_time_horizon

    # Import price data.
    data = pd.read_csv('../Sample-Data/stock_prices.csv', index_col='Date', parse_dates=True)
    custom_threshold = pd.Series(np.random.random(len(data)), index = data.index)

    # Create labels.
    labels = fixed_time_horizon(prices=data, threshold=0.01, lag=True)

    # Create labels with a dynamic threshold.
    labels = fixed_time_horizon(prices=data, threshold=custom_threshold, lag=True)

    # Create labels with standardization.
    labels = fixed_time_horizon(prices=data, threshold=1, lag=True, standardized=True, window=5)

    # Create labels after resampling weekly with standardization.
    labels = fixed_time_horizon(prices=data, threshold=1, resample_by='W', lag=True,
                                standardized=True, window=4)

|

------------------------------------

|

.. _labelling-tb_meta:

Triple-Barrier and Meta-Labelling
=================================

The primary labeling method used in financial academia is the fixed-time horizon method. While ubiquitous, this method
has many faults which are remedied by the triple-barrier method discussed below. The triple-barrier method can be
extended to incorporate meta-labeling which will also be demonstrated and discussed below.

Triple-Barrier Method
---------------------

The idea behind the triple-barrier method is that we have three barriers: an upper barrier, a lower barrier, and a
vertical barrier. The upper barrier represents the threshold an observation's return needs to reach in order to be
considered a buying opportunity (a label of 1), the lower barrier represents the threshold an observation's return needs
to reach in order to be considered a selling opportunity (a label of -1), and the vertical barrier represents the amount
of time an observation has to reach its given return in either direction before it is given a label of 0. This concept
can be better understood visually and is shown in the figure below taken from Advances in Financial Machine
Learning (`reference`_):

.. image:: media/triple_barrier.png
   :scale: 100 %
   :align: center

One of the major faults with the fixed-time horizon method is that observations are given a label with respect to a certain
threshold after a fixed interval regardless of their respective volatilities. In other words, the expected returns of every
observation are treated equally regardless of the associated risk. The triple-barrier method tackles this issue by dynamically
setting the upper and lower barriers for each observation based on their given volatilities.

.. _reference: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

Meta-Labeling
-------------

Advances in Financial Machine Learning, Chapter 3, page 50. Reads:

"Suppose that you have a model for setting the side of the bet (long or short). You just need to learn the size of that
bet, which includes the possibility of no bet at all (zero size). This is a situation that practitioners face regularly.
We often know whether we want to buy or sell a product, and the only remaining question is how much money we should risk
in such a bet. We do not want the ML algorithm to learn the side, just to tell us what is the appropriate size. At this
point, it probably does not surprise you to hear that no book or paper has so far discussed this common problem. Thankfully,
that misery ends here.""

I call this problem meta-labeling because we want to build a secondary ML model that learns how to use a primary exogenous model.

The ML algorithm will be trained to decide whether to take the bet or pass, a purely binary prediction. When the predicted
label is 1, we can use the probability of this secondary prediction to derive the size of the bet, where the side (sign) of
the position has been set by the primary model.

Meta-Labeling User Guide
^^^^^^^^^^^^^^^^^^^^^^^^

Binary classification problems present a trade-off between type-I errors (false positives) and type-II errors (false negatives).
In general, increasing the true positive rate of a binary classifier will tend to increase its false positive rate. The receiver
operating characteristic (ROC) curve of a binary classifier measures the cost of increasing the true positive rate, in terms of
accepting higher false positive rates.

.. image:: media/confusion_matrix.png
   :scale: 40 %
   :align: center

The image illustrates the so-called “confusion matrix.” On a set of observations, there are items that exhibit a condition
(positives, left rectangle), and items that do not exhibit a condition (negative, right rectangle). A binary classifier predicts
that some items exhibit the condition (ellipse), where the TP area contains the true positives and the TN area contains the true negatives.
This leads to two kinds of errors: false positives (FP) and false negatives (FN). “Precision” is the ratio between the TP area and
the area in the ellipse. “Recall” is the ratio between the TP area and the area in the left rectangle. This notion of recall
(aka true positive rate) is in the context of classification problems, the analogous to “power” in the context of hypothesis testing.
“Accuracy” is the sum of the TP and TN areas divided by the overall set of items (square). In general, decreasing the FP area comes at
a cost of increasing the FN area, because higher precision typically means fewer calls, hence lower recall. Still, there is some
combination of precision and recall that maximizes the overall efficiency of the classifier. The F1-score measures the efficiency of a
classifier as the harmonic average between precision and recall.

**Meta-labeling is particularly helpful when you want to achieve higher F1-scores**. First, we build a model that achieves high recall,
even if the precision is not particularly high. Second, we correct for the low precision by applying meta-labeling to the positives predicted
by the primary model.

Meta-labeling will increase your F1-score by filtering out the false positives, where the majority of positives have already been identified
by the primary model. Stated differently, the role of the secondary ML algorithm is to determine whether a positive from the primary (exogenous)
model is true or false. It is not its purpose to come up with a betting opportunity. Its purpose is to determine whether we should act or pass
on the opportunity that has been presented.

Meta-labeling is a very powerful tool to have in your arsenal, for four additional reasons:

**First**, ML algorithms are often criticized as black boxes. Meta-labeling allows you to build an ML system on top of a white box
(like a fundamental model founded on economic theory). This ability to transform a fundamental model into an ML model should make
meta-labeling particularly useful to “quantamental” firms.

**Second**, the effects of overfitting are limited when you apply metalabeling, because ML will not decide the side of your bet, only the size.

**Third**, by decoupling the side prediction from the size prediction, meta-labeling enables sophisticated strategy structures. For instance, consider
that the features driving a rally may differ from the features driving a sell-off. In that case, you may want to develop an ML strategy exclusively
for long positions, based on the buy recommendations of a primary model, and an ML strategy exclusively for short positions, based on the sell
recommendations of an entirely different primary model.

**Fourth**, achieving high accuracy on small bets and low accuracy on large bets will ruin you. As important as identifying good opportunities is to size them
properly, so it makes sense to develop an ML algorithm solely focused on getting that critical decision (sizing) right. We will retake this fourth point in
Chapter 10. In my experience, meta-labeling ML models can deliver more robust and reliable outcomes than standard labeling models.

Model Architecture
^^^^^^^^^^^^^^^^^^

The following image explains the model architecture. The **first** step is to train a primary model (binary classification).
**Second** a threshold level is determined at which the primary model has a high recall, in the coded example you will find that
0.30 is a good threshold, ROC curves could be used to help determine a good level. **Third** the features from the first model
are concatenated with the predictions from the first model, into a new feature set for the secondary model. Meta Labels are used
as the target variable in the second model. Now fit the second model. **Fourth** the prediction from the secondary model is combined
with the prediction from the primary model and only where both are true, is your final prediction true. I.e. if your primary model
predicts a 3 and your secondary model says you have a high probability of the primary model being correct, is your final prediction
a 3, else not 3.

.. image:: media/meta_labeling_architecture.png
   :scale: 70 %
   :align: center


Implementation
--------------
.. py:currentmodule:: mlfinpy.labeling.labeling

The following functions are used for the triple-barrier method which works in tandem with meta-labeling.

.. autofunction:: add_vertical_barrier

.. autofunction:: get_events

.. autofunction:: get_bins

.. autofunction:: drop_labels


Example
-------

Suppose we use a mean-reverting strategy as our primary model, giving each observation a label of -1 or 1.
We can then use meta-labeling to act as a filter for the bets of our primary model.

Assuming we have a pandas series with the timestamps of our observations and their respective labels given by the primary
model, the process to generate meta-labels goes as follows.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import mlfinpy as ml

   # Read in data
   data = pd.read_csv('FILE_PATH')

   # Compute daily volatility
   daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=50)

   # Apply Symmetric CUSUM Filter and get timestamps for events
   # Note: Only the CUSUM filter needs a point estimate for volatility
   cusum_events = ml.filters.cusum_filter(data['close'],
                                          threshold=daily_vol['2011-09-01':'2018-01-01'].mean())

   # Compute vertical barrier
   vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                        close=data['close'],
                                                        num_days=1)

Once we have computed the daily volatility along with our vertical time barriers and have downsampled our series using
the CUSUM filter, we can use the triple-barrier method to compute our meta-labels by passing in the side predicted by
the primary model.

.. code-block:: python

   pt_sl = [1, 2]
   min_ret = 0.005
   triple_barrier_events = ml.labeling.get_events(close=data['close'],
                                                  t_events=cusum_events,
                                                  pt_sl=pt_sl,
                                                  target=daily_vol,
                                                  min_ret=min_ret,
                                                  num_threads=3,
                                                  vertical_barrier_times=vertical_barriers,
                                                  side_prediction=data['side'])

As can be seen above, we have scaled our lower barrier and set our minimum return to 0.005.

Meta-labels can then be computed using the time that each observation touched its respective barrier.

.. code-block:: python

   meta_labels = ml.labeling.get_bins(triple_barrier_events, data['close'])

|

------------------------------------

|

.. _labeling-trend_scanning:

Trend Scanning
==============

.. image:: media/trend_scanning_plot.png
   :scale: 100 %
   :align: center

Trend Scanning is both a classification and regression labeling technique introduced by Marcos Lopez de Prado in the
following lecture slides: `Advances in Financial Machine Learning, Lecture 3/10`_, and again in his text book `Machine Learning for Asset Managers`_.

.. _Advances in Financial Machine Learning, Lecture 3/10: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419
.. _Machine Learning for Asset Managers: https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545

For some trading algorithms, the researcher may not want to explicitly set a fixed profit / stop loss level, but rather detect overall
trend direction and sit in a position until the trend changes. For example, market timing strategy which holds ETFs except during volatile
periods. Trend scanning labels are designed to solve this type of problems.

This algorithm is also useful for defining market regimes between downtrend, no-trend, and uptrend.

The idea of trend-scanning labels are to fit multiple regressions from time t to t + L (L is a maximum look-forward window)
and select the one which yields maximum t-value for the slope coefficient, for a specific observation.

.. tip::
    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend, upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for the trend, and bin.

Implementation
--------------

.. py:currentmodule:: mlfinpy.labeling.trend_scanning
.. automodule:: mlfinpy.labeling.trend_scanning
   :members:

Example
-------
.. code-block::

    import numpy as np
    import pandas as pd

    from mlfinpy.labeling import trend_scanning_labels

    eem_close = pd.read_csv('./test_data/stock_prices.csv', index_col=0, parse_dates=[0])
    # In 2008, EEM had some clear trends
    eem_close = eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]


    t_events = eem_close.index # Get indexes that we want to label
    # We look at a maximum of the next 20 days to define the trend, however we fit regression on samples with length >= 10
    tr_scan_labels = trend_scanning_labels(eem_close, t_events, look_forward_window=20, min_sample_length=10)
