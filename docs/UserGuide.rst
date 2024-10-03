.. _user-guide:

##########
User Guide
##########

This is designed to be a practical guide, mostly aimed at users who are interested in a
quick way of optimally combining some assets (most likely stocks). However, when
necessary I do introduce the required theory and also point out areas that may be
suitable springboards for more advanced optimization techniques. Details about the
parameters can be found in the respective documentation pages (please see the sidebar).

For this guide, we will be focusing on mean-variance optimization (MVO), which is what
most people think of when they hear "portfolio optimization". MVO forms the core of
PyPortfolioOpt's offering, though it should be noted that MVO comes in many flavours,
which can have very different performance characteristics. Please refer to the sidebar
to get a feeling for the possibilities, as well as the other optimization methods
offered. But for now, we will continue with the standard Efficient Frontier.

PyPortfolioOpt is designed with modularity in mind; the below flowchart sums up the
current functionality and overall layout of PyPortfolioOpt.