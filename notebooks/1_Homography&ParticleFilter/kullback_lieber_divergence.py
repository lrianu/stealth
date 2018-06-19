#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 18 14:15:34 2017

@author: lrianu
"""

import numpy as np
""" Description of what KL divergence is. """

def discrete_kl_divergence(P, Q):
    """ Calculates the Kullback-Lieber divergence
    according to the discrete definition:
    sum [P(i)*log[P(i)/Q(i)]]
    where P(i) and Q(i) are discrete probability
    distributions. In this case the one """

    """ Epsilon is used here to avoid conditional code for
    checking that neither P or Q is equal to 0. """
    epsilon = 0.00001

    # To avoid changing the color model, a copy is made
    temp_P = P+epsilon
    temp_Q = Q+epsilon

    divergence=np.sum(temp_P*np.log(temp_P/temp_Q))
    return divergence