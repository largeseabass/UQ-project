#!/usr/bin/env python
#
import numpy as np


def prior_U(q):
    """
    This routine should return the log of the 
    prior probability distribution: P(q|X)
    evaluated for the given value of q.
    """
    # Mean and standard deviation for q
    mu_q = 1.1627
    sigma_q = (0.05*mu_q/1.96)  # Adjust based on confidence interval given

    # Log of the normal distribution
    log_prior_q = -0.5 * ((q - mu_q) / sigma_q) ** 2 - np.log(sigma_q * np.sqrt(2.0 * np.pi))
    
    return log_prior_q

def prior_C(C):
    """
    This routine should return the log of the 
    prior probability distribution: P(C|X)
    evaluated for the given value of C.
    """
    mu_q = 1.1627
    # Standard deviation for C based on the problem constraints
    sigma_C = (0.005*mu_q/1.96)  # Adjust based on 0.5% of U_c divided by 1.96

    # Log of the normal distribution
    log_prior_C = -0.5 * (C / sigma_C) ** 2 - np.log(sigma_C * np.sqrt(2.0 * np.pi))
    
    return log_prior_C

def prior_p(p):
    """
    This routine should return the log of the 
    prior probability distribution: P(p|X)
    evaluated for the given value of p.
    """

    
    #Uniform prior: p is between 1 and 10
    if 1 <= p <= 10:
        log_prior_p = -np.log(10.0 - 1.0)  # log(1 / (upper_bound - lower_bound))
    else:
        log_prior_p = -20.0  # we chose it to be -10000.0 because - np.inf caused problems with the sampler

    return log_prior_p


#
# One should not have to edit the routine below
#
def prior(q,C,p):
    """
    This routine should return the log of the
    prior probability distribution: P(q,C,p|X)
    evaluated for the given values of q, C, p.
    """

    # for some reason the p guesses are sometimes negative, this 
    # patches that up
    if(p < 0):
        return -1.0 * np.inf

    return prior_U(q) + prior_C(C) + prior_p(p)
