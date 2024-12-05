#!/usr/bin/env python
#
import numpy as np

def likelihood(q,C,p):
    """
    This routine should return the log of the
    likelihood function: P(qi|q,C,p,X)
    evaluated for given values of q, C and p
    """
    from math import log, pi
    # Assume data is given as an array of tuples: (h, U_ch, sigma_Uch)
    data = [(2.0, 1.16828362427, 0.0001283), (np.sqrt(2), 1.16429173392, 0.0003982),
            (1.0, 1.16367827195, 0.0001282), (0.5, 1.16389876649, 0.0002336)]

    log_likelihood = 0.0
    for h, U_ch, sigma_Uch in data:
        model_value = q - C * h ** p
        log_likelihood += -0.5 * ((U_ch - model_value) / sigma_Uch) ** 2 - log(sigma_Uch * np.sqrt(2.0 * pi))
    
    return log_likelihood

    
