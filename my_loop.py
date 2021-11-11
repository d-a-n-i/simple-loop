# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:46:48 2021

@author: Dani Cohen
"""
## boilerplate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.spatial.distance import cdist
#import loop

# verify results using PyNomaly https://github.com/vc1492a/PyNomaly
#def external_loop(data, k = 20, l = 2): # k nearest neighbors, lambda
#    return loop.LocalOutlierProbability(data, extent=l, n_neighbors=k).fit().local_outlier_probabilities


def my_loop(data, k = 20, l = 2):
    ## distance matric and k nearset neighbors
    D = cdist(data, data)
    k_i = D.argpartition((0,k), axis = 0)[1:k+1] # indices
    k_d = np.take_along_axis(D, k_i, axis = 0) # distances
    ## actual calc
    pdist = l * np.sqrt(np.sum(np.square(k_d), axis=0) / k)
    PLOF = pdist / np.mean(pdist[k_i], axis=0) - 1
    nPLOF = l * np.sqrt(np.mean(np.square(PLOF)))
    return np.maximum(0, erf(PLOF/(nPLOF*np.sqrt(2))))

if __name__ == "__main__":
    ## get data
    data = np.genfromtxt('data.csv', delimiter=',')
    #data = data[:25,:]
    ## see data
    print(data.shape)
    plt.scatter(data[:,0], data[:,1])
    ## check with ready implementation    
    # ext_loop_result = external_loop(data)
    # plt.scatter(data[:,0], data[:,1], c=ext_loop_result)
    ## this implementation
    my_loop_result = my_loop(data)
    #plt.plot(ext_loop_result - my_loop_result)
    plt.scatter(data[:,0], data[:,1], c=my_loop_result)
    