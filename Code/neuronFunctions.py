# -*- coding: utf-8 -*-
"""
Functions to create and initialise neurons
@author: lmun373
"""

import brian2 as b2
import numpy as np
import cortexParameters as params1
#from cortexParameters import *
import re

#@profile
def create_neurongroup(l_name, n_layer, model_eqs, ref, reset_eqs = None, cols=False, num_cols = 1, method = 'linear'):
    '''
    This function is to create groups of neurons or the motor cortex model
    l_name : str, array
            name for each layer/group
    n_layer : int, array 
              of number of neurons in each layer
    model_eqs : str
                neuron model differential functions
    reset_eqs : str
                reset function
    ref : str or double
          refractory period (condition or in milliseconds)
    method : str
             numerical solver method (eg. 'linear', 'euler', 'rk4')
    
    '''
    if len(l_name) != len(n_layer):
        raise ValueError("Arguments l_name and n_layer must be of the same size")
    nn_cum = [0]
    nn_cum.extend(list(np.cumsum(n_layer*num_cols)))
    N = sum(n_layer*num_cols)
    
    neurons = b2.NeuronGroup(N, model_eqs, threshold='v>theta', reset=reset_eqs, \
                        method=method, refractory=ref)
    
    n_groups = [] # Create neuron group for each population
    for r in range(0, len(n_layer)):
        n_groups.append(neurons[nn_cum[r]*num_cols:nn_cum[r+1]*num_cols])
        
    n_dict = dict(zip(l_name, [x*num_cols for x in n_layer]))
    
    return neurons, n_groups, n_dict

#@profile
def initialise_neurons(neurons, model_type, n_groups = None, n_dict = None):
    '''
    This function is to initialise the parameters depending on the model used.
    neurons : brian2 NeuronGroup
    model_type : str
                 'Shimoura'
    n_dict : dictionary
             contains name of each group
    '''
        
    if model_type == 'LIF':
        neurons.v = '-58.0*mV + 10.0*mV*randn()'
        neurons.Ie = 0.0*b2.pA      # initial value for synaptic currents
        neurons.Ii = 0.0*b2.pA      # initial value for synaptic currents
        neurons.Iext = 0.0*b2.pA   # constant external current
    
    else:
        raise ValueError("Model type not defined")

#@profile
def _spatial_arrangement(numcol, pref, arr_type, n_layer):   
    ''' 
    Get X and Y values for neurons' spatial location in 
        alternating column structure
    '''
    X=[]
    Y=[]
    Xlength = 300
    Ylength = 300
    space = 50

    if arr_type == 'checkercolumn':
        dim = np.sqrt(numcol*2)
        num = n_layer
        
        for i in range(len(num)):
            
            for j in range(numcol): 
                
                for k in range(num[i]):                        
                        X.append(np.random.randint(Xlength) + 50*j%(dim) + (j%(dim)) * (Xlength+space))
                        Y.append(np.random.randint(Ylength)  + 350*(j%(2)) + np.floor(j/dim) * 2 * (Ylength+space))
                        
    if arr_type == 'randomcolumn':
        num = n_layer
        dim = np.sqrt(numcol)
        
        for i in range(len(num)):
            
            for j in range(numcol): 
                
                for k in range(num[i]):                        
                        X.append(np.random.randint(Xlength) + 50*j%(dim) + (j%(dim)) * (Xlength+space))
                        Y.append(np.random.randint(Ylength) + np.floor(j/dim) * (Ylength+space))

    return X, Y

#@profile
def initialise_spatial_loc(neurons, n_groups, n_dict, numcol, arr_type):
    n_layer = [int(x/numcol) for x in list(n_dict.values())[:-1]]
    X, Y = _spatial_arrangement(numcol, 0, arr_type, n_layer)
    X.extend(np.zeros(list(n_dict.values())[-1]))
    Y.extend(np.zeros(list(n_dict.values())[-1]))
    neurons.X = X * b2.um
    neurons.Y = Y * b2.um
    
    n_layer_grouped = []
    l_name = list(n_dict.keys())
    layers = ['2/3', '4', '5', '6']
    for l in layers:
        indices = [i for i, s in enumerate(l_name) if l in s]
        n_layer_grouped.append(np.array(n_layer)[indices].sum())
    
    Zlength = 2300
    thickness = []
    z_boundaries = [0]
    for i in range(len(n_layer_grouped)):
        thickness.append(Zlength*(n_layer_grouped[i]/sum(n_layer_grouped)))
    
    z_boundaries.extend(np.cumsum(thickness))

    midpoints = []
    for i in range(len(z_boundaries)-1):
        midpoints.append((z_boundaries[i]+z_boundaries[i+1])/2)
        
    if arr_type == 'randomcolumn':    
        for i, g in enumerate(list(n_dict.keys())[:-1]):
            if g in n_dict:
                n_groups[list(n_dict.keys()).index(g)].Z = np.random.uniform(z_boundaries[int(np.ceil((i+1)/2))], z_boundaries[int(np.floor((i)/2))], n_layer[i]*numcol) * b2.um
        n_groups[-1].Z = 0* b2.um
#    if arr_type == 'random':
#        Z = []
#        for i in range(len(n_layer)):
#            idx = layers.index(re.sub('[EI]', '', l_name[i]))
#            Z.extend(np.random.uniform(z_boundaries[idx], z_boundaries[idx+1], n_layer[i]))
#        neurons.Z = Z * b2.um
    