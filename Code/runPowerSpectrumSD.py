# -*- coding: utf-8 -*-
"""
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""

'''Import Packages'''
import parameters as params1
from parameters import *
import cortex as cortexFunctions
import brian2 as b2
import numpy as np
import measures
import plotting as my_plots
import gc
import pandas as pd


#Storage
numruns = 10
freq = np.zeros((numruns, 8, 67))

'''Set Up Run'''
#Create Neurons
for i in range(numruns):
    print('Run ', i, '/', numruns)
    b2.start_scope() #restart workspace
    gc.collect()
    
    cortex = cortexFunctions.cortex_neurons(params1.n_layer, params1.l_name, params1.eqs, 
                                            params1.tau_ref, threshold='v>theta', 
                                            reset =  params1.eqs_reset, reset_potential = params1.v_r)
    ### Create Neuron Groups ###   
    n_groups, n_dict = cortex.create_neurons(cols = True, n_cols = 9)
    ### Define Spatial arrangement ###
    cortex.spatial_loc(300, 300, 2300, 50, n_groups, n_dict, n_cols = 9)
    
    #####################  Define Input #####################
    syn_group = cortex.poission_input(n_groups, params1.bg_layer, 'Ie', params1.bg_freq, params1.w_ex)
    
    ##################### Create Synapses #####################
    ### Initialise Synapses ###
    cortex.init_synapses(params1.eqs_syn, params1.e_pre, params1.i_pre)
    ### Synapses ### ### ### ### ### ### ### ### ### ### ### ### ### Define connectivity type here ###
    syn_g, nsyn_array = cortex.create_synapses(n_dict, n_groups, params1.con_tab, 'random') ### 'random', 'spatial'
    syn_group.append(syn_g)

    statemon, spikemon, ratemon, spikemon_list, ratemon_list = cortex.run_model(n_groups, syn_group, params1.simulation_time)
    
    freq[i] = measures.oscillation_peaks(ratemon_list, plot = True)
    
frequency = np.arange(3, 70)
    
beta_table = pd.DataFrame(columns = ['Run', 'Group', 'Frequency', 'Power'])

for i in range(freq.shape[0]):
    for j in range(freq.shape[1]):
        for k in range(freq.shape[2]):
            data = [i, list(n_dict.keys())[j], frequency[k], freq[i][j][k]]
            beta_table.loc[len(beta_table)] = data
            
beta_table.to_csv('../Data/PowerSpectrum_random_2.csv', sep=' ', index = False)


