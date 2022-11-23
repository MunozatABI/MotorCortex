# -*- coding: utf-8 -*-
"""
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""

'''Import Packages'''
import parameters as params1
from parameters import *
import cortex as cortexFunctions
import measures as my_measures
import brian2 as b2
import numpy as np
import time
import changeConnTable
import pandas as pd


start_time = time.time()

#freq = np.zeros((10, 8, 67))

scale_factors = np.linspace(0.2, 2, 19)

df = pd.DataFrame(columns=['Run', 'NumNeurons', 'Group', 'meanFR', 'meanCV'])

for s in scale_factors: 
    n_layer_scaled = [int(round(n*s)) for n in params1.n_layer]
    
    changeConnTable.change_conn_table(n_layer_scaled, params1.l_name, '../Connection_Tables/cortex_spatial.csv', '../Connection_Tables/cortex_spatial_new.csv')
    
    new_con_tab = pd.read_csv('../Connection_Tables/cortex_spatial_new.csv', delimiter=' ', index_col=False)
    
    for i in range(10):
        '''Set Up Run'''
        b2.start_scope() #restart workspace
        #Create Neurons
        ### Initialise Neuron Model ###
        cortex = cortexFunctions.cortex_neurons(params1.n_layer, params1.l_name, params1.eqs, 
                                                params1.tau_ref, threshold='v>theta', 
                                                reset =  params1.eqs_reset, reset_potential = params1.v_r)
        
        print(f'*****Run {i+1} at scale {s}*****')
        ### Create Neuron Groups ###   
        n_groups, n_dict = cortex.create_neurons(cols = True, n_cols = 9)
        ### Define Spatial arrangement ###
        cortex.spatial_loc(300, 300, 2300, 50, n_groups, n_dict, n_cols = 9)
        
        #####################  Define Input #####################
        syn_group = cortex.poission_input(n_groups, params1.bg_layer, 'Ie', params1.bg_freq, params1.w_ex)
        
        ##################### Create Synapses #####################
        ### Initialise Synapses ###
        cortex.init_synapses(params1.eqs_syn, params1.e_pre, params1.i_pre)
        ### Synampes ### ### ### ### ### ### ### ### ### ### ### ### ### Define connectivity type here ###
        syn_g, nsyn_array = cortex.create_synapses(n_dict, n_groups, params1.con_tab, 'random') ### 'random', 'spatial'
        syn_group.append(syn_g)

        statemon, spikemon, ratemon, spikemon_list, ratemon_list = cortex.run_model(n_groups, syn_group, params1.simulation_time)
        
        '''Store data'''
        #Calculate mean Rates    
        mean_rates = [np.mean(ratemon.rate[500:])]
        for ratemon in ratemon_list:
            mean_rates.append(np.mean(ratemon.rate[500:]))
            
        isi_times = my_measures.calculate_isi(spikemon, n_dict)
        CV_list, meanCVs, SD_CVs = my_measures.calculate_cv(isi_times, list(n_dict.keys()))
        
        a = ['all']
        a.extend(params1.l_name)
        
        mean_CV_all = [np.nanmean(meanCVs)]
        mean_CV_all.extend(meanCVs)
        mean_CV_all = np.round(mean_CV_all, 4)
        
        mean_rates = np.round(mean_rates/b2.hertz, 4)
        
        new_data = {'Run': [i]*9,
                'NumNeurons': [cortex.neurons.N] * 9, 
                'Group': a,
                'meanFR': mean_rates,
                'meanCV': mean_CV_all}
        
        df = df.append(pd.DataFrame(new_data), ignore_index=True)
        
df.to_csv('Data/number_sensitivity.csv', sep=' ', index = False)




    