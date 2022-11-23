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
import measures as my_measures
import plotting as my_plots
import gc

''' Define Paramater to explore'''
firing_rate = np.linspace(1, 25, 25)
bge_neurons = np.linspace(700, 3000, 25)

#Storage
FR_data = np.zeros((9, len(firing_rate), len(bge_neurons)))
CV_data = np.zeros((9, len(firing_rate), len(bge_neurons)))

'''Set Up Run'''
#Create Neurons
for fr_idx, bg_freq in enumerate(firing_rate):
    for bg_idx in range(len(bge_neurons)):
        b2.start_scope() #restart workspace
        gc.collect()
        
        #Find new bg input nums    
        bg_layer = []            
        for i in range(len(params1.bg_layer)):
            if i%2 == 0:
                bg_layer.append(int(bge_neurons[bg_idx]))
            elif i%2 == 1:
                bg_layer.append(int(np.rint(bge_neurons[bg_idx] * 0.925)))
        
        #Print
        print('Running simulation:', (fr_idx), ', ', (bg_idx), 'with parameters')
        print('BG Firing Rate:', bg_freq)
        print('BG Num:', bg_layer)
                        
        cortex = cortexFunctions.cortex_neurons(params1.n_layer, params1.l_name, params1.eqs, 
                                                params1.tau_ref, threshold='v>theta', 
                                                reset =  params1.eqs_reset, reset_potential = params1.v_r)
        ### Create Neuron Groups ###   
        n_groups, n_dict = cortex.create_neurons(cols = True, n_cols = 9)
        ### Define Spatial arrangement ###
        cortex.spatial_loc(300, 300, 2300, 50, n_groups, n_dict, n_cols = 9)
        
        #####################  Define Input #####################
        syn_group = cortex.poission_input(n_groups, bg_layer, 'Ie', bg_freq, params1.w_ex)
        
        ##################### Create Synapses #####################
        ### Initialise Synapses ###
        cortex.init_synapses(params1.eqs_syn, params1.e_pre, params1.i_pre)
        ### Synapses ### ### ### ### ### ### ### ### ### ### ### ### ### Define connectivity type here ###
        syn_g, nsyn_array = cortex.create_synapses(n_dict, n_groups, params1.con_tab, 'random') ### 'random', 'spatial'
        syn_group.append(syn_g)

        statemon, spikemon, ratemon, spikemon_list, ratemon_list = cortex.run_model(n_groups, syn_group, params1.simulation_time)
        
        mean_rates = []
        for j, ratemon in enumerate(ratemon_list):
            mean_rates.append(np.mean(ratemon.rate[500:]))
            FR_data[j][len(bge_neurons) - 1 - bg_idx][fr_idx]=(np.mean(ratemon.rate[500:]))
        
        isi_times = my_measures.calculate_isi(spikemon, n_dict)
        CVs, meanCVs = my_measures.calculate_cv(isi_times, params1.l_name)
        
        for k, CV in enumerate(meanCVs):
            CV_data[k][len(bge_neurons) - 1 - bg_idx][fr_idx]=CV
            
        my_plots.raster_plot(spikemon_list[:-1], n_groups, filename='run'+str(fr_idx)+str(bg_idx))
        my_plots.rates_plot(ratemon_list[:-1], params1.l_name, filename='run'+str(fr_idx)+str(bg_idx))
        
np.save('Data/FR_data_Input', FR_data)
np.save('Data/CV_data_Input', CV_data)
