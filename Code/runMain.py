# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:03:22 2022

@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""
import cortex as cortexFunctions
import parameters as params1
from parameters import *
import plotting as my_plt
import measures as my_measures

def main():
    ##################### Create neurons and neuron groups #####################  
    print("Creating Cortex Model")
    ### Initialise Neuron Model ###
    cortex = cortexFunctions.cortex_neurons(params1.n_layer, params1.l_name, params1.eqs, 
                                            params1.tau_ref, threshold='v>theta', 
                                            reset =  params1.eqs_reset, reset_potential = params1.v_r)
    ### Create Neuron Groups ###   
    n_groups, n_dict = cortex.create_neurons(cols = True, n_cols = 9)
    ### Define Spatial arrangement ###
    cortex.spatial_loc(300, 300, 2300, 50, n_groups, n_dict, n_cols = 9)
    
    print("Creating Synapses")
    
    #####################  Define Input #####################
    syn_group = cortex.poission_input(n_groups, params1.bg_layer, 'Ie', params1.bg_freq, params1.w_ex)
    
    ##################### Create Synapses #####################
    ### Initialise Synapses ###
    cortex.init_synapses(params1.eqs_syn, params1.e_pre, params1.i_pre)
    ### Synampes ### ### ### ### ### ### ### ### ### ### ### ### ### Define connectivity type here ###
    syn_g, nsyn_array = cortex.create_synapses(n_dict, n_groups, params1.con_tab, 'random') ### 'random', 'spatial'
    syn_group.append(syn_g)
    
    print("Running Model")
    statemon, spikemon, ratemon, spikemon_list, ratemon_list = cortex.run_model(n_groups, syn_group, params1.simulation_time)
    
    ##################### Generate Plots & Figures #####################
    ###Plotting random connectivity results - change suffix to spatial###
    print("Plotting")
    
    ### Plot Figure 1
    my_plt.network_plot(params1.con_tab, nsyn_array, list(n_dict.keys()))
    
    ### Plot Figure 2
    my_plt.matrix_connectivity_nsyn(list(n_dict.keys()), nsyn_array)
    
    ### Plot Figure 3
    my_plt.single_neuron_connectivity(1500, cortex.neurons, syn_g, plot = '3D')
    
    ### Plot Figure 4
    my_plt.plot_inputsensitivity('../Data/FR_data_Input_random.npy', 8) ###'_random', '_spatial'
    
    ### Plot Figure 5
    my_plt.num_analysis('../Data/number_sensitivity_random.csv') ###'_random', '_spatial'
    
    ### Plot Figure 6
    my_plt.raster_plot(spikemon_list, n_groups, list(n_dict.keys()))
    my_plt.rates_plot(ratemon_list, list(n_dict.keys()))
    
    ### Plot Figure 7
    # Calculate FRs
    mean_firing, group_frequencies = my_measures.calculate_firing_frequencies(cortex.neurons, spikemon, n_groups)
    my_plt.FR_boxplot(group_frequencies, list(n_dict.keys()))
    
    # Calculate CVs
    isi_times = my_measures.calculate_isi(spikemon, n_dict)
    CVs, meanCVs, SDCVs = my_measures.calculate_cv(isi_times, list(n_dict.keys()))
    my_plt.CV_boxplot(CVs, list(n_dict.keys()))

    ### Plot Figure 8
    my_plt.power_spectrum_SD('../Data/PowerSpectrum_random.csv', list(n_dict.keys()))

if __name__=="__main__":
    main()
 