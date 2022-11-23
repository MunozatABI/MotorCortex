# -*- coding: utf-8 -*-
"""
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""

import matplotlib.pyplot as plt
import brian2 as b2
from brian2 import ms, pA
import numpy as np
from itertools import compress
#import parameters
import random
import scipy
from scipy.signal import find_peaks
import seaborn as sns
import re
import random
import pandas as pd
import igraph as ig


###################### Figure 1 ##########################################
#Network Plot
def network_plot(con_tab, nsyn_array, l_name):
    ''' If this function doesn't generate a plot, copy and paste it into the console'''
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']

    g = ig.Graph()

    g.add_vertices(len(l_name))

    g.vs["name"] = l_name
    
    connections = nsyn_array/np.linalg.norm(nsyn_array)

    con_tab['src']= con_tab['Source'] + con_tab['SourceType']
    con_tab['tgt']= con_tab['Target'] + con_tab['TargetType']

    con_tab['src_idx'] = [l_name.index(a) for a in con_tab["src"]]
    con_tab['tgt_idx'] = [l_name.index(b) for b in con_tab["tgt"]]

    g.add_edges(list(zip(con_tab['src_idx'].astype(int), con_tab['tgt_idx'].astype(int))))

    weights = [abs(float(re.sub('[*][a-z][A-Z]', '', w))) for w in con_tab['Weight']]
    weights = weights/np.linalg.norm(weights)
    connection_weights = []
    for i, r in con_tab.iterrows():
        src = str("".join([r.loc['Source'], r.loc['SourceType']]))
        tgt = str("".join([r.loc['Target'], r.loc['TargetType']]))
        connection_weights.append(connections[l_name.index(tgt)][l_name.index(src)])
    g.es["weights"] = weights * connection_weights
    g.es["origin"] = con_tab['src']

    color_dict = dict(zip(l_name, colours))

    layout = g.layout("kk")
    visual_style = {}
    visual_style["vertex_label"] = g.vs["name"]
    visual_style["vertex_label_size"] = 1000
    visual_style["vertex_label_color"] = 'black'
    visual_style["vertex_shape"] = 'rectangle'
    visual_style["vertex_size"] = 50
    visual_style["vertex_color"] = [color_dict[Layer] for Layer in g.vs["name"]]
    visual_style["edge_color"] = [color_dict[origin] for origin in g.es["origin"]]
    visual_style["edge_width"] = [w*500 for w in g.es["weights"]]
    visual_style["margin"] = 100
    g.es["curved"] = True
    ig.plot(g, layout = layout, vertex_frame_width = 0, **visual_style)


###################### Figure 2 ##########################################
#Connectivity Plot
def matrix_connectivity_nsyn(l_name, nsyn_array, annot=False):
   # my_cmap = sns.color_palette("Spectral", as_cmap=True)
    plt.figure()
    ax = sns.heatmap(nsyn_array, linewidth=0.5, yticklabels = l_name, xticklabels = l_name, cmap = "Spectral", annot = annot, cbar_kws={'label': 'Number of Connections'})
    ax.set(xlabel='Source Group', ylabel='Target Group')
    plt.savefig('../Figures/connectivity.png', dpi=300)


###################### Figure 3 ##########################################
#Single Neuron Connectivity Plot
def single_neuron_connectivity(neuron_num, neurons, synapses, plot = '3D'):
   
    idx_src = np.where(synapses[0].i == neuron_num)
    tgts = synapses[0].j[idx_src[0]]
    
    x_ori = [neurons[neuron_num].X]
    y_ori = [neurons[neuron_num].Y]
    z_ori = [neurons[neuron_num].Z]
   
    x_vals = [neurons[k].X[0]*1000 for k in tgts] #flat_tgt
    y_vals = [neurons[k].Y[0]*1000 for k in tgts]
    z_vals = [neurons[k].Z[0]*1000 for k in tgts]
   
    if plot == '3D':
        #### 3D Connectivity Plot ####
         fig = plt.figure(figsize=(10,8))
         ax = fig.add_subplot(111, projection='3d')
         ax.scatter(x_vals, y_vals, z_vals, color = 'k', marker='o', alpha = 0.1)
         for x, y, z in zip(x_vals, y_vals, z_vals):
            ax.plot([x_ori[0][0]*1000 , x], [y_ori[0][0]*1000 , y], [z_ori[0][0]*1000 , z], color ='k', alpha = 0.15)
         ax.scatter(x_ori[0][0]*1000, y_ori[0][0]*1000, z_ori[0][0]*1000, s = 100, color = 'red', marker='^', alpha = 1)
         ax.set_xlabel("Distance (mm)")
         plt.savefig('../Figures/single_neuron.png', dpi=300, bbox_inches='tight')
    return


###################### Figure 4 ##########################################
#Input Sensitivity Plot
def plot_inputsensitivity(datapath, index):
    '''
    plotting sensitivitiy to input
    data: (eg. FR_data in Data/FR_data_input.npy)
    index: (0 - 7 for each neuron group, 8 for average of population)
    '''
    data = np.load(datapath)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data[index],cmap = 'hsv', interpolation='gaussian', extent=([1, 25, 700, 3000]))
    ax.set_aspect("auto")
    ax.set_xlabel('Input Frequency (Hz)')
    ax.set_ylabel('Number of Excitatory Input Neurons')
    fig.colorbar(im)
    plt.savefig('../Figures/input_sensitivity.png', dpi=300, bbox_inches='tight')


###################### Figure 5 ##########################################
#Number of Neurons Sensitivity Plot
def num_analysis(datapath):
    '''
    df data frame in Data/number_sensitivity.csv
    Groups: all, 23E - 6I
    '''
    
    df = pd.read_csv(datapath, delimiter=' ', index_col=False)
    
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    
    df_noL6E = df[df.Group!='6E']
    
    fig1 = sns.relplot(x='NumNeurons', y='meanFR', hue='Group', kind='line', ci='sd', data=df_noL6E, palette = colours, legend = False, height = 6, aspect = 1)
    fig1.set_xlabels('Number of Neurons')
    fig1.set_ylabels('Mean Population Firing Rate (hertz)')
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.savefig('../Figures/meanFR', dpi=300)
    
    fig2 = sns.relplot(x='NumNeurons', y='meanCV', hue='Group', kind='line', ci='sd', data=df_noL6E, palette = colours, legend = False, height = 6, aspect = 1)
    fig2.set_xlabels('Number of Neurons')
    fig2.set_ylabels('Mean Irregularity (CV)')
    plt.tick_params(axis='x', which='major', labelsize=14)
    plt.tick_params(axis='y', which='major', labelsize=14)
    plt.savefig('../Figures/meanCV', dpi=300)

###################### Figure 6 ##########################################
#Spiking Behaviour - Raster & Frequency plots
#Raster Plot
def raster_plot(spikemon_list, n_groups, lname, filename=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111) 
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    for idx, spikemon in enumerate(spikemon_list):
        neuron_num_array = np.random.randint(n_groups[idx].N, size=int(n_groups[idx].N*0.1))
        n_indexes=np.empty(0)
        for neuron_num in neuron_num_array:
            n_indexes = np.concatenate((n_indexes, np.where(np.array(spikemon.i[spikemon.t>50*ms]) == neuron_num)[0]))
        neuron_index = np.take(np.array(spikemon.i[spikemon.t>50*ms]), n_indexes.astype(int))
        neuron_times = np.take(np.array(spikemon.t[spikemon.t>50*ms]), n_indexes.astype(int))
        ax.scatter(neuron_times, neuron_index+(n_groups[idx].start), marker = '.', c = colours[idx])
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    tick_values = []
    for k in range(len(n_groups)):
        tick_values.append(int(np.median([n_groups[k].start, n_groups[k].stop])))
    ax.set_yticks(tick_values)
    ax.set_yticklabels(lname)
    ax.set_ylabel('Neuron Group')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if filename==None:
        plt.savefig('../Figures/raster', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('../Figures/' + filename, dpi=300)

#Rates Plot
def rates_plot(ratemon_list, lname, filename=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    for i in range(len(ratemon_list)):
        plt.subplot(len(ratemon_list), 1, i+1).set_xticks([])
        #plt.title('Spontaneous Neuron Firing')
        plt.plot(ratemon_list[i].t[500:]/b2.ms, ratemon_list[i].smooth_rate(window = 'gaussian', width=0.1*ms)[500:]/b2.Hz, color = colours[i], label = lname[i])
        plt.axhline(y=np.mean(ratemon_list[i].smooth_rate(window = 'gaussian', width=0.1*ms)[500:]/b2.Hz), color = 'k' , linestyle = '--')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.annotate(lname[i], xy=(0.9,0.8),xycoords='axes fraction', fontsize=14)
        plt.tick_params(axis='y', which='major', labelsize=14)
        if i == 4:
            plt.ylabel('Frequency (hz)')
        if i == 7:
            plt.xlabel('Time')
    if filename==None:
        plt.savefig('../Figures/rates', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('../Figures/' + filename, dpi=300)
    
    
###################### Figure 7 ##########################################
#FR Box Plots
def FR_boxplot(group_frequencies, l_name):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    colours.reverse() 
    l_name.reverse()
    group_frequencies.reverse()
    
    fig, ax = plt.subplots()
    bp = ax.boxplot(group_frequencies, vert = 0, patch_artist = True, showfliers = True)
    ax.set_yticklabels(l_name)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Firing Rates (Hz)')
    ax.set_ylabel('Neuron Groups')
    
    for patch, color in zip(bp['boxes'], colours):
        patch.set_facecolor(color)
    
    for median in bp['medians']:
        median.set(color = 'black')
        
    plt.savefig('../Figures/FRs_box', dpi=300, bbox_inches='tight')
    
def CV_boxplot(CVs, l_name):
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    new_CVs = []
    for i in range(len(CVs)):
        CVslist = [x for x in CVs[len(CVs) - 1 - i] if not(pd.isnull(x))]
        new_CVs.append(CVslist)
        
    colours.reverse() 
    l_name.reverse()
    
    fig, ax = plt.subplots()
    bp = ax.boxplot(new_CVs, vert = 0, patch_artist = True)
    ax.set_yticklabels(l_name)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Irregularity (CV)')
    ax.set_ylabel('Neuron Groups')
    
    for patch, color in zip(bp['boxes'], colours):
        patch.set_facecolor(color)
    
    for median in bp['medians']:
        median.set(color = 'black')
        
    plt.savefig('../Figures/CVs_box', dpi=300, bbox_inches='tight')


###################### Figure 8 ##########################################
#Power Spectrum Plots
def power_spectrum_SD(datapath, l_name):
    '''
    Plot power spectrum 
    '''
    plt.rcParams['font.size'] = 20
    
    beta_table = pd.read_csv(datapath, delimiter=' ', index_col=False)
    
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    
    sns.set_palette(colours)
    
    fig = sns.relplot(x="Frequency", y="Power", hue="Group", kind="line", ci="sd", data=beta_table)
    fig.set_xlabels('Frequency (Hz)')
    fig.set_ylabels('Power')
    
    plt.savefig('../Figures/PowerSpectrum', dpi=300, bbox_inches='tight')

###################### Other ##########################################
#3D Spatial Plot
def spatial_plot(n_dict, n_groups, take_all = True):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    l_names = list(n_dict.keys())
    for i in range(len(n_groups)):       
        #sample from group
        if take_all == False:
            idx_take = np.random.randint(n_groups[i].N, size = int(n_groups[i].N * 0.1))
            
            neurons_X = np.take(np.array(n_groups[i].X), idx_take)
            neurons_Y = np.take(np.array(n_groups[i].Y), idx_take)
            neurons_Z = np.take(np.array(n_groups[i].Z), idx_take)
        
            ax.scatter(neurons_X*1000, neurons_Y*1000, neurons_Z*1000, label = l_names[i], s = 7, alpha = 0.7, color = colours[i])
        
        if take_all == True:
            ax.scatter(n_groups[i].X*1000, n_groups[i].Y*1000, n_groups[i].Z*1000, label = l_names[i], s = 7, alpha = 0.7, color = colours[i])
    ax.legend(markerscale=5)
    ax.set_xlabel('Distance (mm)')
    ax.set_ylabel('Distance (mm)')
    ax.set_zlabel('Distance (mm)')
    plt.savefig('../Figures/spatial3D', dpi=300)
    return
