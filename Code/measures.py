# -*- coding: utf-8 -*-

#import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, signal
import brian2 as b2
import parameters
from parameters import *
import random
import collections
import pandas as pd
from scipy.spatial.distance import pdist

def calculate_isi(spikemon, n_dict, plot = False):
    '''This function calculates the interspike intervals'''
    isi_times = []
    
    data = pd.DataFrame(list(zip(spikemon.i[spikemon.t>50*ms], spikemon.t[spikemon.t>50*ms]/b2.ms)), columns = ['i', 't'])
    lname = list(n_dict.keys())
    num_layer = list(n_dict.values())
    
    n_layer = [0]
    n_layer.extend(num_layer)
    l_bins = np.cumsum(n_layer) # cumulative number of neurons by layer
    N = np.sum(n_layer)         # total number of neurons
    
    # grouping spiking times for each neuron
    keys,values = data.sort_values(['i','t']).values.T
    ukeys,index=np.unique(keys,True)
    arrays=np.split(values,index[1:])

    spk_neuron = pd.DataFrame({'i':range(0,N),'t':[[]]*N})
    spk_neuron.iloc[ukeys.astype(int),1] = arrays
    
    spk_neuron['layer'] = pd.cut(spk_neuron['i'], l_bins, labels=lname, right=False)
    data['layer'] = pd.cut(data['i'], l_bins, labels=lname, right=False)
    
    del keys, values, ukeys, index, arrays
    
    for i in range(len(lname)):
        dat = spk_neuron.loc[spk_neuron['layer']==lname[i]]
        isi = []
        isi = [np.diff(dat.t[l_bins[i]+j]) for j in range(len(dat))]
        isi_times.append(isi)
    
        if plot == True:
            isi_list = np.concatenate(isi).ravel().tolist()
            plt.figure()
            plt.hist(isi_list, bins = 50)
    
    return isi_times


def calculate_cv(isi_times, lname):
    '''This function calculates the coefficient of variation'''
    CV_list = []
    mean_CVs = []
    SD_CVs = []
    #n_sample = 25*parameters.num_cols
    for i in range(len(lname)):
        CV = [np.std(isi_times[i][j])/np.mean(isi_times[i][j]) if len(isi_times[i][j])>1 else np.nan\
            for j in range(len(isi_times[i]))]
        CV_list.append(CV)
        mean_CVs.append(np.nanmean(CV))
        SD_CVs.append(np.nanstd(CV))
    
    return CV_list, mean_CVs, SD_CVs


def calculate_firing_frequencies(neurons, spikemon, n_groups):
    ''' This function calculates the individual firing frequencies of neurons'''
    time_vector=np.arange(0, parameters.simulation_time/b2.ms + 0.1, 0.1)
    frequency_matrix = np.zeros([len(time_vector), neurons.N])
    times = spikemon.t[spikemon.t>50*ms]/b2.ms
    indices = spikemon.i[spikemon.t>50*ms]
    timestep = 0.0001 #s
    for i in range(len(time_vector)):
        time_mask = (np.round(times,2) == np.round(time_vector[i],2)) #& (times <= time_vector[i + 1]))
        indexbin = indices[time_mask]
        for j in range(neurons.N):
            num_count=indexbin.tolist().count(j)
            frequency_matrix[i][j] = num_count/timestep #np.random.rand(1) * idx
    
    mean_firing = np.mean(frequency_matrix, axis = 0)
    
    group_frequencies = []
    for n_group in n_groups:
        group_frequencies.append(mean_firing[n_group.start:n_group.stop])
        
    return mean_firing, group_frequencies


def count_connections(synapses_group, combinedEI = True):
    ''' This function creases histograms of the number of connections to each neuron'''
    synapses = np.array([])
    if combinedEI == True:
        for i in range(len(synapses_group)): #-1
            synapses = np.append(synapses, np.array(synapses_group[i].j))
            
    elif combinedEI == False:
        synapses = np.append(synapses, np.array(synapses_group.j))
   
    counter=collections.Counter(synapses) #+1
        
    plt.figure()
    plt.hist(counter.values(), 30, range=[0, 10000])
    print('Mean:', np.array(list(counter.values())).mean())
    plt.axvline(x=np.array(list(counter.values())).mean(), color='r')
    plt.xlabel('number of connections (degree)')
    plt.ylabel('number of neurons containting this \n many number of connections (frequency)')


def oscillation_peaks(ratemon_list, plot = False):
    ''' This function finds uses the fast fourier transform in the real domain to plot the frequency spectrum
    and calculate the main frequency in a rate monitor signal
    source: https://www.adamsmith.haus/python/answers/how-to-plot-a-power-spectrum-in-python
    '''
    colours = ['#8ca465','#dec47c', '#487a99', '#d19f7f', '#385f49', '#ae5b5e', '#2e4876', '#85588c']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    powerspec = np.zeros((8, 67))
    for i, ratemon in enumerate(ratemon_list):
        sample_rate = len(ratemon.rate)
        fourier_transform = np.fft.rfft(ratemon.rate)
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)
        frequency = np.linspace(0, sample_rate/2, len(power_spectrum))
        smooth_powerspectrum = scipy.ndimage.gaussian_filter1d(power_spectrum, 1.5)
        if plot == True:
            ax.plot(frequency[3:70], smooth_powerspectrum[3:70], color = colours[i])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #ax.set_yscale('log')
            #ax.set_ylim((0, 1e10))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectrum Density')
            #plt.savefig('Figures_RecentRun/power', dpi=300)
        powerspec[i, :] = smooth_powerspectrum[3:70]
    # array_index = scipy.signal.find_peaks(smooth_powerspectrum[10:20])
    # beta_freq = frequency[10 + array_index[0][0]]
    # beta_peak = smooth_powerspectrum[10 + array_index[0][0]]
    
    return powerspec

