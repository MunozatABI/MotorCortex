# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:00:32 2022

@author: lmun373
"""
import numpy as np
import brian2 as b2
import collections
import random
import parameters as params1
from parameters import *

def _find_p(src, tgt, n_dict, n_groups, radius, p): #
    
    l_name = list(n_dict.keys())
    
    array_X_src = np.ones((n_dict[src], n_dict[tgt]), dtype='float32') * np.array([n_groups[l_name.index(src)].X], dtype='float32').T
    array_X_tgt = np.ones((n_dict[src], n_dict[tgt]), dtype='float32') * np.array([n_groups[l_name.index(tgt)].X], dtype='float32')
    
    array_Y_src = np.ones((n_dict[src], n_dict[tgt]), dtype='float32') * np.array([n_groups[l_name.index(src)].Y], dtype='float32').T
    array_Y_tgt = np.ones((n_dict[src], n_dict[tgt]), dtype='float32') * np.array([n_groups[l_name.index(tgt)].Y], dtype='float32')
    
    p[:] = 1 * np.exp(-(np.add(np.power(np.subtract(array_X_src, array_X_tgt), 2), np.power(np.subtract(array_Y_src, array_Y_tgt), 2)))/(2*radius**2))
    
    np.fill_diagonal(p, 0) #prevent self connections

class cortex_neurons:
    
    def __init__(self, n_layer, l_name, model_eqs, refractory, threshold='v>theta', reset = None, reset_potential = None):
        '''
        l_name : str, array
                name for each layer/group
        n_layer : int, array 
                  number of neurons in each layer
        model_eqs : str
                    neuron model - differential functions
        reset_eqs : str
                    reset function
        threshold: str
                   logic condition
        ref : str or double
              refractory period (condition or in milliseconds)
        '''
        
        if len(l_name) != len(n_layer):
            raise ValueError("Arguments l_name and n_layer must be of the same size")
        
        self.n_layer = n_layer
        self.l_name = l_name
        self.eqs = model_eqs
        self.ref = refractory
        self.threshold = threshold
        self.reset = reset
        self.v_r = reset_potential
        
    def create_neurons(self, cols=False, n_cols = 1, method = 'linear'):
        '''
        This function is to create groups of neurons or the motor cortex model using brian2
        cols : boolean
               implement column structure
        num_cols : int
                   number of columns
        method : str
                 numerical solver method (eg. 'linear', 'euler', 'rk4')
        '''

        nn_cum = [0]
        nn_cum.extend(list(np.cumsum(self.n_layer*n_cols)))
        N = sum(self.n_layer*n_cols)
        
        self.neurons = b2.NeuronGroup(N, self.eqs, threshold=self.threshold, reset=self.reset, \
                            method=method, refractory=self.ref)
        
        n_groups = [] # Create neuron group for each population
        for r in range(0, len(self.n_layer)):
            n_groups.append(self.neurons[nn_cum[r]*n_cols:nn_cum[r+1]*n_cols])
            
        n_dict = dict(zip(self.l_name, [x*n_cols for x in self.n_layer]))
        
        return n_groups, n_dict
    
    def initialise(self, n_type):
        '''
        This function is to initialise the parameters depending on the model used.
        neurons : brian2 NeuronGroup
        n_type : str
                 'LIF'
        '''
            
        if n_type == 'LIF':
            self.neurons.v = '-58.0*mV + 10.0*mV*randn()'
            self.neurons.Ie = 0.0*b2.pA      # initial value for synaptic currents
            self.neurons.Ii = 0.0*b2.pA      # initial value for synaptic currents
            self.neurons.Iext = 0.0*b2.pA   # constant external current
        
        else:
            raise ValueError("Model type not defined")
    
    def spatial_loc(self, X_col, Y_col, Z_col, space, n_groups, n_dict, n_cols = 1):
        '''
        This function is to initialise the spatial location of the neurons.
        X_col : int
                X dimension of cortical column in um
        Y_col : int
                Y dimension of cortical column in um
        Z_col : int
                Z depth dimention of cortical column in um
        space : int
                spacing between columns in um
        n_groups : object from create neurons function contains definitions of neuron groups
        n_dict : dictionary from create neurons function containing groups and numbers of neurons
        '''
        X=[]
        Y=[]
        num = [int(x/n_cols) for x in list(n_dict.values())]
        dim = np.sqrt(n_cols)
        
        #Define X, Y locations
        for i in range(len(num)):
            for j in range(n_cols): 
                for k in range(num[i]):                        
                        X.append(np.random.randint(X_col) + space*j%(dim) + (j%(dim)) * (X_col+space))
                        Y.append(np.random.randint(Y_col) + np.floor(j/dim) * (Y_col+space))
        
        n_layer = [int(x/n_cols) for x in list(n_dict.values())]
        self.neurons.X = X * b2.um
        self.neurons.Y = Y * b2.um

        n_layer_grouped = []
        l_name = list(n_dict.keys())
        layers = ['2/3', '4', '5', '6']
        for l in layers:
            indices = [i for i, s in enumerate(l_name) if l in s]
            n_layer_grouped.append(np.array(n_layer)[indices].sum())
        
        thickness = []
        z_boundaries = [0]
        for i in range(len(n_layer_grouped)):
            thickness.append(Z_col*(n_layer_grouped[i]/sum(n_layer_grouped)))
        
        z_boundaries.extend(np.cumsum(thickness))
        z_boundaries = [Z_col - x for x in z_boundaries]
              
        for i, g in enumerate(list(n_dict.keys())[:-1]):
            if g in n_dict:
                self.neurons[n_groups[list(n_dict.keys()).index(g)].start:n_groups[list(n_dict.keys()).index(g)].stop].Z = np.random.uniform(z_boundaries[int(np.ceil((i+1)/2))], z_boundaries[int(np.floor((i)/2))], n_layer[i]*n_cols) * b2.um
        self.neurons[n_groups[-1].start:n_groups[-1].stop].Z = 0* b2.um
        
    def poission_input(self, n_groups, input_groups, var, freq, weight):
        ''' 
        This function creates a random poisson input for the neuron groups
        n_groups : brian2 neuron group
        input_groups : number of inputs for each neuron
        freq : frequency
        weight : weight of input connection
        '''
        syn_in  = []
        for r in range(0, len(n_groups)):
            syn_in.append(b2.PoissonInput(n_groups[r], var, input_groups[r], freq*b2.Hz, weight=weight)) #'I'
        return syn_in

    def init_synapses(self, syn_model, pre_model_E, pre_model_I):
        '''
        This function is to initialise the synapse model.
        '''
        self.syn_model = syn_model
        self.pre_model_E = pre_model_E
        self.pre_model_I = pre_model_I

    #@profile
    def create_synapses(self, n_dict, n_groups, conn_table, conn_type, calc_nsyn = True, nsyn_list = None):
        '''
        This function builds the synaptic connections in the neurons according to a table definition
        
        neurons : brian2 neurongroup 
        n_dict : dictionary containing neurongroup names and quantities 
        n_groups : subgroups of neurons
        conn_table : table of synaptic connections 
        conn_type : random or spatial  
        syn_model : synapse model 
        pre_model_E : synaptic model for excitatory groups 
        pre_model_I : synaptic model for inhibitory groups
        weight_scale : scaling factor for weights 
        calc_nsyn : whether or not to calculate the number of syanpses based on Peter's rule (default = True)
        nsyn_list : definition of number of syanspes (default = None):
        
        '''
        syn_group = []    
        nsyn_array = np.zeros([len(n_groups), len(n_groups)])
        l_name = self.l_name
    
        syn_E = b2.Synapses(self.neurons, self.neurons, model=self.syn_model, on_pre=self.pre_model_E)
        syn_I = b2.Synapses(self.neurons, self.neurons, model=self.syn_model, on_pre=self.pre_model_I)
    
        for i, r in conn_table.iterrows():
            src = str("".join([r.loc['Source'], r.loc['SourceType']]))
            tgt = str("".join([r.loc['Target'], r.loc['TargetType']]))
            nsyn = 0
    
            if calc_nsyn == True:
                nsyn = int(np.log(1.0-r.loc['Pmax'])/np.log(1.0 - (1.0/float(n_dict[src]*n_dict[tgt]))))
                nsyn = int(nsyn)
                
            else: 
                nsyn = nsyn_list[i]
        
            if nsyn < 1:
                        pass
            else:
                #print('Connecting:', src, '->', tgt)
                #print('Number of Synapses:', nsyn)
                nsyn_array[l_name.index(tgt), l_name.index(src)] = nsyn#/n_dict[tgt]
                pre_index = np.random.randint(n_groups[l_name.index(src)].start, n_groups[l_name.index(src)].stop, size=nsyn)
                
                if conn_type =='random':
                    post_index = np.random.randint(n_groups[l_name.index(tgt)].start, n_groups[l_name.index(tgt)].stop, size=nsyn)
                    
                elif conn_type == 'spatial':
                    pre_counter=collections.Counter(pre_index)
                    pre_index_list = list(pre_counter.keys())
                    num_conn = list(pre_counter.values())
                    
                    p = np.empty((n_groups[l_name.index(src)].N, n_groups[l_name.index(tgt)].N))
                    _find_p(src, tgt, n_dict, n_groups, r.loc['Radius'], p)
    
                    post_index = np.zeros_like(pre_index)
                    pre_indices = np.zeros_like(pre_index)
                    for i, point_oi in enumerate(pre_index_list): 
                        idxs = random.choices(np.arange(n_groups[list(n_dict.keys()).index(tgt)].start, n_groups[list(n_dict.keys()).index(tgt)].stop), weights=p[point_oi-n_groups[l_name.index(src)].start], k = num_conn[i]) #First Method
                        #idxs = random.choices(np.arange(n_groups[l_name.index(tgt)].start, n_groups[l_name.index(tgt)].stop), weights=prob_array[n_groups[l_name.index(tgt)].start:n_groups[l_name.index(tgt)].stop, point_oi], k = num_conn[i]) #Large Array
                        post_index[sum(num_conn[:i]):sum(num_conn[:i+1])] = idxs
                        pre_indices[sum(num_conn[:i]):sum(num_conn[:i+1])] = point_oi
                    pre_index = pre_indices
                    del p
                    
                if r.loc['SourceType'] == 'E':
                    syn_E.connect(i = pre_index.astype(int), j = post_index.astype(int))
                    syn_E.delay[-nsyn:] = 'clip({}*ms + {}*randn()*ms, 0.1*ms, inf*ms)'.format(r.loc['Delay'], r.loc['Dstd'])
                    syn_E.w[-nsyn:] = '(({} + {}*randn()))'.format(r.loc['Weight'], r.loc['Wstd'])
    
    				# Inhibitory connections
                elif r.loc['SourceType'] == 'I':
                    syn_I.connect(i = pre_index.astype(int), j = post_index.astype(int))
                    syn_I.delay[-nsyn:] = 'clip({}*ms + {}*randn()*ms, 0.1*ms, inf*ms)'.format(r.loc['Delay'], r.loc['Dstd'])
                    syn_I.w[-nsyn:] = '(({} + {}*randn()))'.format(r.loc['Weight'], r.loc['Wstd'])
                    
        syn_group.append(syn_E)
        syn_group.append(syn_I)
                        
        return syn_group, nsyn_array
    
    def run_model(self, n_groups, syn_group, simulation_time):

        ########### Define Monitors
        statemon = b2.StateMonitor(self.neurons, 'v', record=range(self.neurons.N))

        spikemon = b2.SpikeMonitor(self.neurons)
        smon_L23E = b2.SpikeMonitor(n_groups[0])
        smon_L23I = b2.SpikeMonitor(n_groups[1])
        smon_L4E = b2.SpikeMonitor(n_groups[2])
        smon_L4I = b2.SpikeMonitor(n_groups[3])
        smon_L5E = b2.SpikeMonitor(n_groups[4])
        smon_L5I = b2.SpikeMonitor(n_groups[5])
        smon_L6E = b2.SpikeMonitor(n_groups[6])
        smon_L6I = b2.SpikeMonitor(n_groups[7])

        ratemon = b2.PopulationRateMonitor(self.neurons)
        rmon_L23E = b2.PopulationRateMonitor(n_groups[0])
        rmon_L23I = b2.PopulationRateMonitor(n_groups[1])
        rmon_L4E = b2.PopulationRateMonitor(n_groups[2])
        rmon_L4I = b2.PopulationRateMonitor(n_groups[3])
        rmon_L5E = b2.PopulationRateMonitor(n_groups[4])
        rmon_L5I = b2.PopulationRateMonitor(n_groups[5])
        rmon_L6E = b2.PopulationRateMonitor(n_groups[6])
        rmon_L6I = b2.PopulationRateMonitor(n_groups[7])

        ###########  Run
        net = b2.Network(b2.collect())
        net.add(self.neurons, n_groups, syn_group)    # Adding objects to the simulation
        net.run(simulation_time, report='stdout')
        
        spikemon_list = [smon_L23E, smon_L23I, smon_L4E, smon_L4I, smon_L5E, smon_L5I, smon_L6E, smon_L6I]
        ratemon_list = [rmon_L23E, rmon_L23I, rmon_L4E, rmon_L4I, rmon_L5E, rmon_L5I, rmon_L6E, rmon_L6I]
        
        return statemon, spikemon, ratemon, spikemon_list, ratemon_list
            