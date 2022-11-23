# -*- coding: utf-8 -*-
"""
@author: Lysea Haggie lmun373@aucklanduni.ac.nz
"""

import brian2 as b2
from brian2 import ms, pF, mV, pA
import pandas as pd
import numpy as np

#Table of connections
con_tab = pd.read_csv('../Connection_Tables/cortex_spatial.csv', delimiter=' ', index_col=False)

#Cortex Layer names
l_name = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']
n_layer = [1148, 324, 268, 60, 1216, 304, 800, 164] #9 columns

bg_layer = [2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850]

num_cols = 9

simulation_time = 550*ms

d_ex = 1.5*ms      	# Excitatory delay
std_d_ex = 0.75*ms 	# Std. Excitatory delay
d_in = 0.80*ms      # Inhibitory delay
std_d_in = 0.4*ms  	# Std. Inhibitory delay
tau_syn = 0.5*ms    # Post-synaptic current time constant
tau_m   = 10.0*ms		# membrane time constant
tau_ref = 2*ms		# absolute refractory period
Cm      = 250.0*pF		# membrane capacity
v_r     = -65.0*mV		# reset potential
theta    = -50.0*mV		# fixed firing threshold
w_ex = 87.8*pA		   	# excitatory synaptic weight
std_w_ex = 0.1*w_ex     # standard deviation weigth
g = 4
bg_freq = 8
b2.defaultclock.dt = 0.1*ms    # timestep of numerical integration method

L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I = 1, 2, 3, 4, 5, 6, 7, 8

# Leaky integrate-and-fire model equations
# dv/dt: equation 1 from the article
# dI/dt: equation 2 from the article
eqs = '''
	dv/dt = (-v + v_r)/tau_m + (Ie + Ii + Iext)/Cm : volt (unless refractory)
	dIe/dt = -Ie/tau_syn : amp
    dIi/dt = -Ii/tau_syn : amp
	Iext : amp
    X : meter
    Y : meter
    Z : meter
	'''

# Reset condition
eqs_reset = 'v = v_r'
    
eqs_syn = 'w:amp'

# equations executed only when presynaptic spike occurs:
# for excitatory connections
e_pre = 'Ie_post += w'
i_pre = 'Ii_post += w'