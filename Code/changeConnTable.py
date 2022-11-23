# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:17:23 2022

@author: lmun373
"""

#Calculate number of connections

import pandas as pd
import numpy as np
import parameters as params1

def find_pconn(nsyn, n_pre, n_post):
    return 1 - np.exp(nsyn*np.log(1-(1/(n_pre*n_post))))

def change_conn_table(n_layer, l_name, filename1, filename2):
    ##Read in Conection Table
    df = pd.read_excel('../Connection_Tables/RelativeConnections.xlsx', engine='openpyxl',  header = 0)
    
    p_array = np.zeros_like(df.loc[:, df.columns != 'Target'])
    
    n_layer = [n*params1.num_cols for n in n_layer]
    
    ###Function to find probability of connections
    for i, src in enumerate(l_name):
        for j in range(len(l_name)):
            p_array[j, i] = find_pconn(df[src][j]*n_layer[l_name.index(df['Target'][j])], n_layer[l_name.index(src)], n_layer[l_name.index(df['Target'][j])])
            
    conn_table = pd.read_csv(filename1, delimiter = ' ', header = 0)
    
    ## Create Pmax column in table
    for i, r in conn_table.iterrows():
        src = r.loc['Source'] + r.loc['SourceType']
        tgt = r.loc['Target'] + r.loc['TargetType']
        conn_table['Pmax'][i] = round(p_array[l_name.index(tgt), l_name.index(src)], 4)
    
    # ### Add column
    # Radius = 1000*np.ones(len(conn_table))
    # conn_table['Radius'] = Radius
    
    # ### Change order of columns
    # cols = conn_table.columns.tolist()
    # cols_list = ['Source', 'SourceType', 'Target', 'TargetType', 'Pmax', 'Radius', 'Weight', 'Wstd', 'Delay', 'Dstd']
    # conn_table = conn_table[cols_list]
        
    ###Output to file
    conn_table.to_csv(filename2, sep=' ', index = False)
    
    return conn_table

def esser_radius(con_tab, filename):
    
    for i, r in con_tab.iterrows():
        
        if r.loc['Source'] == r.loc['Target'] and r.loc['SourceType'] == 'E' and r.loc['Source'] != '6' and r.loc['Target'] != '6':
            con_tab['Radius'][i] = 0.0003
        if r.loc['Source'] == r.loc['Target'] and r.loc['SourceType'] == 'E' and r.loc['Source'] == '6' and r.loc['Target'] == '6':
            con_tab['Radius'][i] = 0.000225
        if r.loc['Source'] == r.loc['Target'] and r.loc['SourceType'] == 'I':
            con_tab['Radius'][i] = 0.000175
        if r.loc['Source'] != r.loc['Target']:
            con_tab['Radius'][i] = 0.00005
    
    con_tab.to_csv(filename, sep=' ', index = False)