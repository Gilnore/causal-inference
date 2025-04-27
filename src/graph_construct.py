# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:41:31 2025

@author: robin

this generates a bidirectional graph using transfer entropy
then uses causation entropy to reduce the links to correct values
"""
import numpy as np
import causal_inference as ci
import networkx as nx

def generate_graph(series:list, names:list=None)->np.ndarray:
    #this generates a grpah in a one hot fasion using NUE
    l = len(series)
    G = nx.MultiDiGraph()
    for i in range(l):
        emb = ci.make_nue(series[i], series)
        for j in emb.keys():
            j = int(j)
            others = [emb.get(n) for n in emb.keys() if (n != i) and (n != j)]
            if len(others) > 0:
                others = np.hstack(others)
            else: others = np.array([[],],dtype=np.float64)
            cmi,k = ci.NN_CMI(emb.get(i), emb.get(j), others)
            if names is None:
                G.add_edge(j, i, weight=cmi[0])
            else:
                G.add_edge(names[j], names[i], weight=cmi[0])
    return G
