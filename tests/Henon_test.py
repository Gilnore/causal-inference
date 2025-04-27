# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 19:48:17 2025

@author: robin
"""
import numpy as np
import causal_inference as ci
import graph_construct as gc
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def Henon_data(N:int,q:float,random_init:bool=True):
    Y = np.zeros((5,N))
    if random_init:
        Y[:,-2:] = np.random.rand(5,2)
    for n in range(N):
        for l in range(5):
            if l in [0,4]:
                Y[l,n] = 1.4 - Y[l,n-1]**2 + .3*Y[l,n-2]
            else:
                Y[l,n] = 1.4 - (.5*q*(Y[l-1,n-1] + Y[l+1,n-1]) + (1-q) * Y[l,n-1])**2 + .3*Y[l,n-2]
    return Y

def AR_data(N:int,q:float):
    Y = np.zeros((5,N))
    for n in range(N):
        Y[0,n] = .95*2**.5*Y[0,n-1] - .9125*Y[0,n-2] + np.random.randn(1)
        Y[1,n] = .5*Y[0,n-2] **2 + np.random.randn(1)
        Y[2,n] = -.4*Y[0,n-3] + .4*Y[1,n-1] + np.random.randn(1)
        Y[3,n] = -.5*Y[0,n-1]**2 + .25*2**.5 *Y[3,n-1] + np.random.randn(1)
        Y[4,n] = -.25*2**.5 *Y[3,n-1] + .25*2**.5 *Y[4,n-1] + np.random.randn(1)
    return Y

def space_test():
    y1,y2,y3,y4,y5 = np.flip(Henon_data(200, .6), axis=1)
    scale = StandardScaler()
    d = 1
    y1 = scale.fit_transform(y1.reshape(-1,d))
    y2 = scale.fit_transform(y2.reshape(-1,d))
    y3 = scale.fit_transform(y3.reshape(-1,d))
    y4 = scale.fit_transform(y4.reshape(-1,d))
    y5 = scale.fit_transform(y5.reshape(-1,d))
    #the original ordering is past is on the left, when reshaped, past is at top
    #we flip it so the past is to the bottom when reshaped
    #we should see 2 embeddings found for y1 and y5 for self dependance
    #then the other ones would have dependance on 2 other series with .5*q dependance
    #on top of their self depandance
    false_pos_avg = 0
    false_neg_avg = 0
    true_pos_avg = 0
    true_neg_avg = 0
    for _ in range(10):
        false_neg = 0 #how many was supposed to be but wasn't
        false_pos = 0 #how many wasn't supposed to be but was
        true_pos = 0
        y1_emb = ci.make_nue(y1,(y1,y5,y2,y3,y4,),blend=1)
        self_err = y1_emb.get(0,np.array([[],])).shape[1] - 2
        false_neg += (abs(self_err) if self_err < 0 else 0)
        false_pos += (self_err if self_err > 0 else 0)
        true_pos += 2 - (abs(self_err) if self_err < 0 else 0)
        
        true_neg = 0
        for key in [1,2,3,4]:
            if y1_emb.get(key) is None:
                if key == 1:
                    true_neg+=1
                else: true_neg +=4

        for key in y1_emb.keys():
            key = int(key)
            if key != 0:
                false_pos+= y1_emb.get(key,np.array([[],])).shape[1]
        
        y2_emb = ci.make_nue(y2,(y1,y5,y2,y3,y4,),blend=1)
        
        self_err = y2_emb.get(2,np.array([[],])).shape[1]-2
        false_neg += (abs(self_err) if self_err < 0 else 0)
        false_pos += (self_err if self_err > 0 else 0)
        true_pos += 2 - (abs(self_err) if self_err < 0 else 0)
        
        int_err = y2_emb.get(0,np.array([[],])).shape[1]-1
        false_neg += (abs(int_err) if int_err < 0 else 0)
        false_pos += (int_err if int_err > 0 else 0)
        true_pos += 1 - (abs(int_err) if int_err < 0 else 0)
        
        int_err1 = y2_emb.get(3,np.array([[],])).shape[1]-1
        false_neg += (abs(int_err1) if int_err1 < 0 else 0)
        false_pos += (int_err1 if int_err1 > 0 else 0)
        true_pos += 1 - (abs(int_err1) if int_err1 < 0 else 0)
        
        for key in [1,4]:
            if y2_emb.get(key) is None:
                if key == 1:
                    true_neg+=1
                else: true_neg +=4
        
        for key in y2_emb.keys():
            key = int(key)
            if key not in [0,2,3]:
                false_pos+= y2_emb.get(key,np.array([[],])).shape[1]
                
        y3_emb = ci.make_nue(y3,(y1,y5,y2,y3,y4,),blend=1)
        
        self_err = y3_emb.get(3,np.array([[],])).shape[1]-2
        false_neg += (abs(self_err) if self_err < 0 else 0)
        false_pos += (self_err if self_err > 0 else 0)
        true_pos += 2 - (abs(self_err) if self_err < 0 else 0)
        
        int_err = y3_emb.get(2,np.array([[],])).shape[1]-1
        false_neg += (abs(int_err) if int_err < 0 else 0)
        false_pos += (int_err if int_err > 0 else 0)
        true_pos += 1 - (abs(int_err) if int_err < 0 else 0)
        
        int_err1 = y3_emb.get(4,np.array([[],])).shape[1]-1
        false_neg += (abs(int_err1) if int_err1 < 0 else 0)
        false_pos += (int_err1 if int_err1 > 0 else 0)
        true_pos += 1 - (abs(int_err1) if int_err1 < 0 else 0)
        
        for key in [0,1]:
            if y3_emb.get(key) is None:
                true_neg+=1
        
        for key in y3_emb.keys():
            key = int(key)
            if key not in [3,2,4]:
                false_pos+= y3_emb.get(key,np.array([[],])).shape[1]
        
        y4_emb = ci.make_nue(y4,(y1,y5,y2,y3,y4,),blend=1)
        
        self_err = y4_emb.get(4,np.array([[],])).shape[1]-2
        false_neg += (abs(self_err) if self_err < 0 else 0)
        false_pos += (self_err if self_err > 0 else 0)
        true_pos += 2 - (abs(self_err) if self_err < 0 else 0)
        
        int_err = y4_emb.get(3,np.array([[],])).shape[1]-1
        false_neg += (abs(int_err) if int_err < 0 else 0)
        false_pos += (int_err if int_err > 0 else 0)
        true_pos += 1 - (abs(int_err) if int_err < 0 else 0)
        
        int_err1 = y4_emb.get(1,np.array([[],])).shape[1]-1
        false_neg += (abs(int_err1) if int_err1 < 0 else 0)
        false_pos += (int_err1 if int_err1 > 0 else 0)
        true_pos += 1 - (abs(int_err1) if int_err1 < 0 else 0)
        
        for key in [0,2]:
            if y4_emb.get(key) is None:
                if key == 0:
                    true_neg+=1
                else: true_neg +=4
        
        for key in y4_emb.keys():
            key = int(key)
            if key not in [3,1,4]:
                false_pos+= y4_emb.get(key,np.array([[],])).shape[1]
        
        y5_emb = ci.make_nue(y5,(y1,y5,y2,y3,y4,),blend=1)

        self_err = y5_emb.get(1,np.array([[],])).shape[1] - 2
        false_neg += (abs(self_err) if self_err < 0 else 0)
        false_pos += (self_err if self_err > 0 else 0)
        true_pos += 2 - (abs(self_err) if self_err < 0 else 0)
        
        for key in [0,2,3,4]:
            if y5_emb.get(key) is None:
                if key == 0:
                    true_neg+=1
                else: true_neg +=4
        
        for key in y5_emb.keys():
            key = int(key)
            if key != 1:
                false_pos+= y5_emb.get(key,np.array([[],])).shape[1]
        
        false_neg_avg += false_neg/10
        false_pos_avg += false_pos/10
        true_pos_avg += true_pos/10
        true_neg_avg += true_neg/10
    return false_neg_avg, false_pos_avg, true_pos_avg, true_neg_avg

def info_acc_test():
    d = 1
    series = np.flip(Henon_data(200, .6), axis=1)
    scale = StandardScaler()
    # gc.idtxl_test(series)
    y1,y2,y3,y4,y5 = series
    
    y1 = scale.fit_transform(y1.reshape(-1,d))
    y2 = scale.fit_transform(y2.reshape(-1,d))
    y3 = scale.fit_transform(y3.reshape(-1,d))
    y4 = scale.fit_transform(y4.reshape(-1,d))
    y5 = scale.fit_transform(y5.reshape(-1,d))
    series = (y1.reshape(-1,d),
             y5.reshape(-1,d),
             y2.reshape(-1,d),
             y3.reshape(-1,d),
             y4.reshape(-1,d),)
    graph = gc.generate_graph(series)
    labels = nx.get_edge_attributes(graph,'weight')
    plt.figure()
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
    nx.draw(graph,pos=pos, with_labels=True)


# false_neg, false_pos, true_pos, true_neg = space_test()
# print((true_pos+true_neg)/(false_pos+false_neg+true_pos+true_neg), "acc")
# print((true_pos)/(true_pos+false_neg), "sensitivity")
# print((true_neg)/(false_pos+true_neg), "specificity")
info_acc_test()