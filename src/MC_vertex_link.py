#!/usr/bin/env python
# coding: utf-8

# In[1]:


import igl
import numpy as np
from src.elastic_energy import elastic_energy
import random

# In[2]:


def MCstep_vertex(ver,TRI,σ,r,k,β): #tries to move N vertices
    neighbour=igl.adjacency_list(TRI) 
    index=np.random.permutation(len(ver))
    for i in index:
        δ=np.array([random.uniform(-σ,σ),random.uniform(-σ,σ),random.uniform(-σ,σ)])
        H_old=elastic_energy(ver,TRI,k)
        x0=ver[i]
        x0+=δ #shift vertex
        for j in range(0,len(ver)): #For now at each step check overlap with all other vertices, O(N^2), SLOW
            d=np.linalg.norm(x0-ver[j])
            if j in neighbour[i]:#For the neighbours check also that thether distance don't exeed max of sqrt(2)
                if d < r or d > np.sqrt(2):
                    x0-=δ #if neighbours too far put back vertex and break cycle
                    break
            else:
                if d < r and d!=0: #for all other vertices check only that they don't overlap
                    x0-=δ    #two  vertices overlap. Put back the vertex in it's original position
                    break
        H_new=elastic_energy(ver,TRI,k) #if already rejected move energies are equal and P=1, so rand never greater
        ΔH=H_new-H_old
        P=min(1, np.exp(-β* ΔH))
        if np.random.rand()>P: #with probability 1-P put vertex back
            x0-=δ


# In[3]:


def fliplink(ver,neig,x,y):
    x_new=0
    y_new=0            #x_new. y_new elements not contained in the edge, that will form new edge
    x_ind=0 #indices of first vertex forming old edge in x and second vertex forming old edge in y
    y_ind=0
    u=np.copy(x)
    v=np.copy(y)
    both=[]
    d=0
    for i in x:
        if i in y:
            both.append(i)
    if len(both)==2:
        for i in range(0,len(x)):
            if x[i]== both[0]:
                x_ind=i
            if x[i] not in both:
                x_new=x[i]
        for j in range(0,len(y)):
            if y[j]== both[1]:
                y_ind=j
            if y[j] not in both:
                y_new=y[j]
        u[x_ind]=y_new
        v[y_ind]=x_new
        d=np.linalg.norm(ver[x_new]-ver[y_new])
    if y_new in neig[x_new]: #Check that new edge that is being added is not already present.(Can happen due to pyramidal
        return x,y,0         #configurations, leading to same triangle appearing multiple times and deleting edges)
      #PRESERV COUNTER-CLOCKWISE ORDER

    return u,v,d


# In[ ]:


def MCstep_link(ver,TRI,β,k,r):
    ev,et,te=igl.edge_topology(ver,TRI)
    index=np.random.permutation(len(ev))
    for n in index[0:len(ver)]:
        ev,et,te=igl.edge_topology(ver,TRI) #Calculate topology at each step, changes indicization, could try to flip same edge
        neig=igl.adjacency_list(TRI)
        l,t=te[n][0],te[n][1]
        if l!=-1 or t!=-1: #Don't flip border edges
            x=np.copy(TRI[l]) #te[n] contains labels of the triangles that share that edge
            y=np.copy(TRI[t])
            u,v,d=fliplink(ver,neig,x,y)
            if r< d< np.sqrt(2) and d!=0: #just need to check if new edge creates overlap of hard beads or too large thether
                H_old=elastic_energy(ver,TRI,k)
                E=igl.edges(TRI)
                TRI[l]=u #substitute new triangles in TRI 
                TRI[t]=v
                 #check that no nodes with less than 3 neighbours
                for z in range(0,len(neig)):
                    if len(neig[z])<3:
                        TRI[l]=x
                        TRI[t]=y
                        break
                H_new=elastic_energy(ver,TRI,k)
                ΔH=H_new-H_old
                P=min(1, np.exp(-β* ΔH))
                if np.random.rand() > P: #with probability 1-P put old triangles back
                    TRI[l]=x 
                    TRI[t]=y

