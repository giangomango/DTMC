#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ADJ List of faces indice on each vertex, so that ADJ(NI(i)+j) =
#f, means that face f is the jth face (in no particular order) incident
#on vertex i.
#NI #V+1 list cumulative sum of vertex-triangle degrees with a
#preceeding zero. “How many faces” have been seen before visiting this
#vertex and its incident faces.


import igl
import numpy as np


def elastic_energy(ver,TRI,k):
    pd1, pd2, c1, c2 = igl.principal_curvature(ver, TRI) #principal curvatures
    ADJ,NI=igl.vertex_triangle_adjacency(TRI, len(ver)) #faces adjacent to vertices to calculate vertex area
    area = igl.doublearea(ver, TRI) / 2.0 #doublearea computes double area of each triangle of the mesh
    A_v=np.zeros(len(ver)) #To calculate vector of vertex areas
    for i in range(0,len(NI)-1):
        neig_faces=ADJ[NI[i]:NI[i+1]]
        aus=0
        for j in neig_faces:
            aus+=area[j]
        A_v[i]=aus/3 
    H_el=0.5*k
    a=0
    for i in range(0,len(c1)):
        a+=((c1[i]+c2[i])**2)*A_v[i]
    H_el=H_el*a
    return H_el

