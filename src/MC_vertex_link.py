#!/usr/bin/env python
# coding: utf-8

# In[1]:


import igl
import numpy as np
from src.elastic_energy import elastic_energy
import random

# In[2]:


def MCstep_vertex(ver,TRI,header,linklis,L,σ,r,k,β): #tries to move N vertices
    neighbour=igl.adjacency_list(TRI)
    index=np.random.permutation(len(ver))
    for i in index:
        δ=np.array([random.uniform(-σ,σ),random.uniform(-σ,σ),random.uniform(-σ,σ)])
        H_old=elastic_energy(ver,TRI,k)
        x0=ver[i]
        cx,cy,cz= x0[0] // 1, x0[1] // 1, x0[2] // 1 #keep track of starting cell of x0
        x0+=δ #shift vertex
        for j in neighbour[i]:#For the neighbours check also that thether distance don't exeed max of sqrt(2)
            d=np.linalg.norm(x0-ver[j])
            if d < r or d > np.sqrt(2):
                x0-=δ #if neighbours too far put back vertex and break cycle
                break
        else:
            cix,ciy,ciz= x0[0] // 1, x0[1] // 1, x0[2] // 1 #cell corresponding to new position
            cell_neig=[]
            for dx in range(-1,2): #also (0,0,0) so also checks that no overlap with particles contained in cell corresponding
                for dy in range(-1,2): #to new position
                    for dz in range(-1,2):
                        nx,ny,nz=int(cix+dx),int(ciy+dy),int(ciz+dz)
                        n=int(nx*L*L+ny*L+nz*L)
                        z=header[n]
                        while(z!=-1):
                            if z!=-1 and z not in cell_neig:#already checked no overlap with mesh neig
                                cell_neig.append(z)
                            z=int(linklis[z])
                            
            for c in cell_neig: #Check no overlaps with particles contained in neig cells
                d=np.linalg.norm(x0-ver[c])
                if d < r and d!=0: 
                        x0-=δ    #two  vertices overlap. Put back the vertex in it's original position
                        break
            else:
                H_new=elastic_energy(ver,TRI,k) #if already rejected move energies are equal and P=1, so rand never greater
                ΔH=H_new-H_old
                P=min(1, np.exp(-β* ΔH))
                if np.random.rand()>P: #with probability 1-P put vertex back
                    x0-=δ
                else: #particle shifted, update cell list linked list
                    c=int(cx*L*L+cy*L+cz*L) #linear index containing x0,that has now been shifted
                    ci=int(cix*L*L+ciy*L+ciz*L) #new linear index of cell containing x0
                    if c!=ci: #shift particle only if it changes cell
                        if header[c]==i: #if header of starting cell was x0
                            header[c]=linklis[i]  #update header of cell where removing x0, new header particle to which x0
                        else:                     #pointed (-1 if none)
                            z=header[c]
                            #print(i)
                            while(z!=i): #find particle that pointed to i
                                q=z
                                z=int(linklis[z])
                                #print(q,z)
                            linklis[q]=linklis[i] #removing x0 particle that pointed to x0 points to particle that was pointed 
                                                  #by x0
                        if header[ci]==-1: #updates for new cell containing x0, if empty x0 now points to empty and becomes new 
                                           #header
                            linklis[i]=-1
                            header[ci]=i
                        else: #if particle already present, x0 points to previous header and becomes new header
                            linklis[i]=header[ci]
                            header[ci]=i

# In[3]:

def fliplink(ver,neig,x,y): #PRESERV COUNTER-CLOCKWISE ORDER
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
    #print(both,"BOTH")
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
        ev_new=[x_new,y_new]
        d=np.linalg.norm(ver[x_new]-ver[y_new])
        if neig[x_new,y_new]==1:#Check that new edge that is being added is not already present.(Can happen due to pyramidal
            return x,y,0,0,0 #configurations, leading to same triangle appearing multiple times and deleting edges)
        for i in range(0,2):
            if neig[both[i]].nnz==3: #Check that removing thether don't leave vertices with less than 3 neighbours
                return x,y,0,0,0
        return u,v,d,ev_new,both
    return x,y,0,0,0


# In[ ]:

#ev,et,te=igl.edge_topology(ver,TRI)
#neig=igl.adjacency_matrix(TRI)
def MCstep_link(ver,TRI,neig,β,k,r,ev,et,te): #tries to flip N links
    index=np.random.permutation(len(te))
    for n in index[0:len(ver)]:
        l,t=te[n][0],te[n][1]
        if l!=-1 and t!=-1: #Don't flip border edges
            x=np.copy(TRI[l]) #te[n] contains labels of the triangles that share that edge
            y=np.copy(TRI[t])
            #print(x,y,"OLD")
            u,v,d,ev_new,both=fliplink(ver,neig,x,y)
            #print(u,v,"NEW")
            if r< d< np.sqrt(2) and d!=0: #just need to check if new edge creates overlap of hard beads or too large thether
                H_old=elastic_energy(ver,TRI,k)
                TRI[l]=u #substitute new triangles in TRI 
                TRI[t]=v
                H_new=elastic_energy(ver,TRI,k)
                ΔH=H_new-H_old
                P=min(1, np.exp(-β* ΔH))
                if np.random.rand() > P: #with probability 1-P put old triangles back
                    TRI[l]=x 
                    TRI[t]=y
                else: #Link flipped, update topology
                    ev[n]=ev_new #ev OK, only egde changing vertices is the flipped one
                    neig_edges=[]
                    for i in range(0,3):#List of neigh edges of tethrahedron
                        if et[l][i]!=n and et[l][i] not in neig_edges:
                            neig_edges.append(et[l][i])
                        if et[t][i]!=n and et[t][i] not in neig_edges:
                            neig_edges.append(et[t][i])
                            
                    for j in neig_edges: #for all edges in neig check at which of the new triangle it belongs and update te
                        if ev[j][0] in TRI[l] and ev[j][1] in TRI[l]:
                            if l not in te[j]:
                                if te[j][0]==t:
                                    te[j][0]=l
                                else:
                                    te[j][1]=l
                        else: #if ev[j] not in TRI[l] then in TRI[t] by def
                            if t not in te[j]:
                                if te[j][0]==l:
                                    te[j][0]=t
                                else:
                                    te[j][1]=t   #te OK
                    
                    et_l_new=[n]
                    et_t_new=[n]
                    for z in neig_edges: #update list of edge used in triangles t and l
                        if t in te[z]:
                            et_t_new.append(z)
                        if l in te[z]:
                            et_l_new.append(z)
                    et[l]=np.array(et_l_new)
                    et[t]=np.array(et_t_new)
                    
                    neig[both[0],both[1]]=0 #Update neig matrix
                    neig[both[1],both[0]]=0
                    neig[ev_new[0],ev_new[1]]=1
                    neig[ev_new[1],ev_new[0]]=1
