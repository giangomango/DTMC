#!/usr/bin/env python
# coding: utf-8

# In[1]:


import igl
import numpy as np
from src.elastic_energy import update_energy_vertex
from src.elastic_energy import update_energy_neig
from src.elastic_energy import update_energy_link
import random

def MCstep_vertexB(ver,TRI,header,linklis,L,σ,r,k,β,H,M,A_v,normals_face,ADJ,NI,neighbour,et,ev,te,SHO,ver_bulk,border,area,h,part,μ): #tries to move N vertices
    index=np.random.permutation(ver_bulk) #border vertices are fixed
    #count=0
    for i in index:
        #count+=1
        δ=np.array([random.uniform(-σ,σ),random.uniform(-σ,σ),random.uniform(-σ,σ)])
        x0=ver[i]
        cx,cy,cz= x0[0] // 1, x0[1] // 1, x0[2] // 1 #keep track of starting cell of x0
        x0+=δ #shift vertex
        boole=0
        for j in neighbour[i]:#For the neighbours check also that thether distance don't exeed max of sqrt(2)
            d=np.linalg.norm(x0-ver[j])
            if d < r or d > np.sqrt(2):
                x0-=δ #if neighbours too far put back vertex and break cycle
                boole=1 
                break
        if boole==0:
            cix,ciy,ciz= x0[0] // 1, x0[1] // 1, x0[2] // 1 #cell corresponding to new position
            cell_neig=[]
            for dx in range(-1,2): #also (0,0,0) so also checks that no overlap with particles contained in cell corresponding
                for dy in range(-1,2): #to new position
                    for dz in range(-1,2):
                        nx,ny,nz=int(cix+dx),int(ciy+dy),int(ciz+dz)
                        n=int(nx*L[1]*L[2]+ny*L[2]+nz)
                        z=header[n]
                        while(z!=-1):
                            if z not in neighbour[i]:#already checked no overlap with mesh neig
                                cell_neig.append(z)
                            z=int(linklis[z])
                            
            for c in cell_neig: #Check no overlaps with particles contained in neig cells
                d=np.linalg.norm(x0-ver[c])
                if d < r and d!=0: 
                    x0-=δ    #two  vertices overlap. Put back the vertex in it's original position
                    boole=1
                    break
            if boole==0:
                H_new, Av_new, Nf_old, N_v,M_new,SH_old,ind,h_old=update_energy_vertex(ver,TRI,H,M,k,A_v,i,normals_face,neighbour,ADJ,NI,area,SHO,et,te,h,part,μ)
                H_new,Avj,SH_old_neig,ind_neig,h_old_neig,M_neig,ind_c=update_energy_neig(ver,TRI,H_new,M,k,A_v,i,normals_face,neighbour,ADJ,NI,area,SHO,et,te,border,h,part,μ)
                H_new=H_new.real
                #print(H," ",H_new)
                ΔH=H_new-H
                P=min(1, np.exp(-β* ΔH))
                if np.random.rand()>P: #with probability 1-P put vertex back
                    x0-=δ
                    for u in range(0,len(Nf_old)): #Put back also normals of faces and face areas
                        area[Nf_old[u][2]]=Nf_old[u][1]
                        normals_face[Nf_old[u][2]]=Nf_old[u][0]
                    for v in range(0,len(ind)):   #Put back shape operators 
                        SHO[:,:,ind[v]]=SH_old[:,:,v]
                        h[ind[v]]=h_old[v]
                    for w in range(0,len(ind_neig)): #already updates shape op centered in both vertices if not border
                        SHO[:,:,ind_neig[w]]=SH_old_neig[:,:,w]
                        h[ind_neig[w]]=h_old_neig[w]
                    
                else: #particle shifted, update cell list linked list
                    c=int(cx*L[1]*L[2]+cy*L[2]+cz) #linear index that containined x0,that has now been shifted
                    ci=int(cix*L[1]*L[2]+ciy*L[2]+ciz) #new linear index of cell containing x0
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
                    #update also A_v, curvatures at vertex,energy
                    A_v[i]=Av_new #new area at vertex 
                    for u in range(0,len(Avj)): #new neighbours area
                        A_v[Avj[u][1]]=Avj[u][0]
                        M[ind_c[u]]=M_neig[u] #new neig curvatures
                    M[i]=M_new
                    H=H_new
                 
    return H

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
def MCstep_linkB(ver,TRI,H,neig,β,M,k,r,ev,et,te,area,normals_face,ADJ,NI,SHO,border,A_v,h,part,μ): #tries to flip N links
    index=np.random.permutation(len(te))
    for n in index[0:len(ver)]: #n index of edge that I'm trying to flip
        l,t=te[n][0],te[n][1]
        if l!=-1 and t!=-1: #Don't flip border edges
            x=np.copy(TRI[l]) #te[n] contains labels of the triangles that share that edge
            y=np.copy(TRI[t])
            #print(x,y,"OLD")
            u,v,d,ev_new,both=fliplink(ver,neig,x,y)
            #print(u,v,"NEW")
            if r< d< np.sqrt(2) and d!=0: #just need to check if new edge creates overlap of hard beads or too large thether
                TRI[l]=u #substitute new triangles in TRI 
                TRI[t]=v
                H_new,Avj,Nf_old,SH_old_tetra,h_old_tetra,ind_tetra,M_tetra,ind_c,ADJ_new,NI_new,ev_old=update_energy_link(ver,ev,TRI,neig,n,H,M,k,A_v,area,normals_face,x,y,ev_new,l,t,ADJ,NI,SHO,et,te,border,h,part,μ)
                ΔH=H_new-H
                P=min(1, np.exp(-β* ΔH))
                print("flip attempt")
                if np.random.rand() > P: #with probability 1-P put old triangles back
                    TRI[l]=x 
                    TRI[t]=y
                    for jy in ind_tetra:
                        if te[jy][0]==-1 or te[jy][1]==-1:
                            print(SH_old_tetra,ind_tetra,n)
                    #restore old topology
                    ev[n]=ev_old
                    
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

                    
                    #restore old neig matrix
                    neig[ev[n][0],ev[n][1]]=1; neig[ev[n][1],ev[n][0]]=1 #update neig matrix, put back if move fails OK
                    neig[ev_new[0],ev_new[1]]=0; neig[ev_new[1],ev_new[0]]=0
    
    
                    #restore old shape ops and normals
                    
                    for u in range(0,len(Nf_old)): #Put back also normals of faces and face areas OK
                        area[Nf_old[u][2]]=Nf_old[u][1]
                        normals_face[Nf_old[u][2]]=Nf_old[u][0]
                    for v in range(0,len(ind_tetra)):   #Put back shape operators 
                        SHO[:,:,ind_tetra[v]]=SH_old_tetra[:,:,v]
                        h[ind_tetra[v]]=h_old_tetra[v]
                    
                    
                else: #Already updated edge topology and adjacency matrix in update_energy_link
                    ADJ=ADJ_new #update face adjacency
                    NI=NI_new
                    #update curvatures at vertex,energy 
                    for u in range(0,len(Avj)): #new area
                        A_v[Avj[u][1]]=Avj[u][0]
                        M[ind_c[u]]=M_tetra[u] #new tetra curvatures
                    H=H_new
                    
                    
                    
    return H,ADJ,NI
