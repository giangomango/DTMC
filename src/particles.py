#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Insert particles in all empty sites with rate k_I, all present particles diffuse with rate K_D/(g^N_nn). No domains extraction for now. Updates shape operators after every particle movement and curvature energy of vertices of relevant triangle.

import random
import numpy as np

def particlesA(ver,TRI,H_el,k,k_I,k_D,g,μ,SHO,h,M,A_v,normals_face,normals_ver,part,et,ev,te,border,neigh,ADJ,NI): #h scalar curvatures at edges
    for i in range(0,len(TRI)): #particles insertion on empty faces
        if part[i]==0:
            if k_I>np.random.rand(): #particle inserted on face i, insert spontaneous curvature at all adjacent edges of -μ
                part[i]=1
                for j in et[i]: #et[i] edges forming i-th triangle 
                    if -1 not in te[j]: #border edges have zero curvature, don't update them
                        if h[j]!=0:
                            SHO[:,:,j]=SHO[:,:,j]/h[j]
                        h[j]-=μ  #update scalar curvature and corresponding shape operator
                        SHO[:,:,j]=SHO[:,:,j]*h[j]
                        
                for x in TRI[i]: #will update twice each vertex, for every neighboring edge
                    if border[x]==False:
                        N_v=normals_ver[x]
                        neig_faces_x=ADJ[NI[x]:NI[x+1]]
                        Av=A_v[x]
                        S_v=np.zeros([3,3])
                        for z in neigh[x]: #Calculate new curvatures
                            neig_faces_z=ADJ[NI[z]:NI[z+1]]
                            faces=[]
                            for q in neig_faces_z:
                                if q in neig_faces_x:
                                    faces.append(q)
                            Nf_1=normals_face[faces[0]]
                            Nf_2=normals_face[faces[1]]
                            et1=et[faces[0]]; et2=et[faces[1]] #list of edges used by faces considered
                            e=0
                            for l in et1:
                                if l in et2:
                                    e=l #index of edge on which calculating curvature

                            N_e=(Nf_1+Nf_2) #new edge normal
                            N_e=N_e/np.linalg.norm(N_e)
                            #print(S_e,e,j)
                            P_v=np.identity(3)-np.tensordot(N_v,N_v,axes=0)
                            W_e=np.dot(N_v,N_e)
                            S_e=SHO[:,:,e] 
                            S_v+=W_e*np.dot((np.conjugate(P_v.transpose())),np.dot(S_e,P_v))

                        S_v=S_v/Av #vertex shape operator has one zero eigenvalue and the other two are the curvatures
                    #print(np.dot(S_v,N_v))

                        #Householder transformation

                        α=[0,0,1] #start from canonical reference frame
                        pvec=α+N_v
                        mvec=α-N_v
                        if np.linalg.norm(pvec)>np.linalg.norm(mvec):
                            pvec=pvec/np.linalg.norm(pvec)
                            T_H=(np.identity(3)-2*np.tensordot(pvec,pvec,axes=0))
                        else:
                            mvec=mvec/np.linalg.norm(mvec)
                            T_H=(np.identity(3)-2*np.tensordot(mvec,mvec,axes=0))

                        C_v=np.dot((np.conjugate(T_H.transpose())),np.dot(S_v,T_H)) #now C 2x2 minor with nonzero elements
                        p=C_v[0,0]; q=C_v[0,1]
                        r=C_v[1,0]; s=C_v[1,1]
                        if q!=0 and r!=0:
                            insq=(p+s)**2-4*(p*s-q*r) #value of which will take squared root

                            if insq>0: #avoid complex eigenvalues
                                c1_new=((p+s)+np.sqrt(insq))*0.5  #largest eigenvalue                            
                                c2_new=((p+s)-np.sqrt(insq))*0.5  #smallest
                            else: #degenerate eigenvalues
                                c1_new=(p+s)*0.5                               
                                c2_new=c1
                        else:
                            if p>s:
                                c1_new=p
                                c2_new=s
                            else:
                                c1_new=s
                                c2_new=p




                        if abs(c1_new)<=10e-10:
                            c1_new=0
                        if abs(c2_new)<=10e-10:
                            c2_new=0

                        H_el=H_el+(((c1_new+c2_new)**2)*Av*0.5*k)-(((M[x])**2)*A_v[x]*0.5*k)
                        M[x]=c1_new+c2_new
        
        else: #particle already present, diffusive jump
            count=0
            neig_empty=[]
            for j in et[i]: #count occupied neighbors and find empty neighbours
                for z in te[j]:
                    if z!=i and z!=-1:
                        if part[z]==1: #occupied neighbour
                            count+=1
                        else:
                            neig_empty.append(z)
            if len(neig_empty)!=0: #if all neigh occupied no jump CHECK
                p=k_D/(g**count)
                if p>np.random.rand(): #particle jump
                    n=random.randrange(0,len(neig_empty)) #choose random empty neighbour
                    part[i]=0
                    b=neig_empty[n]
                    part[b]=1
                    for j in et[i]: #update shape operators of neigh edges removing spontaneous curvature around i
                        if -1 not in te[j]:
                            
                            if h[j]!=0:
                                SHO[:,:,j]=SHO[:,:,j]/h[j]
                            h[j]+=μ
                            SHO[:,:,j]=SHO[:,:,j]*h[j]

                    
                    for j in et[b]:#insert spontaneous curvature on edges around n
                        if -1 not in te[j]:
                            
                            if h[j]!=0:
                                SHO[:,:,j]=SHO[:,:,j]/h[j]
                            h[j]-=μ
                            SHO[:,:,j]=SHO[:,:,j]*h[j]
                    
                    deg=np.concatenate((TRI[i],TRI[b]))
                    tetra=[]
                    for j in deg:
                        if j not in tetra:
                            tetra.append(j)
                    for x in tetra: #will update twice each vertex, for every neighboring edge
                        if border[x]==False:
                            N_v=normals_ver[x]
                            neig_faces_x=ADJ[NI[x]:NI[x+1]]
                            Av=A_v[x]
                            S_v=np.zeros([3,3])
                            for z in neigh[x]: #Calculate new curvatures
                                neig_faces_z=ADJ[NI[z]:NI[z+1]]
                                faces=[]
                                for q in neig_faces_z:
                                    if q in neig_faces_x:
                                        faces.append(q)
                                Nf_1=normals_face[faces[0]]
                                Nf_2=normals_face[faces[1]]
                                et1=et[faces[0]]; et2=et[faces[1]] #list of edges used by faces considered
                                e=0
                                for l in et1:
                                    if l in et2:
                                        e=l #index of edge on which calculating curvature

                                N_e=(Nf_1+Nf_2) #new edge normal
                                N_e=N_e/np.linalg.norm(N_e)
                                #print(S_e,e,j)
                                P_v=np.identity(3)-np.tensordot(N_v,N_v,axes=0)
                                W_e=np.dot(N_v,N_e)
                                S_e=SHO[:,:,e] 
                                S_v+=W_e*np.dot((np.conjugate(P_v.transpose())),np.dot(S_e,P_v))

                            S_v=S_v/Av #vertex shape operator has one zero eigenvalue and the other two are the curvatures
                        #print(np.dot(S_v,N_v))

                            #Householder transformation

                            α=[0,0,1] #start from canonical reference frame
                            pvec=α+N_v
                            mvec=α-N_v
                            if np.linalg.norm(pvec)>np.linalg.norm(mvec):
                                pvec=pvec/np.linalg.norm(pvec)
                                T_H=(np.identity(3)-2*np.tensordot(pvec,pvec,axes=0))
                            else:
                                mvec=mvec/np.linalg.norm(mvec)
                                T_H=(np.identity(3)-2*np.tensordot(mvec,mvec,axes=0))

                            C_v=np.dot((np.conjugate(T_H.transpose())),np.dot(S_v,T_H)) #now C 2x2 minor with nonzero elements
                            p=C_v[0,0]; q=C_v[0,1]
                            r=C_v[1,0]; s=C_v[1,1]
                            if q!=0 and r!=0:
                                insq=(p+s)**2-4*(p*s-q*r) #value of which will take squared root

                                if insq>0: #avoid complex eigenvalues
                                    c1_new=((p+s)+np.sqrt(insq))*0.5  #largest eigenvalue                            
                                    c2_new=((p+s)-np.sqrt(insq))*0.5  #smallest
                                else: #degenerate eigenvalues
                                    c1_new=(p+s)*0.5                               
                                    c2_new=c1
                            else:
                                if p>s:
                                    c1_new=p
                                    c2_new=s
                                else:
                                    c1_new=s
                                    c2_new=p




                            if abs(c1_new)<=10e-10:
                                c1_new=0
                            if abs(c2_new)<=10e-10:
                                c2_new=0

                            H_el=H_el+(((c1_new+c2_new)**2)*Av*0.5*k)-(((M[x])**2)*A_v[x]*0.5*k)
                            M[x]=c1_new+c2_new

                    
                    
       
    return part,H_el



#The two functions below toghether find the clusters and remove the particles of clusters contanining a number of faces greater than a treshold N_E
def DFS(i,et,te,part,cluster,visited):
    visited[i]=1
    cluster.append(i)
    for j in et[i]: #cycle over the edges forming the triangle i
        for z in te[j]: #triangles neighboring edge j
            if z!=i and z!=-1: #select the neighbor
                if part[z]==1 and visited[z]==0: #if particle present and not already visited
                    DFS(z,et,te,part,cluster,visited)
                    
def extraction(TRI,et,te,part,N_E):
    clusters=[]
    visited=np.zeros(len(TRI))
    for i in range(0,len(TRI)):
        if part[i]==1 and visited[i]==0:
            cluster=[]
            DFS(i,et,te,part,cluster,visited)
            if len(cluster)>N_E: #if cluster have number of elements greater than extraction number remove it
                for z in cluster: #remove all particles forming the cluster
                    part[z]=0
            else:
                if len(cluster)>2:
                    clusters.append(cluster) #store clusters present after extraction
    return clusters
                


# In[ ]:





# In[ ]:





#Uses calculation in calcolo_aggiornamento_curvatura.pdf, would be faster but doesn't work for now


import random
def particlesB(ver,TRI,H_el,k,k_I,k_D,g,μ,SHO,h,M,A_v,normals_face,normals_ver,part,et,ev,te,border,neigh): #h scalar curvatures at edges
    for i in range(0,len(TRI)): #particles insertion on empty faces
        if part[i]==0:
            if k_I>np.random.rand(): #particle inserted on face i, insert spontaneous curvature at all adjacent edges of -μ
                part[i]=1
                for j in et[i]: #et[i] edges forming i-th triangle 
                    if -1 not in te[j]: #border edges have zero curvature, don't update them
                        TR_Se=np.matrix.trace(SHO[:,:,j])
                        if h[j]!=0:
                            SHO[:,:,j]=SHO[:,:,j]/h[j]
                        h0=np.copy(h[j])
                        h[j]-=μ  #update scalar curvature and corresponding shape operator
                        SHO[:,:,j]=SHO[:,:,j]*h[j]
                        #update mean curvature of neighboring vertices
                        Nf_1=normals_face[te[j][0]]; Nf_2=normals_face[te[j][1]]
                        N_e=(Nf_1+Nf_2)
                        N_e=N_e/np.linalg.norm(N_e)
                        for x in ev[j]:
                            if border[x]==False:
                                N_v=normals_ver[x]
                                W_e=np.dot(N_v,N_e)
                                H_el=H_el-(((M[x])**2)*A_v[x]*0.5*k)
                                M[x]=M[x]-(μ*W_e*TR_Se)/(A_v[x]*h0)
                                H_el=H_el+(((M[x])**2)*A_v[x]*0.5*k) #update curvature energy with new mean curvature
        else: #particle already present, diffusive jump
            count=0
            neig_empty=[]
            for j in et[i]: #count occupied neighbors and find empty neighbours
                for z in te[j]:
                    if z!=i and z!=-1:
                        if part[z]==1: #occupied neighbour
                            count+=1
                        else:
                            neig_empty.append(z)
            if len(neig_empty)!=0: #if all neigh occupied no jump CHECK
                print("possible jump",i)
                p=k_D/(g**count)
                if p>np.random.rand(): #particle jump
                    n=random.randrange(0,len(neig_empty)) #choose random empty neighbour
                    part[i]=0
                    b=neig_empty[n]
                    part[b]=1
                    print("jump",i,b)
                    for j in et[i]: #update shape operators of neigh edges removing spontaneous curvature around i
                        if -1 not in te[j]:
                            h0=np.copy(h[j])
                            TR_Se=np.matrix.trace(SHO[:,:,j]) #trace of not yet updated shape operator
                            if h[j]!=0:
                                SHO[:,:,j]=SHO[:,:,j]/h[j]
                            h[j]+=μ
                            SHO[:,:,j]=SHO[:,:,j]*h[j]

                            #update mean curvature of neighboring vertices
                            vert=ev[j]
                            Nf_1=normals_face[te[j][0]]; Nf_2=normals_face[te[j][1]]
                            N_e=(Nf_1+Nf_2)
                            N_e=N_e/np.linalg.norm(N_e)
                            for x in vert: #will update twice each vertex, for every neighboring edge
                                if border[x]==False:
                                    N_v=normals_ver[x]
                                    W_e=np.dot(N_v,N_e)
                                    H_el=H_el-(((M[x])**2)*A_v[x]*0.5*k)
                                    M[x]=M[x]+(μ*W_e*TR_Se)/(A_v[x]*h0)
                                    H_el=H_el+(((M[x])**2)*A_v[x]*0.5*k) #update curvature energy with new mean curvature

                    
                    for j in et[b]:#insert spontaneous curvature on edges around b
                        if -1 not in te[j]:
                            TR_Se=np.matrix.trace(SHO[:,:,j])
                            h0=np.copy(h[j])
                            if h[j]!=0:
                                SHO[:,:,j]=SHO[:,:,j]/h[j]
                            h[j]-=μ
                            SHO[:,:,j]=SHO[:,:,j]*h[j]

                            #update mean curvature of neighboring vertices
                            vert=ev[j]
                            Nf_1=normals_face[te[j][0]]; Nf_2=normals_face[te[j][1]]
                            N_e=(Nf_1+Nf_2)
                            N_e=N_e/np.linalg.norm(N_e)
                            for x in vert: #will update twice each vertex, for every neighboring edge
                                if border[x]==False:
                                    N_v=normals_ver[x]
                                    W_e=np.dot(N_v,N_e)
                                    H_el=H_el-(((M[x])**2)*A_v[x]*0.5*k)
                                    M[x]=M[x]-(μ*W_e*TR_Se)/(A_v[x]*h0)
                                    H_el=H_el+(((M[x])**2)*A_v[x]*0.5*k) #update curvature energy with new mean curvatur
    
    
    return part,H_el

