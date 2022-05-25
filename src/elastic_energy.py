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
#Calculate curvature at each vertex following Ramakrishnan Mesoscale computational studies of membrane bilayer
#remodeling by curvature-inducing proteins or Monte Carlo simulations of fluid vesicles with in plane orientational ordering

def Elastic_Local(ver,TRI,k,normals_ver,normals_face,area,neigh,ADJ,NI,et,te,border,μ,part):
    aus=0
    curv_E=[]
    f1=[]; f2=[]; eigenv=[]
    h=np.zeros(len(te))
    M=np.zeros(len(ver))
    SHO=np.zeros([3,3,len(te)]) #matrix that store shape operator at each edge
    A_v=np.zeros(len(ver))
    for i in range(0,len(ver)): #calculate curvature at each vertex
        if border[i]==True:
            f1.append(0.0); f2.append(0.0) #zero curvature at border vertices
        else:
            #Calculate Area associated to each vertex for later use
            neig_faces_i=ADJ[NI[i]:NI[i+1]] #faces neighbouring vertex i
            for m in neig_faces_i:
                A_v[i]+=area[m]
            A_v[i]=A_v[i]/3
            N_v=normals_ver[i]#Normal to vertex
            S_v=np.zeros([3,3])
            for j in neigh[i]:
                r_e=ver[j]- ver[i] #vector connecting central vertex to adjacent ones
           
                #to find faces that share edge connecting i,j without searching ev
                neig_faces_j=ADJ[NI[j]:NI[j+1]]
                faces=[]
                for q in neig_faces_i:
                    if q in neig_faces_j:
                        faces.append(q)
                        
                ed1=[] #TO FIND FACES IN COUNTERCLOCKWISE ORDER, S.T. DIHEDRAL ANGLE SIGN IS CORRECT (give cross product involved in a consistent order)
                ed2=[]
                for z in range(0,3): #uses the fact that triangle indices are spatially in a counterclockwise order
                    if TRI[faces[0]][z]==i: #ed will be the two "border" vertices of the hexagon in counterclockwise order
                        ed1.append(TRI[faces[0]][(z+1)%3])
                        ed1.append(TRI[faces[0]][(z+2)%3])#if z=0 take 1,2 if z=1 take 2,0 if z=2 take 0,1 
                    if TRI[faces[1]][z]==i:
                        ed2.append(TRI[faces[1]][(z+1)%3])
                        ed2.append(TRI[faces[1]][(z+2)%3])
                if ed1[1]==ed2[0]: #if they are already in counter clockwise order
                    Nf_1=normals_face[faces[0]]
                    Nf_2=normals_face[faces[1]]
                else: #if not exchange them
                    Nf_1=normals_face[faces[1]]
                    Nf_2=normals_face[faces[0]]
    
                et1=et[faces[0]]; et2=et[faces[1]] #list of edges used by faces considered
                e=0
                for z in et1:
                    if z in et2:
                        e=z #index of edge on which calculating curvature
                #find signed dihedral angle
                ϕ=np.sign(np.dot(np.cross(Nf_1,Nf_2),r_e))*np.arccos(np.dot(Nf_1,Nf_2))+np.pi
                #now can find edge curvature
                H_e=2*np.linalg.norm(r_e)*np.cos(0.5*ϕ)
                #To calculate shape operator calculate first edge normal using face normals
                N_e=(Nf_1+Nf_2)
                N_e=N_e/np.linalg.norm(N_e)
                #Shape operator on edge e
                R_e=r_e/np.linalg.norm(r_e)
                b=np.cross(R_e,N_e)
                b=b/np.linalg.norm(b)
                for tri in te[e]: #add -mu to scalar curvature for every adjacent face occupied by a particle
                    if tri!=-1:
                        if part[tri]==1:
                            H_e-=μ
                S_e=H_e*np.tensordot(b,b,axes=0) #Shape operator
                h[e]=H_e
                SHO[:,:,e]=S_e #Save shape operator in position corresponding to edge
                P_v=np.identity(3)-np.tensordot(N_v,N_v,axes=0) #Projection operator
                #Project shape operator on edge and obtain shape operator at vertex
                #print(N_v," ",N_e)
                #print(np.dot(N_e,N_v))
                W_e=np.dot(N_v,N_e)
                S_v+=W_e*np.dot((np.conjugate(P_v.transpose())),np.dot(S_e,P_v))


            S_v=S_v/A_v[i] #vertex shape operator has one zero eigenvalue and the other two are the curvatures
            #print(np.dot(S_v,N_v))# N_v IS eigenvector with zero eigenvalue
            
            #Householder transformation
            x=[0,0,1] #start from canonical reference frame
            pvec=x+N_v
            mvec=x-N_v
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
                    c1=((p+s)+np.sqrt(insq))*0.5  #largest eigenvalue                            
                    c2=((p+s)-np.sqrt(insq))*0.5  #smallest
                else: #degenerate eigenvalues
                    c1=(p+s)*0.5                               
                    c2=c1
            else:
                if p>s:
                    c1=p
                    c2=s
                else:
                    c1=s
                    c2=p

                    
            if abs(c1)<=10e-10:
                c1=0
            if abs(c2)<=10e-10:
                c2=0
            f1.append(c1)
            f2.append(c2)
            M[i]=c1+c2
            aus+=((c1+c2)**2)*A_v[i]*0.5*k
            #curv_E.append(((c1+c2)**2)*A_v[i]*0.5*k)

    
    H_elastic=aus
    return H_elastic,M,A_v,SHO,h



#VERTEX
#Update energy after vertex move, calculates new curvature and energy contribution of shifted vertex


def update_energy_vertex(ver,TRI,H_old,M,k,A_v,i,normals_face,neigh,ADJ,NI,area,SHO,et,te,h,part,μ):
    #aus=H_old-(((c1[i]+c2[i])**2)*A_v[i]*0.5*k)
    N_v=np.zeros(3)
    Nf_old=[] #list with old face normals, areas and associated indices
    Av_new=0
    neig_faces_i=ADJ[NI[i]:NI[i+1]]
    SH_old=np.zeros([3,3,len(neigh[i])]) #Shape operators of old edges
    ind=[] #corresponding indices
    h_old=np.zeros(len(neigh[i])) #old scalar curvatures, same indices as shape operators
    for m in neig_faces_i: #calculate new vertex area, new face normals, new vertex normal
        x,y,z=ver[TRI[m][0]],ver[TRI[m][1]],ver[TRI[m][2]]
        α=x-y; β=x-z
        A_face=0.5*np.linalg.norm(np.cross(α,β)) #area of j-th neig face
        Nf=np.cross(α,β)/np.linalg.norm(np.cross(α,β)) #new normal of j-th neig face
        Nf_old.append([np.copy(normals_face[m]),area[m],m])
        N_v+=Nf*A_face
        Av_new+=A_face
        area[m]=A_face #update lists, then in MCvertex will put back old ones if vertex was put back
        normals_face[m]=Nf
        
    N_v=N_v/np.linalg.norm(N_v)
    Av_new=Av_new/3
    count=0
    S_v=np.zeros([3,3])
    for j in neigh[i]: #calculate new energy contribution of shifted vertex
        r_e=ver[j]- ver[i] #vector connecting central vertex to adjacent ones
        #to find faces that share edge connecting i,j without searching ev
        neig_faces_j=ADJ[NI[j]:NI[j+1]]
        faces=[]
        for q in neig_faces_i:
            if q in neig_faces_j:
                faces.append(q)
        ed1=[] #TO FIND FACES IN COUNTERCLOCKWISE ORDER, S.T. DIHEDRAL ANGLE SIGN IS CORRECT
        ed2=[]
        for z in range(0,3):
            if TRI[faces[0]][z]==i:
                ed1.append(TRI[faces[0]][(z+1)%3])
                ed1.append(TRI[faces[0]][(z+2)%3])
            if TRI[faces[1]][z]==i:
                ed2.append(TRI[faces[1]][(z+1)%3])
                ed2.append(TRI[faces[1]][(z+2)%3])
        if ed1[1]==ed2[0]:
            Nf_1=normals_face[faces[0]]
            Nf_2=normals_face[faces[1]]
        else:
            Nf_1=normals_face[faces[1]]
            Nf_2=normals_face[faces[0]]
        et1=et[faces[0]]; et2=et[faces[1]] #list of edges used by faces considered
        e=0
        for z in et1:
            if z in et2:
                e=z #index of edge on which calculating curvature
        SH_old[:,:,count]=np.copy(SHO[:,:,e]) #store old shape op
        h_old[count]=h[e] #store old scalar curvature
        ind.append(e) #edge e centered in vertex i
        count+=1
                
        ϕ=np.sign(np.dot(np.cross(Nf_1,Nf_2),r_e))*np.arccos(np.dot(Nf_1,Nf_2))+np.pi
        #now can find edge curvature
        H_e=2*np.linalg.norm(r_e)*np.cos(0.5*ϕ)
        #To calculate shape operator calculate first edge normal using face normals
        N_e=(Nf_1+Nf_2)
        N_e=N_e/np.linalg.norm(N_e)
        #Shape operator on edge e
        R_e=r_e/np.linalg.norm(r_e)
        b=np.cross(R_e,N_e)
        for tri in te[e]: #add -mu to scalar curvature for every adjacent face occupied by a particle
            if tri!=-1:
                if part[tri]==1:
                    H_e-=μ
        h[e]=H_e
        S_e=H_e*np.tensordot(b,b,axes=0) #Shape operatordot()
        SHO[:,:,e]=S_e #update shape operator related to edge e 
        
        P_v=np.identity(3)-np.tensordot(N_v,N_v,axes=0) #Projection operator
        #Project shape operator on edge and obtain shape operator at vertex
        #print(N_v," ",N_e)
        #print(np.dot(N_e,N_v))
        W_e=np.dot(N_v,N_e)
        S_v+=W_e*np.dot((np.conjugate(P_v.transpose())),np.dot(S_e,P_v))

        
    S_v=S_v/Av_new #vertex shape operator has one zero eigenvalue and the other two are the curvatures
    
    
    #Householder transformation
    x=[0,0,1] #start from canonical reference frame
    pvec=x+N_v
    mvec=x-N_v
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

    H_new=H_old+(((c1_new+c2_new)**2)*Av_new*0.5*k)-(((M[i])**2)*A_v[i]*0.5*k) #M[i]=c1[i]+c2[i]
    M_new=c1_new+c2_new

    return H_new, Av_new, Nf_old, N_v,M_new,SH_old,ind,h_old




#Calculates new curvature and energy contributions of neighborhood of shifted VERTEX. All face adjacent to shifted vertex changed normals and areas, changing normals areas of neigh vertices and shape operators of edges neighboring changed faces. Calculates only the new shape operators.

def update_energy_neig(ver,TRI,H_old,M,k,A_v,i,normals_face,neigh,ADJ,NI,area,SHO,et,te,border,h,part,μ): #to be used after update energy vertex,uses already updated lists of face normals and areas
    H_new=H_old
    Avj=[]
    SH_old_neig=np.zeros([3,3,2*len(neigh[i])])
    h_old_neig=np.zeros(2*len(neigh[i]))
    M_neig=[]
    ind_c=[]
    ind_neig=[]
    count=0
    visited=[]
    for j in neigh[i]: #Calculate new normals and vertex areas
        if border[j]==False: #new curvature of j only if not border vertex 
            N_v=0; Av=0
            SH_old=np.zeros([3,3,len(neigh[i])]) #Shape operators of old edges
            ind=[] #corresponding indices
            neig_faces_j=ADJ[NI[j]:NI[j+1]]
            for m in neig_faces_j:
                Nf=normals_face[m]
                Af=area[m]
                N_v+=Nf*Af
                Av+=Af
            N_v=N_v/np.linalg.norm(N_v)
            Av=Av/3
            Avj.append([Av,j])
            S_v=np.zeros([3,3])
            for z in neigh[j]: #Calculate new curvatures
                neig_faces_z=ADJ[NI[z]:NI[z+1]]
                faces=[]
                for q in neig_faces_z:
                    if q in neig_faces_j:
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
                if z in neigh[i] and e not in visited: #calculate new shape operator for edges neigh triangles that changed face normal
                    SH_old_neig[:,:,count]=np.copy(SHO[:,:,e]) #store old shape op
                    h_old_neig[count]=h[e]
                    ind_neig.append(e) #edge e centered in vertex i
                    count+=1
                    r_e=ver[z]- ver[j]

                    ed1=[]; ed2=[]
                    for w in range(0,3):#give normals in counterclockwise order
                        if TRI[faces[0]][w]==j:
                            ed1.append(TRI[faces[0]][(w+1)%3])
                            ed1.append(TRI[faces[0]][(w+2)%3])
                        if TRI[faces[1]][w]==j:
                            ed2.append(TRI[faces[1]][(w+1)%3])
                            ed2.append(TRI[faces[1]][(w+2)%3])
                    #print(faces,ed1,ed2)
                    if ed1[1]==ed2[0]:
                        Nf_1=normals_face[faces[0]]
                        Nf_2=normals_face[faces[1]]
                    else:
                        Nf_1=normals_face[faces[1]]
                        Nf_2=normals_face[faces[0]]

                    ϕ=np.sign(np.dot(np.cross(Nf_1,Nf_2),r_e))*np.arccos(np.dot(Nf_1,Nf_2))+np.pi
                    #now can find edge curvature
                    H_e=2*np.linalg.norm(r_e)*np.cos(0.5*ϕ)
                    R_e=r_e/np.linalg.norm(r_e)
                    b=np.cross(R_e,N_e)
                    for tri in te[e]: #add -mu to scalar curvature for every adjacent face occupied by a particle
                        if tri!=-1:
                            if part[tri]==1:
                                H_e-=μ
                    h[e]=H_e
                    S_e=H_e*np.tensordot(b,b,axes=0) #Shape operatordot() 
                    SHO[:,:,e]=S_e #update shape operator related to edge e centered in i
                    visited.append(e)

                else:
                    S_e=SHO[:,:,e] #extract shape operator corresponding to edge considered (edges neigh shifted vertex already updated in vertex)
                S_v+=W_e*np.dot((np.conjugate(P_v.transpose())),np.dot(S_e,P_v))

            S_v=S_v/Av #vertex shape operator has one zero eigenvalue and the other two are the curvatures
            #print(np.dot(S_v,N_v))

            #Householder transformation

            x=[0,0,1] #start from canonical reference frame
            pvec=x+N_v
            mvec=x-N_v
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
            M_neig.append(c1_new+c2_new)
            ind_c.append(j)
            H_new=H_new+(((c1_new+c2_new)**2)*Av*0.5*k)-(((M[j])**2)*A_v[j]*0.5*k)
        #print(H_new)
    return H_new,Avj,SH_old_neig,ind_neig,h_old_neig,M_neig,ind_c


#LINKFLIP

#Updates list of faces adjacent to each vertex after linkflip
def updateADJ(x,y,n,ev_new,l,t,ADJ,NI,ev,TRI): #TRI must be already updated with new triangles in pos l and t, IF NOT GIVES ERRORS
    for i in ev_new: #vertices connected by new edge neighbors of both faces
        if l in ADJ[NI[i]:NI[i+1]]:
            ADJ=np.concatenate((ADJ[:NI[i+1]],[t],ADJ[NI[i+1]:]))
            NI=np.concatenate((NI[:i+1],NI[i+1:]+1))
        else:
            ADJ=np.concatenate((ADJ[:NI[i+1]],[l],ADJ[NI[i+1]:]))
            NI=np.concatenate((NI[:i+1],NI[i+1:]+1))
    for j in ev[n]: #vertices of old edge, need to remove lost neigh face
        if j in TRI[l]: #if j is in l-th new triangle, t must be removed from its neig
            pos=0
            for z in range(NI[j],NI[j+1]): #find position of t
                if ADJ[z]==t:
                    pos=z
                    break
            ADJ=np.concatenate((ADJ[:pos],ADJ[pos+1:]))
            NI=np.concatenate((NI[:j+1],NI[j+1:]-1))
            
        else: #if j is in t-th new triangle, l must be removed from its neig
            pos=0
            for z in range(NI[j],NI[j+1]): #find position of l
                if ADJ[z]==l:
                    pos=z
                    break
            ADJ=np.concatenate((ADJ[:pos],ADJ[pos+1:]))
            NI=np.concatenate((NI[:j+1],NI[j+1:]-1))
    
                
            
    return ADJ,NI

#Updates curvature of vertices forming the tethrahedron around the flipped edge, all shape operators forming the tethrahedron change.

def update_energy_link(ver,ev,TRI,neig,n,H_old,M,k,A_v,area,normals_face,x,y,ev_new,l,t,ADJ,NI,SHO,et,te,border,h,part,μ): #x,y old triangles, l,t indices of corresponding new triangles
    tetra=np.concatenate((ev_new,ev[n])) #curvature changed in four vertices around flipped edge
    H_new=H_old
    ADJ_new,NI_new=updateADJ(x,y,n,ev_new,l,t,ADJ,NI,ev,TRI) #updated list of neigh faces
    Nf_old=[] #old face normals, area and indices
    M_tetra=[]; ind_c=[]
    neig[ev[n][0],ev[n][1]]=0; neig[ev[n][1],ev[n][0]]=0 #update neig matrix, put back if move fails
    neig[ev_new[0],ev_new[1]]=1; neig[ev_new[1],ev_new[0]]=1
    
    #UPDATE TOPOLOGY, NECESSARY TO HAVE CORRECT INDICIZATION FOR SHAPE OPERATORS ON EDGES
    ev_old=np.copy(ev[n]) #store old topology, use if move fails
    ev[n]=ev_new #update ev in the only changed edge, if move fails put back
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
    

    #edge was flipped, neig faces areas and normals changed
    for m in te[n]: #n index of flipped edge, in te[n] indices of neig triangles
        x,y,z=ver[TRI[m][0]],ver[TRI[m][1]],ver[TRI[m][2]]
        α=x-y; β=x-z
        A_face=0.5*np.linalg.norm(np.cross(α,β)) #area of neig face
        Nf=np.cross(α,β)/np.linalg.norm(np.cross(α,β)) #new normal of neig face
        Nf_old.append([np.copy(normals_face[m]),area[m],m])
        area[m]=A_face #update lists, then in MClink will put back old ones if vertex was put back
        normals_face[m]=Nf
    Avj=[]   
    visited=[]
    count=0
    SH_old_tetra=np.zeros([3,3,5]) #Shape operators of old edges, at most 5 edges to be updated
    h_old_tetra=np.zeros(5)
    ind_tetra=[] #corresponding indices
    for j in tetra:#Calculate new curvatures
        if border[j]==False:
            Av_new=0
            N_v=0
            #calculate new vertex area and normal
            neig_faces_j=ADJ_new[NI_new[j]:NI_new[j+1]]

            for m in neig_faces_j: #calculate new vertex area, new vertex normal
                N_v+=normals_face[m]*area[m]
                Av_new+=area[m]

            N_v=N_v/np.linalg.norm(N_v)
            Av_new=Av_new/3
            Avj.append([Av_new,j])
            neig_j=neig[j,:].rows[0]
            S_v=np.zeros([3,3])   
            for z in neig_j: #Calculate curvature contributions of neighboursì faces
                neig_faces_z=ADJ_new[NI_new[z]:NI_new[z+1]]
                faces=[]
                for q in neig_faces_z:
                    if q in neig_faces_j:
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
                if z in tetra and e not in visited: #calculate new shape operator for edges neigh triangles that changed face normal
                    SH_old_tetra[:,:,count]=np.copy(SHO[:,:,e]) #store old shape op
                    h_old_tetra[count]=np.copy(h[e])
                    ind_tetra.append(e) #edge e 
                    count+=1
                    r_e=ver[z]- ver[j]

                    ed1=[]; ed2=[]
                    for w in range(0,3):#give normals in counterclockwise order
                        if TRI[faces[0]][w]==j:
                            ed1.append(TRI[faces[0]][(w+1)%3])
                            ed1.append(TRI[faces[0]][(w+2)%3])
                        if TRI[faces[1]][w]==j:
                            ed2.append(TRI[faces[1]][(w+1)%3])
                            ed2.append(TRI[faces[1]][(w+2)%3])
                    if ed1[1]==ed2[0]:
                        Nf_1=normals_face[faces[0]]
                        Nf_2=normals_face[faces[1]]
                    else:
                        Nf_1=normals_face[faces[1]]
                        Nf_2=normals_face[faces[0]]
                    ϕ=np.sign(np.dot(np.cross(Nf_1,Nf_2),r_e))*np.arccos(np.dot(Nf_1,Nf_2))+np.pi
                    #now can find edge curvature
                    H_e=2*np.linalg.norm(r_e)*np.cos(0.5*ϕ)
                    R_e=r_e/np.linalg.norm(r_e)
                    b=np.cross(R_e,N_e)
                    for tri in te[e]: #add -mu to scalar curvature for every adjacent face occupied by a particle
                        if tri!=-1:
                            if part[tri]==1:
                                H_e-=μ
                    h[e]=H_e
                    S_e=H_e*np.tensordot(b,b,axes=0) #Shape operator
                    SHO[:,:,e]=S_e #update shape operator related to edge e
                    visited.append(e)

                else:
                    S_e=SHO[:,:,e] #extract shape operator corresponding to edge considered (edges neigh shifted vertex already updated in vertex)
                S_v+=W_e*np.dot((np.conjugate(P_v.transpose())),np.dot(S_e,P_v))

            S_v=S_v/Av_new #vertex shape operator has one zero eigenvalue and the other two are the curvatures
                #print(np.dot(S_v,N_v))
            #Householder transformation

            x=[0,0,1] #start from canonical reference frame
            pvec=x+N_v
            mvec=x-N_v
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

            #print(c2_new)


            if abs(c1_new)<=10e-10:
                c1_new=0
            if abs(c2_new)<=10e-10:
                c2_new=0
            M_tetra.append(c1_new+c2_new)
            ind_c.append(j)
            H_new=H_new+(((c1_new+c2_new)**2)*Av_new*0.5*k)-(((M[j])**2)*A_v[j]*0.5*k)
        #print(H_new)
    return H_new,Avj,Nf_old,SH_old_tetra,h_old_tetra,ind_tetra,M_tetra,ind_c,ADJ_new,NI_new,ev_old









#OLD ELASTIC ENERGY
def elastic_energy(ver,TRI,k):
    pd1, pd2, c1, c2 = igl.principal_curvature(ver, TRI,2) #principal curvatures
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


