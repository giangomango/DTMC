#!/usr/bin/env python
# coding: utf-8

# In[1]:



import scipy as sp
import numpy as np
import random
import math
from matplotlib import pyplot as plt
import os
root_folder = os.getcwd()

from matplotlib import tri


# In[2]:
#Calculates set of poiints forming a triangular lattice in 2D

def triangular_lattice(point_distance ,expansion_level = 4, starting_point = (0, 0)):

    set_of_points_on_the_plane = set()

    for current_level in range(expansion_level+1):

        temporary_hexagon_coordinates = {}

        equilateral_triangle_side = current_level * point_distance
        equilateral_triangle__half_side = equilateral_triangle_side / 2
        equilateral_triangle_height = (math.sqrt(3) * equilateral_triangle_side) / 2
        if current_level != 0:
            point_distance_as_triangle_side_percentage = point_distance / equilateral_triangle_side

        temporary_hexagon_coordinates['right'] = (starting_point[0] + point_distance * current_level, starting_point[1]) #right
        temporary_hexagon_coordinates['left'] = (starting_point[0] - point_distance * current_level, starting_point[1]) #left
        temporary_hexagon_coordinates['top_right'] = (starting_point[0] + equilateral_triangle__half_side, starting_point[1] + equilateral_triangle_height) #  top_right
        temporary_hexagon_coordinates['top_left'] = (starting_point[0] - equilateral_triangle__half_side, starting_point[1] + equilateral_triangle_height) #  top_left
        temporary_hexagon_coordinates['bottom_right'] = (starting_point[0] + equilateral_triangle__half_side, starting_point[1] - equilateral_triangle_height) #  bottom_right
        temporary_hexagon_coordinates['bottom_left'] = (starting_point[0] - equilateral_triangle__half_side, starting_point[1] - equilateral_triangle_height) # bottom_left

        if current_level > 1:
            for intermediate_points in range(1, current_level):

                set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][1] - temporary_hexagon_coordinates['left'][1])  ))        #from left to top left
                print(intermediate_points)
                print((temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][1] - temporary_hexagon_coordinates['left'][1])  ))
                set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['left'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['bottom_left'][0] - temporary_hexagon_coordinates['left'][0]) , temporary_hexagon_coordinates['left'][1] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['bottom_left'][1] - temporary_hexagon_coordinates['left'][1]) ))  # from left to bottom left

                set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['top_right'][0] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['top_left'][0] - temporary_hexagon_coordinates['top_right'][0]) , temporary_hexagon_coordinates['top_right'][1] ))  #from top right to top left
                set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['top_right'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][0] - temporary_hexagon_coordinates['top_right'][0]) , temporary_hexagon_coordinates['top_left'][1] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][1] - temporary_hexagon_coordinates['top_right'][1]) ))    # from top right to right

                set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['bottom_right'][0] - intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['bottom_left'][0] - temporary_hexagon_coordinates['bottom_right'][0]) , temporary_hexagon_coordinates['bottom_right'][1] ))   #apo bottom right pros aristera
                set_of_points_on_the_plane.add( (temporary_hexagon_coordinates['bottom_right'][0] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][0] - temporary_hexagon_coordinates['bottom_right'][0]) , temporary_hexagon_coordinates['bottom_right'][1] + intermediate_points * point_distance_as_triangle_side_percentage * abs(temporary_hexagon_coordinates['right'][1] - temporary_hexagon_coordinates['bottom_right'][1]) ))         # from bottom right to right

        # dictionary to set

        set_of_points_on_the_plane.update( temporary_hexagon_coordinates.values() )

    return list(set_of_points_on_the_plane)


# In[3]:

#Create the actual triangulation of 2D points and add a z coordinate
from scipy.spatial import Delaunay

def create_mesh(triangular_lattice):
    TRI_SCIPY=Delaunay(triangular_lattice)
    TRI=TRI_SCIPY.simplices

    ver=[]
    for i in range(0,len(triangular_lattice[:,0])):
            a=[triangular_lattice[i,0],triangular_lattice[i,1],random.uniform(0,1)]
            ver.append(a)
    ver=np.array(ver)
    return ver,TRI


# In[ ]:

def linked_list_cell(ver):
    max_x=max(ver[:,0]) #DEFINE LINKED-LIST CELL
    max_y=max(ver[:,1])
    max_z=max(ver[:,2])
    Lx=int(5*max_x) #define space of simulation as box 5 times x_max to ensure particles don't exit during simulation
    Ly=int(5*max_y)
    Lz=int(5*max_z)
    
    linklis=np.zeros(len(ver)) #vector of pointers for each particle to preceeding particle in same cell. -1 if no other particl
    header=np.full(Lx*Ly*Lz,-1) #vector of headers containing last particle inserted in each cell

    for i in range(0,len(ver)):
            cx,cy,cz= ver[i][0] // 1, ver[i][1] // 1, ver[i][2] // 1 #floor division to find cartesian coordinates of cell
            c=int(cx*Ly*Lz+cy*Lz+cz) #linear coordinate
            if header[c]==-1: #if cell empty i-th element that is being inserted points to empty linklis[i]=-1
                linklis[i]=-1
            else:
                linklis[i]=header[c] #point i-th element to the preceding one already contained in same cell
            header[c]=i #update header of the cell
    L=[Lx,Ly,Lz]
    return header,linklis,L

        
        
        
        
        
        


