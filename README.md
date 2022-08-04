DTMC

generate_mesh contains:

1. triangular_lattice, generates planar triangular lattice, can change spacing and number of points.
2. create_planar_mesh, performs Delauney triangulation using triangular lattice
3. create_spherical_mesh, creates triangulated spherical surface
4. linked_list_cell, defines linked list that will be used by MC_vertex, and assign vertices to corresponding cells in 3D space.


MCstep_vertex_link contains:
1. MC_step_vertex, tries to move N vertices using the linklist check that there is no overlapping between hard spheres potential and updates the elastic energy locally. Border vertices are fixed.
2. MC_step_link tries to flip N links, updates the energy, the face adjacency list and the edge topology locally.

elastic_energy contains:
1. Elastic_Local, calculates curvature at each vertex and corresponding energy, assign zero curvature at border vertices. Defines also list of Shape operators at each edge.
2. update_energy_vertex, update elastic energy contribution of shifted vertex
3. update_energy_neig, updates elastic energy contribution of neighborhood of shifted vertex
4. updateADJ, updates vertex face adjacency list after linkflip
5. update_energy_link, updates curvature of neig tethrahedron after linkflip

particles contains:
1. particlesA, particles insertion and diffusion. To update energy calculates new curvature in each corner of triangle from which particle inserted/removed
2. DFS, search connected components of particles onto the surface
3. extraction, removes clusters larger than a certain size. Do not update curvature, with the idea that some relaxation time is necessary for the configuration to relax back. Curvature is updated during the subsequent MC dynamics.
