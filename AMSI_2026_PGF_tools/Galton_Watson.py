import numpy as np
import networkx as nx
import random

def GaltonWatsonSimulation(OffspringDist, num_generations=10):
    """
    Simulate a Galton–Watson branching process.

    This function simulates a discrete-time branching process in which
    each individual in generation g independently produces a random
    number of offspring according to a specified offspring distribution.
    The process starts with a single individual in generation 0 and
    evolves forward for a fixed number of generations.

    Parameters
    ----------
    OffspringDist : callable
        A function (or other callable) with no arguments that returns
        a non-negative integer representing the number of offspring
        produced by a single individual. Each call is assumed to be
        independent and identically distributed.
    num_generations : int, optional (default=10)
        The number of generations to simulate. The simulation produces
        population sizes for generations g = 0, 1, ..., num_generations.

    Returns
    -------
    Xlist : numpy array of int
        A numpy array of length num_generations + 1, where Xlist[g] is the
        population size in generation g. In particular, Xlist[0] = 1
        corresponds to the initial population size.

    Notes
    -----
    - This is a *single realisation* of the Galton–Watson process.
      Repeated calls to this function are required to estimate
      extinction probabilities or other statistical properties.
    - If the population reaches zero at some generation g, all
      subsequent generations will also have population size zero.
    - The offspring distribution is assumed to be time-homogeneous
      (the same for all generations and individuals).

    """

    Xlist = np.zeros(num_generations + 1, dtype = int)  # array to hold population sizes
    Xlist[0] = 1  #set the initial X_0 = 1
    for gen in range(num_generations): #gen = 0, 1, ..., num_generations-1
        X_current = Xlist[gen]   # Current population size X_g
        X_next = 0               # Initialize X_{g+1}

        # Each individual in generation g produces offspring independently
        for j in range(X_current):
            offspring = OffspringDist()
            X_next += offspring

        # Append the population size of the next generation
        Xlist[gen+1] = X_next

    return Xlist
 

import numpy as np
import networkx as nx

def GaltonWatsonTreeSimulation(OffspringDist, num_generations=10):
    """
    Simulate a Galton–Watson branching process and generate its tree.

    Nodes are labeled as (g, j), corresponding to u_{g,j}, the j-th
    individual in generation g.

    Parameters
    ----------
    OffspringDist : callable
        A function returning a non-negative integer number of offspring.
    num_generations : int, optional (default=10)
        Number of generations to simulate (g = 0, ..., num_generations).

    Returns
    -------
    G : networkx.DiGraph
        Directed graph representing the Galton–Watson tree.
    Xlist : numpy.ndarray
        Array where Xlist[g] is the population size in generation g.
    """

    G = nx.DiGraph()

    # Generation 0: single root u_{0,0}
    current_generation = [(0, 1)]
    G.add_node((0, 1), generation=0, index=0)

    Xlist = np.zeros(num_generations + 1, dtype=int)
    Xlist[0] = 1

    for g in range(num_generations):
        next_generation = []
        child_index = 1  # j index for generation g+1

        for parent in current_generation:
            offspring = OffspringDist()

            for _ in range(offspring):
                child = (g + 1, child_index)
                G.add_node(child, generation=g + 1, index=child_index)
                G.add_edge(parent, child)

                next_generation.append(child)
                child_index += 1

        Xlist[g + 1] = len(next_generation)
        current_generation = next_generation

        if Xlist[g + 1] == 0:
            break  # extinction reached

    return G, Xlist







def Phi_values_on_unit_circle(g, num_points, mu):
    """
    Compute values of Phi_g(z) at equally spaced points on the unit circle:
        z_m = exp(2*pi*i*m/num_points),  m = 0,...,num_points-1

    Here mu is a function that acts on an array of complex values (pointwise),
    returning mu(values).
    """
    if g < 0 or not isinstance(g, int):
        raise ValueError("g must be a nonnegative integer")

    m = np.arange(num_points)
    z_points = np.exp(2j * np.pi * m / num_points)

    values = z_points.copy()   # Phi_0(z) = z
    for _ in range(g):
        values = mu(values)

    return values, z_points


def Phi_coefficients_via_cauchy(g, num_points, mu, FFT=True):
    """
    Compute coefficients of Phi_g(x) either using the discrete Cauchy integral
    (trapezoidal rule on the unit circle), or the FFT.

    Returns coefficients a_0, ..., a_{num_points-1}.
    """
    values, z_points = Phi_values_on_unit_circle(g, num_points, mu)

    if FFT:  # Fast Fourier Transform is equivalent to calculation below, but much faster
        #coefficients = np.fft.ifft(values)
        coefficients = np.fft.fft(values) / num_points
    else:
        coefficients = np.zeros(num_points, dtype=complex)
        for k in range(num_points):
            # a_k = (1/num_points) * sum_m Phi_g(z_m) * z_m^{-k}
            coefficients[k] = np.sum(values * z_points**(-k)) / num_points

    return coefficients


def Phi_coefficients_real(g, num_points, mu, tolerance=1e-12, FFT=True):
    """
    Return real coefficients when imaginary parts are numerical noise.
    """
    coeffs = Phi_coefficients_via_cauchy(g, num_points, mu, FFT=FFT)
    if np.max(np.abs(coeffs.imag)) < tolerance:
        return coeffs.real
    return coeffs




def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
    if G.number_of_nodes() == 1:
        node = next(iter(G.nodes))
        return {node: (0.5, vert_loc)}
    
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos