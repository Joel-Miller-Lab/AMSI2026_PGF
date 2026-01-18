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


import numpy as np
import matplotlib.pyplot as plt

def Gillespie(r, FindK, maxtime, rng = np.random.default_rng):
    time = 0
    PopSize = 1
    times = [time]
    sizes = [PopSize]
    while PopSize>0 and time<maxtime:
        TimeToNextEvent = rng.exponential(1/(PopSize*r))
        time += TimeToNextEvent
        k = FindK()
        PopSize += k-1
        times.append(time)
        sizes.append(PopSize)
    return times, sizes

def GillespieTree(r, FindK, maxtime, rng=None):
    """
    Continuous-time branching process simulation (Gillespie / SSA) that
    also records the genealogy as a directed tree.

    Model assumptions:
    - Each *alive* individual experiences "an event" at rate r.
      So with N alive individuals, total event rate is N*r.
    - At an event, we pick which individual experienced it uniformly
      among the alive individuals (because they all have the same rate).
    - FindK() returns an integer k:
        * k == 0  : the individual dies (population decreases by 1)
        * k >= 1  : the individual gives birth to (k-1) offspring and remains alive
      (So k=2 is a single birth, and k=0 is a death)

    Returns
    -------
    G_labeled : networkx.DiGraph
        A directed tree with nodes labeled (g, i). Node attributes include:
          - 'birth_time'
          - 'death_time' (None if still alive when the simulation stops)
          - 'generation' (g)
    times : list[float]
        Event times (including time 0).
    sizes : list[int]
        Population size after each event time.
    """

    if rng is None:
        rng = np.random.default_rng()

    # --- Build the tree using temporary integer node ids first ---
    G = nx.DiGraph()

    time = 0.0

    # Create the root individual
    next_temp_id = 0
    root = next_temp_id
    next_temp_id += 1

    G.add_node(
        root,
        generation=0,
        birth_time=0.0,
        death_time=None,
        parent_temp=None,
        birth_order_within_parent=None,
    )

    alive = [root]  # list of temp node ids of currently alive individuals

    times = [time]
    sizes = [len(alive)]

    # --- Gillespie loop ---
    while len(alive) > 0 and time < maxtime:
        N = len(alive)

        # Time to next event: exponential with mean 1/(N*r)
        dt = rng.exponential(scale=1.0 / (N * r))
        time += dt

        # If we stepped past maxtime, stop (do not record events beyond maxtime)
        if time > maxtime:
            break

        # Choose which individual experienced the event
        idx = rng.integers(0, N)
        u = alive[idx]

        # Determine what happens at the event
        k = FindK()

        if k == 0:
            # Death: mark death time and remove from alive list
            G.nodes[u]["death_time"] = time
            alive.pop(idx)

        else:
            # Birth: parent stays alive; create (k-1) offspring
            # All offspring have generation = parent_generation + 1
            parent_g = G.nodes[u]["generation"]
            child_g = parent_g + 1

            # birth_order_within_parent is used only to break ties
            # when siblings are born at the same time.
            for birth_order in range(k - 1):
                child = next_temp_id
                next_temp_id += 1

                G.add_node(
                    child,
                    generation=child_g,
                    birth_time=time,
                    death_time=None,
                    parent_temp=u,
                    birth_order_within_parent=birth_order,
                )
                G.add_edge(u, child)
                alive.append(child)

        times.append(time)
        sizes.append(len(alive))

    # --- Post-process: relabel temp ids -> (g, i) following ordering rules ---

    # Group nodes by generation
    gen_to_nodes = {}
    for node in G.nodes:
        g = G.nodes[node]["generation"]
        gen_to_nodes.setdefault(g, []).append(node)

    # Map temp_id -> final label (g, i)
    temp_to_label = {}

    # Generation 0: root only
    temp_to_label[root] = (0, 1)

    # Helper: within each generation, assign i by sorting with keys that enforce:
    # For generation g>=1, we rely on parent indices (already assigned in generation g-1).
    gmax = max(gen_to_nodes.keys()) if gen_to_nodes else 0

    for g in range(1, gmax + 1):
        nodes_g = gen_to_nodes.get(g, [])
        if not nodes_g:
            continue

        def sort_key(v):
            bt = G.nodes[v]["birth_time"]
            parent = G.nodes[v]["parent_temp"]
            parent_i = temp_to_label[parent][1]  # parent's index within generation g-1
            bo = G.nodes[v]["birth_order_within_parent"]
            # temp id as a final deterministic tie-breaker
            return (bt, parent_i, bo, v)

        nodes_g_sorted = sorted(nodes_g, key=sort_key)

        for i, v in enumerate(nodes_g_sorted):
            temp_to_label[v] = (g, i + 1)  # indices start at 1

    # Create a relabeled copy
    G_labeled = nx.relabel_nodes(G, temp_to_label, copy=True)

    return G_labeled, times, sizes


# Plotting stuff

def _get_root(G):
    """Return the unique root (node with in-degree 0)."""
    roots = [u for u in G.nodes if G.in_degree(u) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly 1 root, found {len(roots)}.")
    return roots[0]


def _children_sorted_by_birth_time(G, u):
    """Children of u sorted by birth_time, then by node id as a tie-breaker."""
    kids = list(G.successors(u))
    kids.sort(key=lambda v: (G.nodes[v]["birth_time"], v))
    return kids


def _leaf_order(G, root):
    """
    Deterministic left-to-right ordering of leaves via DFS,
    visiting children in increasing birth_time.
    """
    order = []

    def dfs(u):
        kids = _children_sorted_by_birth_time(G, u)
        if len(kids) == 0:
            order.append(u)
        else:
            for v in kids:
                dfs(v)

    dfs(root)
    return order


def _compute_x_positions_leaf_midpoint(G, root):
    """
    Base x positions:
      - leaves get x = 0,1,2,... in DFS leaf order
      - internal nodes get midpoint of descendant leaves
    """
    leaves = _leaf_order(G, root)
    x = {leaf: float(i) for i, leaf in enumerate(leaves)}

    def post(u):
        kids = _children_sorted_by_birth_time(G, u)
        if len(kids) == 0:
            return x[u]
        child_xs = [post(v) for v in kids]
        x[u] = 0.5 * (min(child_xs) + max(child_xs))
        return x[u]

    post(root)
    return x


def _apply_collision_offsets(G, x_base, eps=None):
    """
    If multiple nodes share exactly the same x_base, their lifelines overlap.
    This function adds small deterministic offsets so each node has its own lane.

    Strategy:
      - group nodes by x_base value
      - within each group, sort nodes by (birth_time, generation, node_id)
      - assign offsets evenly spaced around 0: ..., -d, 0, +d, ...
      - scale d so offsets are small relative to horizontal spacing
    """
    # Compute a sensible default epsilon from spacing between distinct base x values
    if eps is None:
        uniq = np.array(sorted(set(x_base.values())), dtype=float)
        if len(uniq) >= 2:
            gaps = np.diff(uniq)
            min_gap = float(np.min(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
        else:
            min_gap = 1.0
        eps = 0.25 * min_gap  # max spread within a lane group

    # Group nodes by base x (exact float match; here floats are dyadic midpoints so match is stable)
    groups = {}
    for u, xb in x_base.items():
        groups.setdefault(xb, []).append(u)

    x = dict(x_base)

    for xb, nodes in groups.items():
        if len(nodes) <= 1:
            continue

        # Deterministic ordering within the collision group
        nodes_sorted = sorted(
            nodes,
            key=lambda u: (
                G.nodes[u].get("birth_time", 0.0),
                G.nodes[u].get("generation", 0),
                u,
            ),
        )

        m = len(nodes_sorted)
        if m == 1:
            continue

        # Evenly spaced offsets spanning [-eps, +eps]
        if m == 2:
            offsets = [-0.5 * eps, 0.5 * eps]
        else:
            offsets = np.linspace(-eps, eps, m)

        for u, off in zip(nodes_sorted, offsets):
            x[u] = xb + float(off)

    return x


def plot_ct_tree_leaf_layout_with_offsets(
    G,
    maxtime=None,
    ax=None,
    node_size=25,
    lifeline_color="0.6",
    lifeline_alpha=0.8,
    branch_color="k",
    branch_linewidth=1.4,
    lifeline_linewidth=1.2,
    show_labels=False,
    label_fontsize=8,
    eps=None,
):
    """
    Plot a continuous-time branching tree using:
      - y = birth_time (time increases downward)
      - x from leaf-order/midpoint layout, plus small offsets to avoid lifeline collisions

    Visual encoding:
      - lifeline: vertical from birth_time to death_time (or maxtime if censored)
      - birth branch: horizontal at child's birth_time from parent x to child x
      - nodes: drawn at birth points only (no explicit death nodes)

    Required node attributes:
      - 'birth_time' : float
      - 'death_time' : float or None
      - 'generation' : int (only used for deterministic tie-breaking)
    """
    if ax is None:
        fig, ax = plt.subplots()

    root = _get_root(G)

    # Base x positions then apply offsets to prevent stacked lifelines
    x_base = _compute_x_positions_leaf_midpoint(G, root)
    x = _apply_collision_offsets(G, x_base, eps=eps)

    # Determine maxtime if not provided
    if maxtime is None:
        death_times = [
            G.nodes[u]["death_time"] for u in G.nodes
            if G.nodes[u].get("death_time", None) is not None
        ]
        if len(death_times) > 0:
            maxtime = float(max(death_times))
        else:
            maxtime = float(max(G.nodes[u]["birth_time"] for u in G.nodes))

    # Birth times
    yb = {u: float(G.nodes[u]["birth_time"]) for u in G.nodes}

    # Draw lifelines first (behind everything)
    for u in G.nodes:
        bt = yb[u]
        dt = G.nodes[u].get("death_time", None)
        if dt is None:
            dt = maxtime
        dt = float(dt)

        if dt > bt:
            ax.plot(
                [x[u], x[u]],
                [bt, dt],
                color=lifeline_color,
                alpha=lifeline_alpha,
                linewidth=lifeline_linewidth,
                zorder=0,
            )

    # Draw birth branches: from parent's lifeline to child's birth dot at child's birth time
    for parent, child in G.edges():
        yc = yb[child]
        ax.plot(
            [x[parent], x[child]],
            [yc, yc],
            color=branch_color,
            linewidth=branch_linewidth,
            zorder=1,
        )

    # Draw birth nodes (no death nodes)
    ax.scatter(
        [x[u] for u in G.nodes],
        [yb[u] for u in G.nodes],
        s=node_size,
        zorder=2,
    )

    if show_labels:
        for u in G.nodes:
            ax.text(
                x[u], yb[u], str(u),
                fontsize=label_fontsize,
                ha="center", va="bottom",
                zorder=3,
            )

    # Axis styling: time increases downward
    ax.set_xlabel("lineage position (leaf-order layout with offsets)")
    ax.set_ylabel("time")
    ax.set_ylim(maxtime, 0.0)
    ax.set_title("Continuous-time genealogy (leaf-order layout)")
    ax.margins(x=0.05)

    return ax
