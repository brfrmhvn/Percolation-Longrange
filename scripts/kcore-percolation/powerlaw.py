import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import math
from joblib import Parallel, delayed
from typing import List
import time
import os
import pickle
from numba import jit

st = time.time()
func = 'powl' #function for powerlaw

zetalist = [0,0.5,1,1.5,2,3,4,5,10] # alphalist
save_path = f'../{func}/'  # your save path
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path+'data', exist_ok=True)
os.makedirs(save_path+'result', exist_ok=True)

runs=20 # independent runs

L = 100  # Lattice size
avg_degree = 20  # Average degree

# initial state
ini_pp=0.9
end_p=0.6
step_p=0.001


@jit(nopython=True)
def power_law(dist, zeta):
    """Calculate the probability based on the exponential distribution."""
    return (dist) ** (-zeta) 

@jit(nopython=True)
def distance(i, j, zeta, L, func):
    # periodic boundary
    dx = min(abs(i[0] - j[0]), L - abs(i[0] - j[0]))
    dy = min(abs(i[1] - j[1]), L - abs(i[1] - j[1]))
    distance = math.sqrt(dx**2 + dy**2)
    probability = power_law(distance, zeta)
    return probability

def calculate_link_probability(positions, zeta, L, func):
    """Calculate the probability of each link existing based on the exponential distribution."""
    N = len(positions)
    link_probabilities = {}

    for i in range(N):
        for j in range(i + 1, N):
            link_probabilities[(i, j)] = distance(positions[i], positions[j], zeta, L, func)

    return link_probabilities

def select_links(link_probabilities, total_links):
    """Select links based on their probabilities using np.random.choice."""
    # Extract links and their corresponding probabilities
    links, probabilities = zip(*link_probabilities.items())

    # Normalize probabilities to sum up to 1
    total_prob = sum(probabilities)
    normalized_probabilities = [p / total_prob for p in probabilities]

    # Randomly select links based on their probabilities
    selected_indices = np.random.choice(len(links), size=total_links, replace=False, p=normalized_probabilities)
    selected_links = [links[i] for i in selected_indices]

    return selected_links

    
def giant_component_size_per_step(G: nx.Graph, k: int) -> List[int]:
    # paralell as the procalation figure
    def nodes_to_remove(G, k):
        """Determine nodes with in-degree and out-degree less than k."""
        return [node for node in G.nodes() if G.degree(node) < k]
        # return [node for node in G.nodes() if G.in_degree(node) < k and G.out_degree(node) < k]
    sizes = []
    to_remove = nodes_to_remove(G, k)

    # If no nodes need to be removed initially, add the size of the giant component of the initial graph
    if not to_remove and G.number_of_nodes() > 0:
        giant_component = max(nx.connected_components(G), key=len)
        sizes.append(len(giant_component))
    if not to_remove and G.number_of_nodes() == 0:
        sizes.append(0)
    
    # Perform the iterative removal process
    while to_remove:
        G.remove_nodes_from(to_remove)
        if G.number_of_nodes() == 0:
            sizes.append(0)
            break
    
        giant_component = max(nx.connected_components(G), key=len)
        sizes.append(len(giant_component))
        to_remove = nodes_to_remove(G, k) 

    return sizes


def main(zeta, func, ini_p, end_p, step_p):
    
    # Parameters
    zeta = zeta  # Parameter for the function distribution
    k = avg_degree//2 # k-core, suggest average / 2
    
    N = L ** 2  # network nodes
    total_links = N * avg_degree // 2  # network links
   
    # Create network positions
    positions = {i: (i % L, i // L) for i in range(N)}
    
    #### generate new G
    # Calculate link probabilities
    link_probabilities = calculate_link_probability(positions, zeta, L, func)
    # Select links using np.random.choice
    selected_links= select_links(link_probabilities, total_links)

    G = nx.Graph()
    G.add_nodes_from(positions.keys())
    G.add_edges_from(selected_links)

    print(f'graph done - cost time: {(time.time()-st)/60} min')
    
    with open(save_path+f'data/{L}_avedeg{avg_degree}_{func}_{zeta}.pkl', 'wb') as f:
        pickle.dump(G, f)
    #### generate new G

    with open(save_path+f'data/{L}_avedeg{avg_degree}_{func}_{zeta}.pkl', 'rb') as f:
        G = pickle.load(f)
    nodes = len(list(G.nodes))


    for ti in range(runs):
        np.random.seed(ti)
        ini_p = ini_pp
        ini_nodes = np.random.choice(nodes,int(np.around(nodes*ini_p)),replace=False)
        pc=0 # only plot percolation process once at pc
        while ini_p>=end_p-step_p/2:
            save_node = ini_nodes 
            new_G = G.subgraph(save_node).copy()
            new_G_plot = G.subgraph(save_node).copy()
            sizes = giant_component_size_per_step(new_G, k)
            norm_size = [i/nodes for i in sizes]
            np.save(save_path+f'data/{L}_avedeg{avg_degree}_{zeta}_{k}core_p{np.round(ini_p,3)}_sd{ti}_gcs_over_time.npy',norm_size)

            update_nodes = np.random.choice(save_node,int(np.around(len(save_node)* (1-step_p))),replace=False)
            # delete update_nodes from ini_nodes
            ini_nodes = update_nodes

            if norm_size[-1] == 0 and pc == 0:
                pc=1
                # plot_nodes_in_percolation_pc(list(G.nodes),new_G_plot,positions,save_node,L,avg_degree,zeta,k,ini_p,ti) # plot percolation process
            ini_p-=step_p
          
        print(f'zeta{zeta},{k}core,p{ini_p},seed{ti}')
    print(f'percolation done - cost time: {(time.time()-st)/60} min')


njobs=20
Parallel(n_jobs=njobs)( delayed(main) (lt,func,ini_pp,end_p,step_p) for (lt) in zetalist)