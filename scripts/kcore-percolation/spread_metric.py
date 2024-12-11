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
from matplotlib.colors import LinearSegmentedColormap

st = time.time()

# choose designed models
func = 'powl-exp'
# func = 'powl' 

alpha=1
zetalist = [1,0.1,0] # choose parameters

save_path = f'/../{func}/'   # save path

runs=20 # independent runs

L = 100  # Lattice size
avg_degree = 20  # Average degree

ini_pp=0.9
end_p=0.6
step_p=0.001

def giant_component_size_per_step(G: nx.Graph, k: int) -> List[int]:
    # paralell as the procalation figure
    def nodes_to_remove(G, k):
        """Determine nodes with in-degree and out-degree less than k."""
        return [node for node in G.nodes() if G.degree(node) < k]
        # return [node for node in G.nodes() if G.in_degree(node) + G.out_degree(node) < k]
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

def giant_component_size_dis_dev_pc(all_nodes,G, nxx, ny, max_var, k, zeta, p, ti):

    def nodes_to_remove(G, k):
        """Determine nodes with in-degree and out-degree less than k."""
        return [node for node in G.nodes() if G.degree(node)< k]
    sizes = []
    to_remove = nodes_to_remove(G, k)
    
    # If no nodes need to be removed initially, add the size of the giant component of the initial graph
    if not to_remove and G.number_of_nodes() > 0:
        giant_component = max(nx.connected_components(G), key=len)
        sizes.append(len(giant_component))
    if not to_remove and G.number_of_nodes() == 0:
        sizes.append(0)
    
    T=0
    dis_dev=[]

    delete_nodes = [i for i in list(G.nodes())] 
    sum_x = [nxx[i] for i in delete_nodes]
    sum_y = [ny[i] for i in delete_nodes]
    variance_x = np.var(sum_x)
    variance_y = np.var(sum_y)
    dis_dev.append(np.sqrt((variance_x+variance_y)/max_var))


    while to_remove:
        G.remove_nodes_from(to_remove)
        if G.number_of_nodes() == 0:
            sizes.append(0)
            break
    
        giant_component = max(nx.connected_components(G), key=len)
        sizes.append(len(giant_component))
        to_remove = nodes_to_remove(G, k)

        gcc =[]
        for node in giant_component:
            gcc.append(node)

        delete_nodes = [i for i in all_nodes if i not in gcc]

        grey_node = [i for i in all_nodes if i not in delete_nodes]

        sum_x = [nxx[i] for i in grey_node]
        sum_y = [ny[i] for i in grey_node]
        variance_x = np.var(sum_x)
        variance_y = np.var(sum_y)
        dis_dev.append(np.sqrt((variance_x+variance_y)/max_var))

        T+=1
        print(T)
        
    os.makedirs(save_path+f'dis_dev', exist_ok=True)

    np.save(save_path+f'dis_dev/{L}_avedeg{avg_degree}_{zeta}_{k}core_sd{ti}_dis_dev.npy',dis_dev)


def main(zeta, func, ini_p, end_p, step_p):
    # Parameters
    zeta = zeta  # Parameter for the function distribution
    k = avg_degree//2 # k-core, suggest average / 2
    
    N = L ** 2  # network nodes
    total_links = N * avg_degree // 2  # network links
    
    # Create network positions
    positions = {i: (i % L, i // L) for i in range(N)}

    nxx = [x for x, y in positions.values()]
    nyy = [y for x, y in positions.values()]
    max_var = np.var(nxx)+np.var(nyy)

    with open(save_path+f'data/{L}_avedeg{avg_degree}_{func}_{zeta}.pkl', 'rb') as f:
        G = pickle.load(f)
    nodes = len(list(G.nodes))

    for ti in range(runs): # take the averaged value
        ini_p = ini_pp
        np.random.seed(ti)
        ini_nodes = np.random.choice(nodes,int(np.around(nodes*ini_p)),replace=False)
        pc=0 # only plot percolation process once at pc
        while ini_p>=end_p-step_p/2:
            save_node = ini_nodes 
            new_G = G.subgraph(save_node).copy()
            new_G_plot = G.subgraph(save_node).copy()
            sizes = giant_component_size_per_step(new_G, k)
            norm_size = [i/nodes for i in sizes]

            update_nodes = np.random.choice(save_node,int(np.around(len(save_node)* (1-step_p))),replace=False)
            #delete update_nodes from ini_nodes
            ini_nodes = update_nodes

            if norm_size[-1] == 0 and pc == 0:
                pc=1
                giant_component_size_dis_dev_pc(list(G.nodes),new_G_plot, nxx, nyy, max_var, k, zeta, ini_p, ti)
                break
            ini_p-=step_p

        print(f'zeta{zeta},{k}core,p{ini_p},seed{ti}')
    print(f'dis dev done - cost time: {(time.time()-st)/60} min')


njobs=20
for lt in zetalist: 
    print(lt)
    main(lt,func,ini_pp,end_p,step_p)
