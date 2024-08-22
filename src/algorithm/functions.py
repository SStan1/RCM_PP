#this version can be tested for same starting node of pseudo_peripheral_node algorithm


import os
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla
from src.utils import *



#GL-RCM
def cuthill_mckee_ordering_1(G, random_nodes, heuristic=None):
    for c, start_node in zip(nx.connected_components(G), random_nodes):
        yield from connected_cuthill_mckee_ordering_1(G.subgraph(c), start_node, heuristic)


def connected_cuthill_mckee_ordering_1(G, start_node, heuristic=None):
    # the cuthill mckee algorithm for connected graphs
    
    start = heuristic(G, start_node)
    visited = {start}
    queue = deque([start])
    while queue:
        parent = queue.popleft()
        yield parent
        nd = sorted(G.degree(set(G[parent]) - visited), key=itemgetter(1))
        children = [n for n, d in nd]
        visited.update(children)
        queue.extend(children)

def GL(G, start_node):

    u = start_node
    lp = 0
    v = u
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        if l <= lp:
            break
        lp = l
        farthest = (n for n, dist in spl.items() if dist == l)
        v, deg = min(G.degree(farthest), key=itemgetter(1))

    return v


def BNF(G, start_node):
    # helper for cuthill-mckee to find a node in a "pseudo peripheral pair"
    # to use as good starting node
    u = start_node
    lp = 0
    v = u
    width=float('inf')
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        w=max_distance_occurrence(spl)
        if w<=width:
            width=w
            a=v

        if l <= lp:
            break
        lp = l
        farthest = (n for n, dist in spl.items() if dist == l)
        v, deg = min(G.degree(farthest), key=itemgetter(1))
    return a

def GL_RCM(G, random_nodes, heuristic=GL):
    
    return reversed(list(cuthill_mckee_ordering_1(G, random_nodes, heuristic=heuristic)))




def BNF_RCM(G, random_nodes, heuristic=BNF):
    return reversed(list(cuthill_mckee_ordering_1(G, random_nodes, heuristic=heuristic)))



  

def connected_cuthill_mckee_ordering(G, start_node, heuristic=None):
    # the cuthill mckee algorithm for connected graphs
    start = heuristic(G)
    visited = {start}
    queue = deque([start])
    while queue:
        parent = queue.popleft()
        yield parent
        nd = sorted(G.degree(set(G[parent]) - visited), key=itemgetter(1))
        children = [n for n, d in nd]
        visited.update(children)
        queue.extend(children)


def cuthill_mckee_ordering(G, random_nodes, heuristic=None):
    for c, start_node in zip(nx.connected_components(G), random_nodes):
        yield from connected_cuthill_mckee_ordering(G.subgraph(c), start_node, heuristic)

def MIND(G):
        return min(G, key=G.degree)

def MIND_RCM(G, random_nodes, heuristic=MIND):
    return reversed(list(cuthill_mckee_ordering(G, random_nodes, heuristic=heuristic)))



