__all__ = [
    'smallest_degree', 
    'is_graph_connected', 
    'are_vectors_equal', 
    'compute_profile', 
    'compute_bandwidth', 
    'max_distance_occurrence', 
    'are_matrices_equal', 
    'benchmark_solve', 
    'random_node_from_components'
]

import networkx as nx
import numpy as np
import timeit
from operator import itemgetter
from collections import Counter
from scipy.io import mmread
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import factorized
from scipy.linalg import solve_triangular
from sksparse.cholmod import cholesky
from  networkx.utils import arbitrary_element
from collections import deque
import random



def smallest_degree(G):
    return min(G, key=G.degree)



def is_graph_connected(G):
    return nx.is_connected(G)



def are_vectors_equal(vec1, vec2):
    return np.array_equal(vec1, vec2)


def compute_profile(matrix):
    # Ensure the matrix is in COO format
    coo_mat = matrix.tocoo()

    # Sort the COO matrix entries by row, then by column
    sorted_indices = np.lexsort((coo_mat.col, coo_mat.row))
    sorted_rows = coo_mat.row[sorted_indices]
    sorted_cols = coo_mat.col[sorted_indices]

    # Identify unique rows and their first occurrences
    unique_rows, first_occurrences = np.unique(sorted_rows, return_index=True)

    # Compute the distances for unique rows
    distances = abs(unique_rows - sorted_cols[first_occurrences])

    # Create an array to store the distances for all rows
    all_distances = np.zeros(matrix.shape[0], dtype=np.int64)
    all_distances[unique_rows] = distances


    # Convert the array to a Python list
    all_distances_list = list(all_distances)

    profile = sum(all_distances_list)

    return profile


def compute_bandwidth(matrix):
    # Ensure the matrix is in COO format
    coo_mat = matrix.tocoo()

    # Compute the bandwidth based on row and column indices
    bandwidth = max(abs(coo_mat.row - coo_mat.col))

    return bandwidth


def max_distance_occurrence(spl):
    """Return the maximum number of occurrences of any distance."""
    distance_counts = Counter(spl.values())
    return max(distance_counts.values())


def are_matrices_equal(mat1, mat2):
    """
    Returns True if the two CSR matrices are equal, False otherwise.
    """
    return (np.array_equal(mat1.data, mat2.data) and 
            np.array_equal(mat1.indices, mat2.indices) and 
            np.array_equal(mat1.indptr, mat2.indptr))


def benchmark_solve(matrix, b, times):
    """Solve Ax = b using CHOLMOD Cholesky decomposition and return the time taken."""
    matrix = csc_matrix(matrix)
    def cholesky_solve():
        factor = cholesky(matrix, ordering_method='natural')
        result=factor(b)
        return result
    elapsed_time = timeit.timeit(cholesky_solve, number=times)
    return elapsed_time




def random_node_from_components(G):

    components = [c for c in nx.connected_components(G)]
    random_nodes = [random.choice(list(component)) for component in components]
    return random_nodes