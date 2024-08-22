
import glob
import gc
import signal
import time
import threading
import logging
import os as os
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla
from src.utils import *
from src.algorithm import *


def benchmark_cholesky_solve(matrix, b, times):
    """Solve Ax = b using CHOLMOD Cholesky decomposition and return the time taken."""
    matrix = csc_matrix(matrix)
    def cholesky_solve():
        factor = cholesky(matrix, ordering_method='natural')
        result=factor(b)
        return result
    elapsed_time = timeit.timeit(cholesky_solve, number=times)
    return elapsed_time



def test_reordering(graph, vec1, vec2, node_ordering, repeats):
    reordered_matrix = nx.to_scipy_sparse_array(graph, nodelist=node_ordering)
    bandwidth = compute_bandwidth(reordered_matrix)
    profile = compute_profile(reordered_matrix)
    reordered_matrix = reordered_matrix.tocsr()

    time_list_1 = []
    time_list_2 = []

    for _ in range(5):
        time_1 = benchmark_cholesky_solve(reordered_matrix, vec1, repeats)
        time_list_1.append(time_1)

        time_2 = benchmark_cholesky_solve(reordered_matrix, vec2, repeats)
        time_list_2.append(time_2)

    return bandwidth, profile, time_list_1, time_list_2, reordered_matrix


def process_sparse_matrix(matrix, experiment_index):
    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        if not isinstance(matrix, coo_matrix):
            matrix = matrix.tocoo()

        graph = nx.Graph()
        graph.add_weighted_edges_from(zip(matrix.row, matrix.col, matrix.data))

        random_nodes = random_node_from_components(graph)
        vec_all_ones = np.ones(matrix.shape[1])
        vec_random = np.random.rand(matrix.shape[1])

        rcm1 = list(GL_RCM(graph, random_nodes))
        rcm2 = list(BNF_RCM(graph, random_nodes))
        rcm3 = list(MIND_RCM(graph, random_nodes))

        if are_vectors_equal(rcm1, rcm2):
            attempt += 1
            continue

        repeats = 50

        results_gl = test_reordering(graph, vec_all_ones, vec_random, rcm1, repeats)
        results_mgl = test_reordering(graph, vec_all_ones, vec_random, rcm2, repeats)
        results_md = test_reordering(graph, vec_all_ones, vec_random, rcm3, repeats)

        if not are_matrices_equal(results_gl[4], results_mgl[4]):
            break

        attempt += 1

    experiment_data = {
        "Experiment Index": experiment_index,
        "GL_RCM Bandwidth": results_gl[0],
        "GL_RCM Profile": results_gl[1],
        "Avg GL Time (All Ones)": sum(results_gl[2]) / len(results_gl[2]),
        "Avg GL Time (Random Floats)": sum(results_gl[3]) / len(results_gl[3]),
        "RCM++ Bandwidth": results_mgl[0],
        "RCM++ Profile": results_mgl[1],
        "Avg RCM++ Time (All Ones)": sum(results_mgl[2]) / len(results_mgl[2]),
        "Avg RCM++ Time (Random Floats)": sum(results_mgl[3]) / len(results_mgl[3]),
        "Matrices Equal?": are_matrices_equal(results_gl[4], results_mgl[4]),
        "Min MIND_RCM Bandwidth": results_md[0],
        "Min MIND_RCM Profile": results_md[1],
        "Avg Min Degree Time (All Ones)": sum(results_md[2]) / len(results_md[2]),
        "Avg Min Degree Time (Random Floats)": sum(results_md[3]) / len(results_md[3]),
    }

    return experiment_data
mtx_files = glob.glob(r"/content/drive/MyDrive/BNF/data/Matrixs/small/*", recursive=True)
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(filename='experiment_errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

sorted_files = sorted(mtx_files, key=os.path.getsize)
total_matrices = len(sorted_files)

for idx, mtx_file in enumerate(sorted_files, start=1):
    matrix_name = os.path.basename(mtx_file)
    csv_filename = os.path.join(output_dir, f"{matrix_name}.csv")

    if os.path.exists(csv_filename):
        print(f"CSV file for matrix '{matrix_name}' already exists. Skipping...")
        continue

    print(f"[{idx}/{total_matrices}] Processing matrix '{matrix_name}'...")

    try:
       
        matrix = mmread(mtx_file)

       
        results = []
        for i in range(5):
            result = process_sparse_matrix(matrix, i + 1)
            results.append(result)
            print(f"Processing {matrix_name}, iteration {i + 1}")

        
        pd.DataFrame(results).to_csv(csv_filename, index=False)
        print(f"Processed matrix '{matrix_name}' and saved results to '{csv_filename}'")

    except Exception as e:
        logging.error(f"Error processing matrix '{matrix_name}': {str(e)}")
