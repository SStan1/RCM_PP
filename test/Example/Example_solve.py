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

#TEST FUNCTION
# Test function for GL+RCM algorithm
def test_gl_rcm(graph, vector, times):
    """Test GL+RCM algorithm, calculate bandwidth, profile, and solve time."""
    # Perform RCM reordering using GL algorithm
    rcm = list(GL_RCM(graph, [0]))
    reordered_matrix = nx.to_scipy_sparse_array(graph, nodelist=rcm)
    
    # Calculate bandwidth and profile
    bandwidth = compute_bandwidth(reordered_matrix)
    profile = compute_profile(reordered_matrix)
    
    # Convert to CSR format and solve
    reordered_matrix = reordered_matrix.tocsr()
    solve_time = benchmark_solve(reordered_matrix, vector, times)

    return bandwidth, profile, solve_time, reordered_matrix

# Test function for BNF+RCM (DWPS) algorithm
def test_bnf_rcm(graph, vector, times):
    """Test BNF+RCM (DWPS) algorithm, calculate bandwidth, profile, and solve time."""
    # Perform RCM reordering using BNF algorithm
    rcm = list(BNF_RCM(graph, [0]))
    reordered_matrix = nx.to_scipy_sparse_array(graph, nodelist=rcm)
    
    # Calculate bandwidth and profile
    bandwidth = compute_bandwidth(reordered_matrix)
    profile = compute_profile(reordered_matrix)
    
    # Convert to CSR format and solve
    reordered_matrix = reordered_matrix.tocsr()
    solve_time = benchmark_solve(reordered_matrix, vector, times)

    return bandwidth, profile, solve_time, reordered_matrix

# Process a single matrix and return results for GL and DWPS algorithms
def process_matrix(matrix, matrix_name):
    """Process the matrix and compute results for GL and DWPS algorithms."""
    if not isinstance(matrix, coo_matrix):
        matrix = matrix.tocoo()

    # Create a graph from the matrix
    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(matrix.row, matrix.col, matrix.data))
    
    # Generate a random vector
    vector = np.random.rand(matrix.shape[1])
    times = 100

    # Test GL+RCM and DWPS algorithms
    gl_bw, gl_profile, gl_solve_time, _ = test_gl_rcm(graph, vector, times)
    dwps_bw, dwps_profile, dwps_solve_time, _ = test_bnf_rcm(graph, vector, times)

    # Calculate accuracy comparison between GL and DWPS
    accuracy = (gl_solve_time - dwps_solve_time) / gl_solve_time * 100

    # Return the results in a dictionary
    return {
        "Matrix_Name": matrix_name,
        "GL_bw": gl_bw,
        "GL_profile": gl_profile,
        "DWPS_bw": dwps_bw,
        "DWPS_profile": dwps_profile,
        "ACC(%)": accuracy
    }

# Setup logging and output directories
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "results")
os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(current_dir, 'experiment_errors.log')
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Read matrix files from the specified directory
mtx_files = glob.glob(r"/content/drive/MyDrive/BNF/test/Matrix/*", recursive=True)
sorted_files = sorted(mtx_files, key=os.path.getsize)
total_matrices = len(sorted_files)

# Initialize list to store all results
all_results = []

# Process each matrix file
for processed_count, mtx_file in enumerate(sorted_files, start=1):
    matrix_name = os.path.basename(mtx_file)
    print(f"[{processed_count}/{total_matrices}] Processing matrix '{matrix_name}'...")

    try:
        # Read and process the matrix
        matrix = mmread(mtx_file)
        result = process_matrix(matrix, matrix_name)
        all_results.append(result)
        print(f"Processed matrix '{matrix_name}'")
    except Exception as e:
        # Log errors and continue with the next file
        logging.error(f"Error processing matrix '{matrix_name}': {str(e)}")
        continue

# Write all results to a CSV file
df = pd.DataFrame(all_results)
csv_filename = os.path.join(output_dir, "all_results.csv")
df.to_csv(csv_filename, index=False)

print(f"Processed all matrices and saved results to '{csv_filename}'")