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


# Timeout decorator
class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function '{func.__name__}' exceeded {seconds} seconds of execution time.")]

            def worker():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise result[0]
            if isinstance(result[0], BaseException):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# Reordering and testing function for combined algorithms
def test_combined_algorithm(graph, random_nodes, algorithm):
    result_list = []
    if algorithm == "GL_RCM":
        result_list.append(list(GL_RCM(graph, random_nodes)))
    elif algorithm == "BNF_RCM":
        result_list.append(list(BNF_RCM(graph, random_nodes)))
    elif algorithm == "MIND_RCM":
        result_list.append(list(MIND_RCM(graph, random_nodes)))

    rcm_order = result_list[0]
    reordered_matrix = nx.to_scipy_sparse_array(graph, nodelist=rcm_order)
    bandwidth = compute_bandwidth(reordered_matrix)
    profile = compute_profile(reordered_matrix)
    return bandwidth, profile

# Process matrix and compute bandwidth and profile for different combined algorithms
def process_matrix(matrix, matrix_name):
    if not isinstance(matrix, coo_matrix):
        matrix = matrix.tocoo()

    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(matrix.row, matrix.col, matrix.data))
    
    random_nodes = random_node_from_components(graph)

    # Test GL_RCM, BNF_RCM, and MIND_RCM algorithms
    gl_rcm_bandwidth, gl_rcm_profile = test_combined_algorithm(graph, random_nodes, "GL_RCM")
    bnf_rcm_bandwidth, bnf_rcm_profile = test_combined_algorithm(graph, random_nodes, "BNF_RCM")
    mind_rcm_bandwidth, mind_rcm_profile = test_combined_algorithm(graph, random_nodes, "MIND_RCM")

    # Calculate expansion ratios
    data = {
        "Matrix Name": matrix_name,
        "GL_RCM_BW": gl_rcm_bandwidth,
        "BNF_RCM_BW": bnf_rcm_bandwidth,
        "MIND_RCM_BW": mind_rcm_bandwidth,
        "GL_RCM_BNF_RCM_BW_EXP": gl_rcm_bandwidth / bnf_rcm_bandwidth - 1,
        "MIND_RCM_GL_RCM_BW_EXP": gl_rcm_bandwidth / mind_rcm_bandwidth - 1,
        "GL_RCM_Profile": gl_rcm_profile,
        "BNF_RCM_Profile": bnf_rcm_profile,
        "MIND_RCM_Profile": mind_rcm_profile,
        "GL_RCM_BNF_RCM_Profile_EXP": gl_rcm_profile / bnf_rcm_profile - 1,
        "MIND_RCM_GL_RCM_Profile_EXP": gl_rcm_profile / mind_rcm_profile - 1
    }
    return data

# Initialize logging and file paths
num_experiments = 4
error_log_file = "errors.log"
csv_info_file = '/content/drive/MyDrive/BNF/data/Matrix/Matrix Information.csv'
folder_path = r'/content/drive/MyDrive/BNF/data/Matrix'
all_files = [f for f in glob.glob(os.path.join(folder_path, "**", "*.mtx"), recursive=True) if os.path.isfile(f)]
sorted_files = sorted(all_files, key=os.path.getsize)

matrix_info_df = pd.read_csv(csv_info_file)

def find_complete_matrix_name(matrix_name, matrix_info_df):
    while matrix_name:
        if matrix_name in matrix_info_df.iloc[:, 0].values:  
            return matrix_name
        matrix_name = '_'.join(matrix_name.split('_')[:-1])
    return None

def load_processed_file_paths(file_path):
    if os.path.exists(file_path):
        processed_matrices_df = pd.read_csv(file_path)
        return processed_matrices_df['File Path'].tolist()
    return []

# Experiment loop
for experiment_num in range(num_experiments):
    results_file = f"results_experiment_{experiment_num + 1}.csv"
    processed_matrices_file = f"processed_matrices_experiment_{experiment_num + 1}.csv"

    # Initialize results CSV
    df_empty = pd.DataFrame(columns=[
        "Matrix Name", "GL_RCM_BW", "BNF_RCM_BW", "MIND_RCM_BW", 
        "GL_RCM_BNF_RCM_BW_EXP", "MIND_RCM_GL_RCM_BW_EXP", 
        "GL_RCM_Profile", "BNF_RCM_Profile", "MIND_RCM_Profile", 
        "GL_RCM_BNF_RCM_Profile_EXP", "MIND_RCM_GL_RCM_Profile_EXP"
    ])
    df_empty.to_csv(results_file, index=False)

    selected_files = load_processed_file_paths(f"processed_matrices_experiment_{experiment_num}.csv") if experiment_num > 0 else sorted_files

    current_experiment_success_files = set()
    processed_matrix_count = 0

    for file_path in selected_files:
        matrix_name = os.path.splitext(os.path.basename(file_path))[0]
        complete_matrix_name = find_complete_matrix_name(matrix_name, matrix_info_df)

        if complete_matrix_name and complete_matrix_name not in [name for name, _ in current_experiment_success_files]:
            try:
                print(complete_matrix_name)
                with open(file_path, 'r') as f:
                    matrix = mmread(f)
                    data = process_matrix(matrix, complete_matrix_name)
                    df = pd.DataFrame([data])
                    df.to_csv(results_file, mode='a', header=False, index=False)
                    del matrix
                    gc.collect()

                # Mark success
                current_experiment_success_files.add((complete_matrix_name, file_path))

            except TimeoutError:
                print(f"Processing {file_path} took too long. Skipping...")
                with open(error_log_file, "a") as log:
                    log.write(f"Timeout for file: {file_path}\n")
            except Exception as e:
                with open(error_log_file, "a") as log:
                    log.write(f"Error processing file: {file_path}\n")
                    log.write(str(e) + "\n\n")
        else:
            print(f"{complete_matrix_name if complete_matrix_name else matrix_name} not found in CSV or already processed. Skipping...")

        processed_matrix_count += 1
        print(f"Experiment {experiment_num + 1}: Processed {processed_matrix_count}/{len(selected_files)} matrices.")

    # Save processed file paths
    if current_experiment_success_files:
        df_success = pd.DataFrame(list(current_experiment_success_files), columns=["Matrix Name", "File Path"])
        df_success.to_csv(processed_matrices_file, index=False)
