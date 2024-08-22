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


#GL
# Redefine GL to handle disconnected graphs
def GL(G, random_nodes, heuristic=None):
    """Find pseudo-peripheral nodes for each connected component."""
    for component, start_node in zip(nx.connected_components(G), random_nodes):
        yield from connected_pseudo_peripheral_1(G.subgraph(component), start_node, heuristic)

def connected_pseudo_peripheral_1(G, start_node, heuristic=None):
    return pseudo_peripheral_node_1(G, start_node)

def pseudo_peripheral_node_1(G, start_node):
    """Find pseudo-peripheral node as a starting point for GL algorithm."""
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

# Redefine BNF (RCM++) to handle disconnected graphs
def BNF(G, random_nodes, heuristic=None):
    """Find pseudo-peripheral nodes with minimal width for each connected component."""
    for component, start_node in zip(nx.connected_components(G), random_nodes):
        yield from connected_pseudo_peripheral_2(G.subgraph(component), start_node, heuristic)

def connected_pseudo_peripheral_2(G, start_node, heuristic=None):
    return pseudo_peripheral_node_2(G, start_node)

def pseudo_peripheral_node_2(G, start_node):
    """Find pseudo-peripheral node with minimal width as a starting point for RCM++."""
    u = start_node
    lp = 0
    v = u
    width = float('inf')
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        w = max_distance_occurrence(spl)
        if w <= width:
            width = w
            a = v
        if l <= lp:
            break
        lp = l
        farthest = (n for n, dist in spl.items() if dist == l)
        v, deg = min(G.degree(farthest), key=itemgetter(1))
    return a

# Redefine MIND to handle disconnected graphs
def MIND(G, random_nodes, heuristic=None):
    """Select the node with the smallest degree for each connected component."""
    for component, start_node in zip(nx.connected_components(G), random_nodes):
        yield smallest_degree(G.subgraph(component))

class TimeoutError(Exception):
    pass

def timeout(seconds):
    """Decorator to set a timeout for a function."""
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
                thread.join()  # Optionally, you can leave the thread running without joining it
                raise result[0]
            if isinstance(result[0], BaseException):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# Test GL algorithm's execution time
def test_gl_rcm(graph, random_nodes):
    """Measure the execution time of the GL algorithm."""
    wrapper = lambda: GL(graph, random_nodes)
    gl_time = timeit.timeit(wrapper, number=100)
    return gl_time

# Test BNF (RCM++) algorithm's execution time
def test_rcm_plus_plus(graph, random_nodes):
    """Measure the execution time of the RCM++ algorithm (BNF)."""
    wrapper = lambda: BNF(graph, random_nodes)
    rcm_plus_plus_time = timeit.timeit(wrapper, number=100)
    return rcm_plus_plus_time

# Test MIND algorithm's execution time
def test_mind_rcm(graph, random_nodes):
    """Measure the execution time of the MIND algorithm."""
    wrapper = lambda: MIND(graph, random_nodes)
    mind_time = timeit.timeit(wrapper, number=100)
    return mind_time

@timeout(900)
def process_matrix(matrix, matrix_name_without_extension):
    """Process the given matrix and record the execution time of each algorithm."""
    if not isinstance(matrix, coo_matrix):
        matrix = matrix.tocoo()

    graph = nx.Graph()
    graph.add_weighted_edges_from(zip(matrix.row, matrix.col, matrix.data))

    random_nodes = random_node_from_components(graph)

    # **1. Why run GL first as warmup?**
    # Since these algorithms execute very quickly, we run GL first as a warmup to stabilize the environment 
    # for more accurate time measurements.
    warmup_time = test_gl_rcm(graph, random_nodes=[0])

    # Measure execution times for GL, RCM++ (BNF), and MIND algorithms
    gl_time = test_gl_rcm(graph, random_nodes=[0])
    rcm_plus_plus_time = test_rcm_plus_plus(graph, random_nodes=[0])
    mind_time = test_mind_rcm(graph, random_nodes=[0])

    # Record and return results
    new_data = {
        "Matrix Name": matrix_name_without_extension,
        "GL_time": gl_time,
        "RCM++_time": rcm_plus_plus_time,
        "MIND_time": mind_time,
        "Warmup_time": warmup_time
    }

    return new_data

# Initialization
num_experiments = 10
error_log_file = "errors.log"
csv_info_file = '/content/drive/MyDrive/BNF/data/Matrix/Matrix Information.csv'
folder_path = r'/content/drive/MyDrive/BNF/data/Matrix'
all_files = [f for f in glob.glob(os.path.join(folder_path, "**", "*.mtx"), recursive=True) if os.path.isfile(f)]
sorted_files = sorted(all_files, key=os.path.getsize)

# Select all files
eighty_five_percent_count = int(1 * len(sorted_files))
files_for_first_experiment = sorted_files[:eighty_five_percent_count]

# Read CSV information file
matrix_info_df = pd.read_csv(csv_info_file)

def find_complete_matrix_name(matrix_name, matrix_info_df):
    """Find the complete matrix name based on a partial matrix name."""
    while matrix_name:
        if matrix_name in matrix_info_df.iloc[:, 0].values:
            return matrix_name
        matrix_name = '_'.join(matrix_name.split('_')[:-1])
    return None

def load_processed_file_paths(file_path):
    """Load the file paths of matrices that have already been processed."""
    if os.path.exists(file_path):
        processed_matrices_df = pd.read_csv(file_path)
        return processed_matrices_df['File Path'].tolist()
    return []

# Experiment loop
for experiment_num in range(num_experiments):
    results_file = f"results_experiment_samestart_{experiment_num + 1}.csv"
    processed_matrices_file = f"processed_matrices_experiment_{experiment_num + 1}.csv"

    # Initialize CSV file and write header
    df_empty = pd.DataFrame(columns=["Matrix Name", "GL_time", "RCM++_time", "MIND_time", "Warmup_time"])
    df_empty.to_csv(results_file, index=False)

    if experiment_num == 0:
        selected_files = files_for_first_experiment
    else:
        selected_files = load_processed_file_paths(f"processed_matrices_experiment_{experiment_num}.csv")

    # Track successfully processed files
    current_experiment_success_files = set()
    processed_matrix_count = 0

    for file_path in selected_files:
        matrix_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        complete_matrix_name = find_complete_matrix_name(matrix_name_without_extension, matrix_info_df)

        # Check if matrix name is in Matrix_information.csv and hasn't been processed yet
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

                # If no errors, add file to successfully processed list
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
            print(f"{complete_matrix_name if complete_matrix_name else matrix_name_without_extension} not found in CSV or already processed in this experiment. Skipping...")

        processed_matrix_count += 1
        print(f"EXP{experiment_num + 1} Processed {processed_matrix_count}/{len(selected_files)} matrices.")

    # Save the list of successfully processed files for this experiment
    if current_experiment_success_files:
        df_success = pd.DataFrame(list(current_experiment_success_files), columns=["Matrix Name", "File Path"])
        df_success.to_csv(processed_matrices_file, index=False)