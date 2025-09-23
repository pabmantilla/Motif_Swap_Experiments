"""
This script is used to sweep the parameters and compare each parameter
set to the reference background, foreground, and cluster-specific background.
The referece attribition maps are made with a library size of 100K, 50
clusters,and hierarchical clustering.

Swwps Parameters:

Library size: [100, 500, 1000, 5000, 10000, 50000, 75000, 100000]

Number of clusters: [10, 20, 30, 40, 50]

Clustering method: [kmans, tsne+kmeans, UMAP+kmeans]

Saves to .csv and creates summary plots.
"""
import os, sys

# Allow GPU usage for most operations, but k-means will be forced to CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Commented out to allow GPU

import time
import random
import numpy as np
import tensorflow as tf
from itertools import product

tf.get_logger().setLevel('ERROR')
# Suppress TensorRT warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Everything will run on CPU - slower but no GPU memory issues
import squid

from urllib.request import urlretrieve
import pandas as pd
import gc
import pyarrow as pa
import pyarrow.feather as feather
import h5py
from keras.models import model_from_json
from seam.logomaker_batch.batch_logo import BatchLogo
import matplotlib.pyplot as plt

# Optional cupy import for GPU memory management
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available for GPU memory management")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - GPU memory management will be limited")
    print("For better GPU memory management, install cupy:")
    print("pip install cupy-cuda11x  # replace with your CUDA version")

# Global persistent model and data storage
PERSISTENT_MODEL = None
PERSISTENT_SEAM_MODEL = None
PERSISTENT_X_DATASET = None
PERSISTENT_ASSETS_DIR = None
PERSISTENT_ALPHABET = ['A','C','G','T']

# Global processing parameters
PERSISTENT_MUT_RATE = 0.1  # mutation rate for in silico mutagenesis
PERSISTENT_NUM_SEQS = 100000  # number of sequences to generate for reference (used in memory pool initialization)
PERSISTENT_NUM_SEQS_SWEEP = [100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000]  # number of sequences to generate (5K for debugging)
PERSISTENT_N_CLUSTERS_SWEEP = [10, 20, 30, 40, 50]  # number of clusters for hierarchical clustering
PERSISTENT_CLUSTERING_METHOD_SWEEP = ['kmeans']  # clustering method
PERSISTENT_N_PCA_COMPONENTS = 10  # number of PCA components for dimensionality reduction
PERSISTENT_SEQ_LEN = 249  # hardcoded sequence length for DeepSTARR (all sequences are 249bp)
PERSISTENT_ATTRIBUTION_METHOD = 'intgrad'  # {saliency, smoothgrad, intgrad, ism}
PERSISTENT_ADAPTIVE_BACKGROUND_SCALING = True  # Whether to use cluster-specific background scaling

# Initialize parameter sweep index
parameter_sweep_set = product(PERSISTENT_CLUSTERING_METHOD_SWEEP, PERSISTENT_NUM_SEQS_SWEEP, PERSISTENT_N_CLUSTERS_SWEEP)
parameter_sweep_set = pd.DataFrame(parameter_sweep_set, columns=['clustering_method', 'num_seqs', 'n_clusters'])
parameter_sweep_index = 0

# Global GPU detection - allow GPU for most operations
PERSISTENT_GPU = True  # Allow GPU usage for most operations

# Store the GPU selection from environment
SELECTED_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
print("="*60)
print("RUNNING IN GPU MODE")
print("="*60)
print("GPU enabled for most operations.")
print("K-means clustering will be forced to CPU for stability.")

# Check GPU availability
try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Check CUDA_VISIBLE_DEVICES
    print(f"CUDA_VISIBLE_DEVICES: {SELECTED_GPU}")
    print(f"Selected GPU will be used throughout the script: {SELECTED_GPU}")
    
    # Check which GPU TensorFlow will use
    if gpus:
        print(f"TensorFlow will use: {gpus[0].name}")
    else:
        print("No GPUs available to TensorFlow")
        
except Exception as e:
    print(f"Error checking GPU availability: {e}")

print("="*60)

class MemoryPool:
    """Memory pool for pre-allocating frequently used arrays to avoid repeated allocations."""
    
    def __init__(self, max_seqs=10000, max_seq_length=249, max_pca_components=20):
        """
        Initialize memory pool with pre-allocated arrays.
        
        Args:
            max_seqs: Maximum number of sequences to support
            max_seq_length: Maximum sequence length to support
            max_pca_components: Maximum number of PCA components to support
        """
        print(f"Initializing memory pool: {max_seqs} sequences × {max_seq_length} positions")
        
        # Pre-allocate arrays for mutagenesis data (optimized data types)
        self.x_mut_pool = np.zeros((max_seqs, max_seq_length, 4), dtype=np.int8)  # One-hot sequences
        self.y_mut_pool = np.zeros((max_seqs, 1), dtype=np.float16)  # Predictions (float16 precision sufficient)
        
        # Pre-allocate arrays for attribution maps (float16 for memory efficiency)
        self.attributions_pool = np.zeros((max_seqs, max_seq_length, 4), dtype=np.float16)
        
        # Pre-allocate array for PCA embedding (float16 for memory efficiency)
        self.pca_embedding_pool = np.zeros((max_seqs, max_pca_components), dtype=np.float16)
        
        # Track available slots
        self.available_slots = list(range(max_seqs))
        self.max_seqs = max_seqs
        self.max_seq_length = max_seq_length
        self.max_pca_components = max_pca_components
        
        print(f"Memory pool initialized with {len(self.available_slots)} available slots")
        print(f"PCA embedding pool: {max_seqs} sequences × {max_pca_components} components")
    
    def get_arrays(self, num_seqs, seq_length):
        """
        Get pre-allocated arrays for the specified number of sequences.
        
        Args:
            num_seqs: Number of sequences needed
            seq_length: Length of each sequence
            
        Returns:
            Tuple of (x_mut, y_mut, attributions, attributions_flat) views
        """
        if num_seqs > len(self.available_slots):
            raise ValueError(f"Requested {num_seqs} sequences but only {len(self.available_slots)} available")
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Requested sequence length {seq_length} exceeds max {self.max_seq_length}")
        
        # Get slots for this request
        slots = self.available_slots[:num_seqs]
        self.available_slots = self.available_slots[num_seqs:]
        
        # Return views into pre-allocated memory
        x_mut = self.x_mut_pool[slots, :seq_length, :]
        y_mut = self.y_mut_pool[slots, :num_seqs]
        attributions = self.attributions_pool[slots, :seq_length, :]
        
        return x_mut, y_mut, attributions, slots
    
    def get_pca_embedding(self, num_seqs, n_components):
        """
        Get pre-allocated PCA embedding array.
        
        Args:
            num_seqs: Number of sequences needed
            n_components: Number of PCA components needed
            
        Returns:
            View into pre-allocated PCA embedding array
        """
        if n_components > self.max_pca_components:
            raise ValueError(f"Requested {n_components} PCA components but only {self.max_pca_components} available")
        
        # Return view into pre-allocated PCA embedding memory
        return self.pca_embedding_pool[:num_seqs, :n_components]
    
    def release_arrays(self, slots):
        """
        Release arrays back to the pool.
        
        Args:
            slots: List of slot indices to release
        """
        # No need to zero arrays - they'll be overwritten on next use
        # Mark slots as available again
        self.available_slots.extend(slots)
        
    def get_pool_status(self):
        """Get current pool status for debugging."""
        return {
            'total_slots': self.max_seqs,
            'available_slots': len(self.available_slots),
            'used_slots': self.max_seqs - len(self.available_slots),
            'max_seq_length': self.max_seq_length
        }
    
    def get_gpu_memory_info(self):
        """Get GPU memory usage information."""
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                return {
                    'used': mempool.used_bytes(),
                    'total': mempool.total_bytes(),
                    'free': mempool.free_bytes(),
                    'pinned_used': pinned_mempool.used_bytes(),
                    'pinned_total': pinned_mempool.total_bytes()
                }
            except Exception as e:
                return {'error': str(e)}
        else:
            return {'error': 'CuPy not available'}

# Global memory pool
PERSISTENT_MEMORY_POOL = None

def initialize_persistent_resources():
    """Initialize persistent resources (model, data) that will be reused across sequences."""
    global PERSISTENT_MODEL, PERSISTENT_SEAM_MODEL, PERSISTENT_X_DATASET, PERSISTENT_ASSETS_DIR, PERSISTENT_MEMORY_POOL
    
    if PERSISTENT_MODEL is not None:
        print("Persistent resources already initialized, skipping...")
        return
    
    print("Initializing persistent resources (model, data, and memory pool)...")
    start_time = time.time()
    
    # Create assets_deepstarr folder if it doesn't exist
    py_dir = os.path.dirname(os.path.abspath(__file__))
    PERSISTENT_ASSETS_DIR = os.path.join(py_dir, 'assets_deepstarr')
    if not os.path.exists(PERSISTENT_ASSETS_DIR):
        os.makedirs(PERSISTENT_ASSETS_DIR)

    def download_if_not_exists(url, filename):
        """Download a file if it doesn't exist locally."""
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urlretrieve(url, filename)
        else:
            print(f"Using existing {filename}")
        
        # Validate file is not empty
        if os.path.getsize(filename) == 0:
            print(f"ERROR: {filename} is empty! Re-downloading...")
            urlretrieve(url, filename)
            if os.path.getsize(filename) == 0:
                raise RuntimeError(f"Failed to download {filename} - file is still empty after retry")
        
        print(f"✓ {filename} validated ({os.path.getsize(filename)} bytes)")

    # Use EvoAug model path (no download needed)
    py_dir = os.path.dirname(os.path.abspath(__file__))
    evoaug_model_path = os.path.join(py_dir, "../../EvoAug_keras_model")

    # Load test sequences from CSV (persistent)
    print("Loading test library from CSV...")
    import pandas as pd
    import ast
    
    test_library_df = pd.read_csv(os.path.join(py_dir, '../test_library_df.csv'))
    
    # Extract one-hot sequences from the CSV
    ref_test_seqs = []
    for i, row in test_library_df.iterrows():
        # Parse the string representation of the numpy array
        seq_str = row['seq_one_hot']
        # Handle newlines and brackets, then parse
        clean_str = seq_str.replace('[', '').replace(']', '').replace('\n', ' ')
        clean_str = ' '.join(clean_str.split())  # Remove multiple spaces
        values = [float(x) for x in clean_str.split() if x.strip()]
        # Reshape to (4, 249) then transpose to (249, 4) to match DeepSTARR format
        seq_one_hot = np.array(values, dtype=np.float32).reshape(4, -1).T
        ref_test_seqs.append(seq_one_hot)
    
    PERSISTENT_X_DATASET = np.array(ref_test_seqs)
    print(f"Loaded {len(PERSISTENT_X_DATASET)} test sequences from CSV")
    print(f"Sequence shape (SEAM format): {PERSISTENT_X_DATASET.shape}")
    
    # Validate hardcoded sequence length matches dataset
    actual_seq_len = PERSISTENT_X_DATASET.shape[1]  # SEAM format: seq length is second dimension
    if actual_seq_len != PERSISTENT_SEQ_LEN:
        raise ValueError(f"Hardcoded sequence length {PERSISTENT_SEQ_LEN} doesn't match dataset sequence length {actual_seq_len}")
    print(f"✓ Sequence length validation passed: {PERSISTENT_SEQ_LEN}bp")

    # Load the EvoAug model (persistent)
    print("Loading EvoAug model...")
    
    # Validate model path
    if not os.path.exists(evoaug_model_path):
        raise RuntimeError(f"EvoAug model not found at: {evoaug_model_path}")
    print(f"✓ EvoAug model path validated: {evoaug_model_path}")
    
   

    # Set random seeds BEFORE loading model (matching original script order)
    np.random.seed(113)
    random.seed(0)
    PERSISTENT_MODEL = tf.keras.models.load_model(evoaug_model_path)
    
    # =============================================================================
    # Create a minimal wrapper for SEAM that handles input transposition
    class SEAMWrapper(tf.keras.Model):
        def __init__(self, original_model):
            super().__init__()
            self.original_model = original_model
        
        def call(self, inputs, training=None):
            # Convert int8 to float32 (model expects float32)
            if inputs.dtype != tf.float32:
                inputs = tf.cast(inputs, tf.float32)
            
            # Always transpose from SEAM format (batch, length, channels) to model format (batch, channels, length)
            if len(inputs.shape) == 3:
                inputs = tf.transpose(inputs, [0, 2, 1])
            
            return self.original_model(inputs, training=False)
    
    # Create the SEAM wrapper for the persistent model
    PERSISTENT_SEAM_MODEL = SEAMWrapper(PERSISTENT_MODEL)
    
    # Initialize memory pool with hardcoded sequence length for maximum efficiency
    print("Initializing memory pool for 100K sequences...")
    PERSISTENT_MEMORY_POOL = MemoryPool(max_seqs=PERSISTENT_NUM_SEQS, max_seq_length=PERSISTENT_SEQ_LEN, max_pca_components=PERSISTENT_N_PCA_COMPONENTS)
    
    init_time = time.time() - start_time 
    # print model output on first seq using SEAM wrapper
    print("Printing SEAM wrapper model output on first sequence...")
    first_seq = PERSISTENT_X_DATASET[0:1]  # Already in SEAM format (batch, length, channels)
    print(PERSISTENT_SEAM_MODEL(first_seq))
    
    print(f"Persistent resources initialized in {init_time:.2f} seconds")
    print("Model, data, and memory pool will be reused for all subsequent sequences")


def run_parameter_sweep(parameter_sweep_index, seq_index=0, task_index=1):
    """Run parameter sweep for a specific parameter combination and sequence"""
    time_start = time.time()
    
    # Get parameter set from index
    parameter_set = parameter_sweep_set.iloc[parameter_sweep_index]
    
    # Initialize persistent resources if not already done
    initialize_persistent_resources()
    
    # Start total timer
    total_start_time = time.time()
    
    from seam import Compiler, Attributer, Clusterer, MetaExplainer

    # In persistent mode, we DON'T clear the session to maintain model state
    # Only run garbage collection to free unused memory
    gc.collect()
    
    # Ensure persistent resources are available
    initialize_persistent_resources()

    # =============================================================================
    # Use global processing parameters
    # =============================================================================
    mut_rate = PERSISTENT_MUT_RATE
    clustering_method = parameter_set['clustering_method']
    num_seqs = parameter_set['num_seqs']
    n_clusters = parameter_set['n_clusters']
    attribution_method = PERSISTENT_ATTRIBUTION_METHOD
    adaptive_background_scaling = PERSISTENT_ADAPTIVE_BACKGROUND_SCALING
    gpu = PERSISTENT_GPU  # Use global GPU detection

    # =============================================================================
    # Set up save paths for parameter sweep results
    # =============================================================================
    py_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_essential = os.path.join(py_dir, f'parameter_set_{num_seqs}_{n_clusters}_{clustering_method}_results')
    if not os.path.exists(save_path_essential):
        os.makedirs(save_path_essential)
        print(f"Created directory: {save_path_essential}")
    else:
        print(f"Using existing directory: {save_path_essential}")
        
        # Check if sequence has already been processed (Arrow file exists)
        arrow_filename = f'seq{seq_index}_task{task_index}.arrow'
        arrow_filepath = os.path.join(save_path_essential, arrow_filename)
        print(f"DEBUG: Checking for existing Arrow file at: {arrow_filepath}")
        if os.path.exists(arrow_filepath):
            print(f"✓ Sequence {seq_index} already processed - Arrow file exists: {arrow_filename}")
            print(f"Skipping sequence {seq_index}")
            return
        else:
            print(f"DEBUG: Arrow file does not exist, proceeding with processing")
        
        # Create sequence-specific folder for PNGs
        seq_folder = os.path.join(save_path_essential, f'seq{seq_index}')
        if not os.path.exists(seq_folder):
            os.makedirs(seq_folder)
            print(f"Created sequence folder: {seq_folder}")
        else:
            print(f"Using existing sequence folder: {seq_folder}")

        # =============================================================================
        # Use persistent model and data
        # =============================================================================
        # Use the persistent model and SEAM wrapper instead of loading them again
        model = PERSISTENT_MODEL
        seam_model = PERSISTENT_SEAM_MODEL
        X_dataset = PERSISTENT_X_DATASET
        alphabet = PERSISTENT_ALPHABET

        x_ref = X_dataset[seq_index]
        x_ref = np.expand_dims(x_ref, 0)  # Already in SEAM format (batch, length, channels)

        # Define mutagenesis window for sequence
        seq_length = x_ref.shape[1]  # SEAM format: sequence length is second dimension
        mut_window = [0, seq_length]  # [start_position, stop_position]
        
        # Forward pass to get output for the specific head using SEAM wrapper
        output = seam_model(x_ref)
        print(f"Output shape: {output.shape}")
        print(f"Output type: {type(output)}")
        pred = output[:, task_index:task_index+1]  # Keep full batch dimension

        print(f"\nWild-type prediction: {pred}")
        print(f"Prediction shape: {pred.shape}")
        print(f"Prediction type: {type(pred)}")

        # =============================================================================
        # SQUID API
        # Create in silico mutagenesis library
        # =============================================================================
        
        # Set up predictor class for in silico MAVE
        pred_generator = squid.predictor.ScalarPredictor(
            pred_fun=lambda x: seam_model.predict_on_batch(x)[:, task_index:task_index+1],
            task_idx=1,  # Use 0 since we extract task in pred_fun
            batch_size=512
        )

        # Set up mutagenizer class for in silico MAVE
        mut_generator = squid.mutagenizer.RandomMutagenesis(
            mut_rate=mut_rate
        )

        # Generate in silico MAVE
        mave = squid.mave.InSilicoMAVE(
            mut_generator,
            pred_generator,
            seq_length,
            mut_window=mut_window
        )
        
        # Get pre-allocated arrays from memory pool
        x_mut, y_mut, attributions_pool, slots = PERSISTENT_MEMORY_POOL.get_arrays(num_seqs, seq_length)
        
        # Generate mutagenesis data into pre-allocated arrays
        x_mut_temp, y_mut_temp = mave.generate(x_ref[0], num_sim=num_seqs)
        
        # Copy data into pre-allocated arrays with optimized data types
        x_mut[:] = x_mut_temp.astype(np.int8)  # SQUID creates int8/uint8, convert to int8 for consistency
        y_mut[:] = y_mut_temp.astype(np.float16)  # Convert to float16 for memory efficiency

        # =============================================================================
        # SEAM API
        # Compile sequence analysis data into a standardized format
        # =============================================================================
        # Initialize compiler
        compiler = Compiler(
            x=x_mut,
            y=y_mut,
            x_ref=x_ref,
            y_bg=None,
            alphabet=alphabet,
            gpu=gpu
        )

        mave_df = compiler.compile()
        ref_index = 0 # index of reference sequence (zero by default)

        # =============================================================================
        # SEAM API
        # Compute attribution maps for each sequence in library
        # =============================================================================
        # Use persistent SEAM wrapper (already created during initialization)
        # =============================================================================
        
        attribution_method = 'intgrad' # {saliency, smoothgrad, intgrad, deepshap, ism}
        print(f"DEBUG: compress_fun input shape: {output.shape}")
        print(f"DEBUG: compress_fun output: {output[0,task_index]}")
        print(f"DEBUG: compress_fun input type: {type(output)}")
        # Convert the input data to the model's expected format BEFORE passing to Attributer
        print(f"DEBUG: x_mut shape before conversion: {x_mut.shape}")
        print(f"DEBUG: x_mut dtype before conversion: {x_mut.dtype}")
        
        # Convert int8 to float32 and transpose from SEAM format to model format
        x_mut_model_format = tf.cast(x_mut, tf.float32)
        x_mut_model_format = tf.transpose(x_mut_model_format, [0, 2, 1])  # (N, L, A) -> (N, A, L)
        x_mut_model_format = x_mut_model_format.numpy()  # Convert back to NumPy for SEAM
        
        print(f"DEBUG: x_mut shape after conversion: {x_mut_model_format.shape}")
        print(f"DEBUG: x_mut dtype after conversion: {x_mut_model_format.dtype}")
        
        # Also convert x_ref
        x_ref_model_format = tf.cast(x_ref, tf.float32)
        x_ref_model_format = tf.transpose(x_ref_model_format, [0, 2, 1])  # (1, L, A) -> (1, A, L)
        x_ref_model_format = x_ref_model_format.numpy()  # Convert back to NumPy for SEAM
        
        print(f"DEBUG: x_ref shape after conversion: {x_ref_model_format.shape}")
        
        # Test the model directly to see what it outputs
        print("DEBUG: Testing model directly...")
        test_input = x_ref_model_format[0:1]  # Take first sequence
        print(f"DEBUG: Test input shape: {test_input.shape}")
        test_output = PERSISTENT_MODEL(test_input)
        print(f"DEBUG: Test output shape: {test_output.shape}")
        print(f"DEBUG: Test output: {test_output}")
        print(f"DEBUG: Test output type: {type(test_output)}")
        
        if task_index is not None:
            try:
                # task_index=1 means we want the second task (index 1 in 0-based indexing)
                # Model output shape is (1, 2), so task_index=1 should work
                print(f"DEBUG: Model output shape: {test_output.shape}")
                print(f"DEBUG: task_index: {task_index}")
                print(f"DEBUG: Available task indices: 0, 1")
                
                if task_index < test_output.shape[1]:
                    test_task_output = test_output[0, task_index]  # First batch, then task
                    print(f"DEBUG: Test task {task_index} output: {test_task_output}")
                    print(f"DEBUG: Test task {task_index} shape: {test_task_output.shape}")
                else:
                    print(f"DEBUG: task_index {task_index} is out of bounds for output shape {test_output.shape}")
            except Exception as e:
                print(f"DEBUG: Error accessing task {task_index}: {e}")
        
        # Create a compress_fun that handles task selection
        def task_compress_fun(pred):
            # pred should be (batch_size, 2) - select the specific task
            return pred[:, task_index]  # Select task_index column
        
        attributer = Attributer(
            PERSISTENT_MODEL,  # Use the original model directly
            method=attribution_method,
            task_index=None,  # Don't use task_index - handle it in compress_fun
            compress_fun=task_compress_fun,  # Custom compress_fun for task selection
            pred_fun=None,  # Not needed when using original model
        )

        # Show params for specific method
        attributer.show_params(attribution_method)  

        t1 = time.time()

        attributions = attributer.compute(
            x=x_mut_model_format,  # Use converted data
            x_ref=x_ref_model_format,  # Use converted data
            save_window=None,
            batch_size=128,
            gpu=gpu
        )

        t2 = time.time() - t1
        print('Attribution time:', t2)
        
        # Debug attribution values
        print(f"DEBUG: Attribution shape: {attributions.shape}")
        print(f"DEBUG: Attribution range: {np.min(attributions):.6f} to {np.max(attributions):.6f}")
        print(f"DEBUG: Attribution non-zero count: {np.count_nonzero(attributions)}")
        print(f"DEBUG: Reference attribution shape: {attributions[ref_index].shape}")
        print(f"DEBUG: Reference attribution range: {np.min(attributions[ref_index]):.6f} to {np.max(attributions[ref_index]):.6f}")
        print(f"DEBUG: Reference attribution non-zero: {np.count_nonzero(attributions[ref_index])}")

        # Transpose attributions back to SEAM format (N, A, L) -> (N, L, A)
        attributions = np.transpose(attributions, [0, 2, 1])
        print(f"DEBUG: Attributions transposed to SEAM format: {attributions.shape}")

        # Render logo of attribution map for reference sequence
        reference_logo = BatchLogo(attributions[ref_index:ref_index+1],
            alphabet=alphabet,
            figsize=[20,2.5],
            center_values=True,
            batch_size=1,
            save_path=os.path.join(seq_folder, 'Reference_logo.png')
        )

        reference_logo.process_all()

        fig, ax = reference_logo.draw_single(ref_index)
        
        # Save the logo to file
        logo_filename = f'reference_attribution_logo_seq{seq_index}_task{task_index}.png'
        logo_path = os.path.join(seq_folder, logo_filename)
        plt.savefig(logo_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reference attribution logo to: {logo_path}")
        
        plt.show()
        

        # =============================================================================
        # SEAM API
        # Cluster attribution maps using Hierarchical Clustering
        # =============================================================================
        

        if clustering_method == 'kmeans':  # Direct K-Means clustering
            # Force CPU for k-means clustering
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            clusterer = Clusterer(
                attributions,
                gpu=False  # Force CPU for k-means
            )

            # For direct k-means, we need to flatten the attributions first
            attributions_flat = attributions.reshape(attributions.shape[0], -1)
            
            # Perform k-means clustering directly on flattened attribution maps (force CPU)
            # Use GPU for k-means clustering
            cluster_labels = clusterer.cluster(
                method='kmeans',
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                embedding=attributions_flat,
                gpu=False  # Force CPU for k-means
            )
            
            # Calculate cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # Ensure cluster labels are zero-indexed and consecutive (0, 1, 2, ...)
            unique_labels_sorted = np.sort(unique_labels)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels_sorted)}
            cluster_labels = np.array([label_map[label] for label in cluster_labels])
                    
            # Store k-means results in clusterer for compatibility with SEAM API
            clusterer.cluster_labels = cluster_labels
            
            # Create a mock kmeans object for compatibility with SEAM API
            class MockKMeans:
                def __init__(self, centroids, labels):
                    self.cluster_centers_ = centroids
                    self.labels_ = labels
            
            # Create dummy centroids (not used by SEAM but needed for compatibility)
            dummy_centroids = np.zeros((n_clusters, attributions.shape[1] * attributions.shape[2]))
            kmeans = MockKMeans(dummy_centroids, cluster_labels)
            clusterer.kmeans_model = kmeans
            
            # Create a mock linkage matrix for compatibility (not used in k-means)
            mock_linkage = np.zeros((n_clusters-1, 4))
            clusterer.linkage = mock_linkage
            
            # Set up membership dataframe that MetaExplainer expects
            clusterer.membership_df = pd.DataFrame({
                'Cluster': cluster_labels,
                'Cluster_Sorted': cluster_labels
            })

            # Get cluster labels (for k-means, these are already the final labels)
            labels_n = clusterer.cluster_labels

            # Reset CUDA_VISIBLE_DEVICES (only if it was originally set)
            if SELECTED_GPU != "Not set":
                os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
            else:
                # If no GPU was originally specified, remove the environment variable
                # to allow all GPUs to be used again
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]


        elif clustering_method == 'tsne+kmeans':  # PCA + K-Means clustering (enabled)
            # Initialize clusterer with PCA method
            clusterer = Clusterer(
                attributions,
                gpu=gpu
            )

            tsne_embedding = clusterer._embed_tsne()

            # Perform k-means clustering on TSNE space using Clusterer's method (force CPU)
            cluster_labels = clusterer.cluster(
                embedding=tsne_embedding,
                method='kmeans',
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                gpu=False  # Force CPU for k-means
            )
                        
             # Calculate cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # Ensure cluster labels are zero-indexed and consecutive (0, 1, 2, ...)
            # This matches how hierarchical clustering handles indexing
            unique_labels_sorted = np.sort(unique_labels)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels_sorted)}
            cluster_labels = np.array([label_map[label] for label in cluster_labels])
                    
            # Store k-means results in clusterer for compatibility with SEAM API
            clusterer.cluster_labels = cluster_labels
            
            # Create a mock kmeans object for compatibility with SEAM API
            # Since we don't have centroids from the Clusterer's method, we'll create empty centroids
            class MockKMeans:
                def __init__(self, centroids, labels):
                    self.cluster_centers_ = centroids
                    self.labels_ = labels
            
            # Create dummy centroids (not used by SEAM but needed for compatibility)
            dummy_centroids = np.zeros((n_clusters, attributions.shape[1] * attributions.shape[2]))
            kmeans = MockKMeans(dummy_centroids, cluster_labels)
            clusterer.kmeans_model = kmeans
            
            # Create a mock linkage matrix for compatibility (not used in k-means)
            # This is just a placeholder to maintain SEAM API compatibility
            mock_linkage = np.zeros((n_clusters-1, 4))
            clusterer.linkage = mock_linkage
            
            # Set up membership dataframe that MetaExplainer expects
            clusterer.membership_df = pd.DataFrame({
                'Cluster': cluster_labels,
                'Cluster_Sorted': cluster_labels
            })

            # Get cluster labels (for k-means, these are already the final labels)
            labels_n = clusterer.cluster_labels
            
        elif clustering_method == 'umap+kmeans':  # Hierarchical clustering with Ward linkage (ENABLED)
            
            clusterer = Clusterer(
                attributions,
                gpu=gpu
            )
            
            umap_embedding = clusterer._embed_umap()

            cluster_labels = clusterer.cluster(
                embedding=umap_embedding,
                method='kmeans',
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                gpu=False  # Force CPU for k-means
            )
                        
             # Calculate cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # Ensure cluster labels are zero-indexed and consecutive (0, 1, 2, ...)
            # This matches how hierarchical clustering handles indexing
            unique_labels_sorted = np.sort(unique_labels)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels_sorted)}
            cluster_labels = np.array([label_map[label] for label in cluster_labels])
                    
            # Store k-means results in clusterer for compatibility with SEAM API
            clusterer.cluster_labels = cluster_labels
            
            # Create a mock kmeans object for compatibility with SEAM API
            # Since we don't have centroids from the Clusterer's method, we'll create empty centroids
            class MockKMeans:
                def __init__(self, centroids, labels):
                    self.cluster_centers_ = centroids
                    self.labels_ = labels
            
            # Create dummy centroids (not used by SEAM but needed for compatibility)
            dummy_centroids = np.zeros((n_clusters, attributions.shape[1] * attributions.shape[2]))
            kmeans = MockKMeans(dummy_centroids, cluster_labels)
            clusterer.kmeans_model = kmeans
            
            # Create a mock linkage matrix for compatibility (not used in k-means)
            # This is just a placeholder to maintain SEAM API compatibility
            mock_linkage = np.zeros((n_clusters-1, 4))
            clusterer.linkage = mock_linkage
            
            # Set up membership dataframe that MetaExplainer expects
            clusterer.membership_df = pd.DataFrame({
                'Cluster': cluster_labels,
                'Cluster_Sorted': cluster_labels
            })

            # Get cluster labels (for k-means, these are already the final labels)
            labels_n = clusterer.cluster_labels

        # =============================================================================
        # SEAM API
        # Generate meta-explanations and related statistics
        # =============================================================================
        sort_method = 'median' # sort clusters by median DNN prediction (default)

        # Initialize MetaExplainer
        print("DEBUG: Initializing MetaExplainer...")
        meta = MetaExplainer(
            clusterer=clusterer,
            mave_df=mave_df,
            attributions=attributions,
            sort_method=sort_method,
            ref_idx=0,
            mut_rate=mut_rate
        )
        print("DEBUG: MetaExplainer initialized")


        # Generate attribution logos with fixed y-axis limits
        logo_type = 'average'  # {average, pwm, enrichment}
        meta_logos = meta.generate_logos(logo_type=logo_type,
            background_separation=False,
            font_name='sans',
            center_values=True
        )
        # Visualize the variation in attribution values across all attribution maps
        meta.plot_attribution_variation(
            scope='all',
            metric='std',
        )
        plt.show()
        # save variation plot to file
        variation_plot_filename = f'variation_plot_seq{seq_index}_task{task_index}.png'
        variation_plot_path = os.path.join(seq_folder, variation_plot_filename)
        plt.savefig(variation_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved variation plot to: {variation_plot_path}")
        plt.show()

        # Visualize the variation in attribution values across cluster averages
        meta.plot_attribution_variation(
            scope='clusters',
            metric='std',
        )
        plt.show()
        # save variation plot to file
        variation_plot_filename = f'variation_plot_seq{seq_index}_task{task_index}_clusters.png'
        variation_plot_path = os.path.join(seq_folder, variation_plot_filename)
        plt.savefig(variation_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved variation plot to: {variation_plot_path}")
        plt.show()

        # Manually create Cluster_Sorted column if sort_method is specified
        if sort_method is not None and meta.cluster_order is not None:
            mapping_dict = {old_k: new_k for new_k, old_k in enumerate(meta.cluster_order)}
            meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping_dict)

        # =============================================================================
        # SEAM API
        # Background separation
        # =============================================================================
        background_multiplier = 0.5  # default threshold factor for background separation

        # View clusters without background
        meta_logos_no_bg = meta.generate_logos(
            logo_type='average',
            background_separation=True,
            mut_rate=mut_rate,
            entropy_multiplier=background_multiplier,
            adaptive_background_scaling=True,
            figsize=(20, 2.5)
        )

        # View average background over all clusters
        average_background_logo = BatchLogo(
            meta.background[np.newaxis, :, :],
            alphabet=meta.alphabet,
            figsize=[20, 2.5],
            batch_size=1,
            font_name='sans'
        )

        average_background_logo.process_all()

        if sort_method is not None:
            ref_cluster = meta.membership_df.loc[ref_index, 'Cluster_Sorted']
        else:
            ref_cluster = meta.membership_df.loc[ref_index, 'Cluster']

        print('WT attribution map')
        fig, ax = reference_logo.draw_single(
            0,
            fixed_ylim=False,
            )
        plt.show()
        # save reference logo to file
        reference_logo_filename = f'reference_logo_seq{seq_index}_task{task_index}_WTavg.png'
        reference_logo_path = os.path.join(seq_folder, reference_logo_filename)
        plt.savefig(reference_logo_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reference logo to: {reference_logo_path}")
        plt.show()

        print('WT cluster: Noise reduction via averaging')
        fig, ax = meta_logos.draw_single(
            ref_cluster,
            fixed_ylim=False,
            )
        plt.show()
        # save reference logo to file
        reference_logo_filename = f'reference_logo_seq{seq_index}_task{task_index}_WTreduced.png'
        reference_logo_path = os.path.join(seq_folder, reference_logo_filename)
        plt.savefig(reference_logo_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reference logo to: {reference_logo_path}")
        plt.show()

        print('WT cluster: Noise reduction and background separation')
        fig, ax = meta_logos_no_bg.draw_single(
            ref_cluster,
            fixed_ylim=False,
            )
        plt.show()

        # save reference logo to file
        reference_logo_filename = f'reference_logo_seq{seq_index}_task{task_index}_WTreduced_no_bg.png'
        reference_logo_path = os.path.join(seq_folder, reference_logo_filename)
        plt.savefig(reference_logo_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reference logo to: {reference_logo_path}")
        plt.show()

        print('Background for WT cluster')
        fig, ax = meta.background_logos.draw_single(
            ref_cluster,
            fixed_ylim=False
        )
        plt.show()

        # save reference logo to file
        reference_logo_filename = f'reference_logo_seq{seq_index}_task{task_index}_WTbg.png'
        reference_logo_path = os.path.join(seq_folder, reference_logo_filename)
        plt.savefig(reference_logo_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reference logo to: {reference_logo_path}")
        plt.show()

        print('Background averaged over all clusters')
        fig, ax = average_background_logo.draw_single(
            0,
            fixed_ylim=False
        )
        plt.show()

        # save reference logo to file
        reference_logo_filename = f'reference_logo_seq{seq_index}_task{task_index}_bg_avg.png'
        reference_logo_path = os.path.join(seq_folder, reference_logo_filename)
        plt.savefig(reference_logo_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved reference logo to: {reference_logo_path}")
        plt.show()

        # Calculate reference cluster-averaged attribution matrix
        print(f"DEBUG: Calculating reference cluster average for cluster {ref_cluster}...")
        ref_cluster_avg = np.mean(meta.get_cluster_maps(ref_cluster), axis=0).astype(np.float16)
        print("DEBUG: Reference cluster average calculated")
        
        reference_attribution = attributions[ref_index].astype(np.float16) # only needed if requested in arrow file below
        
        # Find the cluster-averaged attribution matrix most similar to background
        # Get all cluster-averaged attribution matrices
        num_clusters = len(np.unique(cluster_labels))
        all_cluster_averages = []
        for cluster_idx in range(num_clusters):
            cluster_avg = np.mean(meta.get_cluster_maps(cluster_idx), axis=0)
            all_cluster_averages.append(cluster_avg)
        
        # Apply background scaling factors if available
        if adaptive_background_scaling and meta.background_scaling is not None:
            # Scale the background by each cluster's scaling factor for fair comparison
            scaled_backgrounds = []
            for i in range(len(all_cluster_averages)):
                if i < len(meta.background_scaling):
                    scaling_factor = meta.background_scaling[i]
                    scaled_background = meta.background * scaling_factor
                    scaled_backgrounds.append(scaled_background)
                else:
                    scaled_backgrounds.append(meta.background)
        else:
            scaled_backgrounds = [meta.background] * len(all_cluster_averages)
        
        # Calculate Euclidean distances to background
        distances = []
        for i, cluster_avg in enumerate(all_cluster_averages):
            scaled_background = scaled_backgrounds[i]
            distance = np.linalg.norm(cluster_avg - scaled_background)
            distances.append(distance)
        
        # Find the cluster with minimum distance
        closest_cluster_idx = np.argmin(distances)
        min_distance = distances[closest_cluster_idx]
        
        # Find the sequence in the closest cluster with median prediction value
        # Get all sequences in the closest cluster using SEAM's data structure
        cluster_seqs_df = meta.show_sequences(closest_cluster_idx)
        cluster_predictions = cluster_seqs_df['DNN'].values
        
        # Check if cluster has sequences
        if len(cluster_seqs_df) == 0:
            print(f"ERROR: Cluster {closest_cluster_idx} has no sequences!")
            return
        
        # Find the sequence with median prediction value
        median_prediction = np.median(cluster_predictions)
        # Sort predictions and find the actual median index
        sorted_indices = np.argsort(cluster_predictions)
        median_idx = sorted_indices[len(sorted_indices) // 2]  # True median index
        
        # Get the sequence and prediction at the median index
        background_sequence_row = cluster_seqs_df.iloc[median_idx]
        background_sequence_str = background_sequence_row['Sequence']
        background_prediction = float(background_sequence_row['DNN'])
        
        background_sequence = np.zeros((len(background_sequence_str), len(meta.alphabet)), dtype=np.int8)
        for pos, char in enumerate(background_sequence_str):
            char_idx = meta.alphabet.index(char)
            background_sequence[pos, char_idx] = 1
        
        # Calculate the exact attribution maps used in the logos you said are correct
        # 1. Background averaged over all clusters (from _bg_avg.png)
        # This uses meta.background directly (same as average_background_logo)
        background_averaged_over_all_clusters = meta.background.astype(np.float16)
        
        # 2. WT cluster-specific background (from _WTbg.png) 
        # This should match what meta.background_logos.draw_single(ref_cluster) shows
        # Use the same calculation that SEAM uses internally for cluster-specific backgrounds
        if adaptive_background_scaling and meta.background_scaling is not None:
            wt_cluster_specific_background = (meta.background * meta.background_scaling[ref_cluster]).astype(np.float16)
        else:
            wt_cluster_specific_background = meta.background.astype(np.float16)
        
        # 3. WT cluster foreground (from _WTreduced_no_bg.png)
        # This uses meta_logos_no_bg which is generated with background_separation=True
        # The foreground is the cluster average minus the cluster-specific background
        wt_cluster_foreground = (ref_cluster_avg - wt_cluster_specific_background).astype(np.float16)
        
        # 4. WT sequence onehot (reference sequence)
        wt_sequence_onehot = x_ref[0].astype(np.int8)  # Take first sequence from batch, convert to int8
        
        # Calculate total execution time before saving
        total_time = time.time() - total_start_time
        
        # Create Arrow table with all essential data
        # Convert arrays to bytes for PyArrow compatibility
        table = pa.table({
            # Your 4 specific attribution maps
            'background_averaged_over_all_clusters': [background_averaged_over_all_clusters.tobytes()],
            'wt_cluster_foreground': [wt_cluster_foreground.tobytes()],
            'wt_cluster_specific_background': [wt_cluster_specific_background.tobytes()],
            'wt_sequence_onehot': [wt_sequence_onehot.tobytes()],
            
            # Additional data for compatibility
            'reference_attribution': [reference_attribution.tobytes()],
            'background_sequence_onehot': [background_sequence.tobytes()],
            # 'msm_data': [meta.msm.to_dict('records')],  # Optional - not needed for attribution maps
            
            # Array shapes and dtypes for reconstruction
            'array_shapes': [str(background_averaged_over_all_clusters.shape) + '|' + str(wt_cluster_foreground.shape) + '|' + str(wt_cluster_specific_background.shape) + '|' + str(wt_sequence_onehot.shape) + '|' + str(reference_attribution.shape) + '|' + str(background_sequence.shape)],
            'array_dtypes': [str(background_averaged_over_all_clusters.dtype) + '|' + str(wt_cluster_foreground.dtype) + '|' + str(wt_cluster_specific_background.dtype) + '|' + str(wt_sequence_onehot.dtype) + '|' + str(reference_attribution.dtype) + '|' + str(background_sequence.dtype)],
            'cluster_order': [meta.cluster_order.tolist() if meta.cluster_order is not None else None],
            'sort_method': [sort_method],
            'reference_cluster_index': [ref_cluster]
        })
        
        # Create new schema with metadata
        metadata = {
            b'seq_index': str(seq_index).encode(),
            b'task_index': str(task_index).encode(),
            b'description': b'SEAM attribution maps: meta_bg, wt_cluster_fg, wt_cluster_bg, ref_attr',
            b'runtime_seconds': str(total_time).encode(),
            b'num_seqs': str(num_seqs).encode(),
            b'n_clusters': str(n_clusters).encode(),
            b'clustering_method': clustering_method.encode()
        }
        table = table.replace_schema_metadata(metadata)
        
        # Save Arrow file with compression
        print("DEBUG: Saving Arrow file...")
        filename = f'seq{seq_index}_task{task_index}.arrow'
        filepath = os.path.join(save_path_essential, filename)
        feather.write_feather(table, filepath, compression='lz4')
        print(f"✓ Saved essential data to: {filepath}")
        print("DEBUG: Arrow file saved successfully")

        # =============================================================================
        # Print total execution time
        # =============================================================================
        print(f"\n{'='*60}")
        print(f"SEQUENCE INDEX: {seq_index}")
        print(f"PARAMETER SWEEP INDEX: {parameter_sweep_index}/{len(parameter_sweep_set)}")
        print(f"TASK INDEX: {task_index}")
        print(f"CLUSTERING METHOD: {clustering_method}")
        print(f"NUMBER OF SEQUENCES: {num_seqs}")
        print(f"NUMBER OF CLUSTERS: {n_clusters}")
        print(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"{'='*60}")
        
        # Always release memory pool arrays, even if processing fails
        PERSISTENT_MEMORY_POOL.release_arrays(slots)
            
        # Optional: Print memory pool status for debugging
        # pool_status = PERSISTENT_MEMORY_POOL.get_pool_status()
        # print(f"Memory pool status: {pool_status}")

    # Return the runtime
    time_end = time.time()
    runtime = time_end - time_start
    return runtime


# =============================================================================
# ARROW FILE LOADING UTILITIES
# Minimalistic code to load saved Arrow files and extract essential data
# =============================================================================

def load_arrow_data(filepath):
        """
        Load essential data from a saved Arrow file.
        
        Args:
            filepath: Path to the Arrow file
            
        Returns:
            dict: Dictionary containing all essential data arrays and metadata
        """
        print(f"Loading data from: {filepath}")
        
        # Read Arrow file using PyArrow's native reader to preserve metadata
        try:
            with pa.ipc.open_file(filepath) as reader:
                table = reader.read_all()
        except:
            # Fallback to feather reader
            table = feather.read_feather(filepath)
            # If it's a DataFrame, convert to Table
            if hasattr(table, 'to_arrow'):
                table = table.to_arrow()
        
        # Extract metadata
        metadata = table.schema.metadata
        seq_index = metadata[b'seq_index'].decode()
        task_index = metadata[b'task_index'].decode()
        description = metadata[b'description'].decode()
        
        print(f"Sequence index: {seq_index}")
        print(f"Task index: {task_index}")
        print(f"Description: {description}")
        
        # Convert to pandas for easier data access
        df = table.to_pandas()
        
        # Parse array shapes and dtypes
        shapes_str = df['array_shapes'][0]
        shapes = shapes_str.split('|')
        
        # Parse array dtypes (with fallback for old files)
        try:
            dtypes_str = df['array_dtypes'][0]
            dtypes = dtypes_str.split('|')
            print("Using stored dtypes from Arrow file")
        except (KeyError, IndexError):
            # Fallback for old Arrow files without dtype information
            print("No dtype information found in Arrow file, using default dtypes...")
            dtypes = ['float16'] * len(shapes)
        
        # Handle both old and new format
        if len(shapes) >= 6:  # New format with 3 attribution maps + reference attribution
            meta_bg_shape = eval(shapes[0])
            wt_fg_shape = eval(shapes[1])
            wt_bg_shape = eval(shapes[2])
            ref_attr_shape = eval(shapes[3])
            ref_cluster_shape = eval(shapes[4])
            bg_seq_shape = eval(shapes[5])
            
            meta_bg_dtype = np.dtype(dtypes[0])
            wt_fg_dtype = np.dtype(dtypes[1])
            wt_bg_dtype = np.dtype(dtypes[2])
            ref_attr_dtype = np.dtype(dtypes[3])
            ref_cluster_dtype = np.dtype(dtypes[4])
            bg_seq_dtype = np.dtype(dtypes[5])
        elif len(shapes) >= 5:  # New format with 3 attribution maps (no reference attribution)
            meta_bg_shape = eval(shapes[0])
            wt_fg_shape = eval(shapes[1])
            wt_bg_shape = eval(shapes[2])
            ref_cluster_shape = eval(shapes[3])
            bg_seq_shape = eval(shapes[4])
            
            meta_bg_dtype = np.dtype(dtypes[0])
            wt_fg_dtype = np.dtype(dtypes[1])
            wt_bg_dtype = np.dtype(dtypes[2])
            ref_cluster_dtype = np.dtype(dtypes[3])
            bg_seq_dtype = np.dtype(dtypes[4])
            ref_attr_shape = None
            ref_attr_dtype = None
        else:  # Old format
            ref_cluster_shape = eval(shapes[0])
            background_shape = eval(shapes[1])
            bg_seq_shape = eval(shapes[2])
            
            ref_cluster_dtype = np.dtype(dtypes[0])
            background_dtype = np.dtype(dtypes[1])
            bg_seq_dtype = np.dtype(dtypes[2])
            meta_bg_shape = wt_fg_shape = wt_bg_shape = ref_attr_shape = None
            meta_bg_dtype = wt_fg_dtype = wt_bg_dtype = ref_attr_dtype = None
        
        print(f"Array shapes: meta_bg={meta_bg_shape}, wt_fg={wt_fg_shape}, wt_bg={wt_bg_shape}, ref_cluster={ref_cluster_shape}, bg_seq={bg_seq_shape}")
        
        # Extract and reconstruct arrays
        if len(shapes) >= 6:  # New format with reference attribution
            meta_background = np.frombuffer(df['background_averaged_over_all_clusters'][0], dtype=meta_bg_dtype).reshape(meta_bg_shape)
            wt_cluster_foreground = np.frombuffer(df['wt_cluster_foreground'][0], dtype=wt_fg_dtype).reshape(wt_fg_shape)
            wt_cluster_background = np.frombuffer(df['wt_cluster_specific_background'][0], dtype=wt_bg_dtype).reshape(wt_bg_shape)
            reference_attribution = np.frombuffer(df['reference_attribution'][0], dtype=ref_attr_dtype).reshape(ref_attr_shape)
            ref_cluster_avg = np.frombuffer(df['reference_cluster_average'][0], dtype=ref_cluster_dtype).reshape(ref_cluster_shape)
            background_sequence = np.frombuffer(df['background_sequence_onehot'][0], dtype=bg_seq_dtype).reshape(bg_seq_shape)
            
            # For backward compatibility
            background = meta_background
        elif len(shapes) >= 5:  # New format without reference attribution
            meta_background = np.frombuffer(df['background_averaged_over_all_clusters'][0], dtype=meta_bg_dtype).reshape(meta_bg_shape)
            wt_cluster_foreground = np.frombuffer(df['wt_cluster_foreground'][0], dtype=wt_fg_dtype).reshape(wt_fg_shape)
            wt_cluster_background = np.frombuffer(df['wt_cluster_specific_background'][0], dtype=wt_bg_dtype).reshape(wt_bg_shape)
            ref_cluster_avg = np.frombuffer(df['reference_cluster_average'][0], dtype=ref_cluster_dtype).reshape(ref_cluster_shape)
            background_sequence = np.frombuffer(df['background_sequence_onehot'][0], dtype=bg_seq_dtype).reshape(bg_seq_shape)
            reference_attribution = None
            
            # For backward compatibility
            background = meta_background
        else:  # Old format
            ref_cluster_avg = np.frombuffer(df['reference_cluster_average'][0], dtype=ref_cluster_dtype).reshape(ref_cluster_shape)
            background = np.frombuffer(df['average_background'][0], dtype=background_dtype).reshape(background_shape)
            background_sequence = np.frombuffer(df['background_sequence_onehot'][0], dtype=bg_seq_dtype).reshape(bg_seq_shape)
            
            # Set new variables to None for old format
            meta_background = background
            wt_cluster_foreground = None
            wt_cluster_background = None
            reference_attribution = None
        
        print(f"ref_cluster_avg: min={np.min(ref_cluster_avg):.6f}, max={np.max(ref_cluster_avg):.6f}, mean={np.mean(ref_cluster_avg):.6f}")
        print(f"background: min={np.min(background):.6f}, max={np.max(background):.6f}, mean={np.mean(background):.6f}")
        print(f"background_sequence: min={np.min(background_sequence):.6f}, max={np.max(background_sequence):.6f}, mean={np.mean(background_sequence):.6f}")
        
        # Load MSM data
        msm_data = df['msm_data'][0]
        # Unwrap the list of dictionaries into a proper DataFrame
        # Use json_normalize to properly expand the dictionaries
        msm_df = pd.json_normalize(msm_data)
        print(f"MSM data loaded: {len(msm_data)} records")
        print(f"MSM DataFrame columns after normalization: {list(msm_df.columns)}")
        
        # Load sorting information
        cluster_order = df['cluster_order'][0] if 'cluster_order' in df.columns else None
        sort_method = df['sort_method'][0] if 'sort_method' in df.columns else None
        reference_cluster_index = df['reference_cluster_index'][0] if 'reference_cluster_index' in df.columns else None
        print(f"Sort method: {sort_method}")
        print(f"Cluster order: {cluster_order}")
        print(f"Reference cluster index: {reference_cluster_index}")
        
        return {
            'seq_index': seq_index,
            'task_index': task_index,
            'ref_cluster_avg': ref_cluster_avg,
            'background': background,
            'background_sequence': background_sequence,
            'meta_background': meta_background,
            'wt_cluster_foreground': wt_cluster_foreground,
            'wt_cluster_background': wt_cluster_background,
            'reference_attribution': reference_attribution,
            'msm_df': msm_df,
            'cluster_order': cluster_order,
            'sort_method': sort_method,
            'reference_cluster_index': reference_cluster_index,
            'shapes': {
                'ref_cluster': ref_cluster_shape,
                'background': background_shape if 'background_shape' in locals() else meta_bg_shape,
                'bg_seq': bg_seq_shape,
                'meta_bg': meta_bg_shape if 'meta_bg_shape' in locals() else None,
                'wt_fg': wt_fg_shape if 'wt_fg_shape' in locals() else None,
                'wt_bg': wt_bg_shape if 'wt_bg_shape' in locals() else None,
                'ref_attr': ref_attr_shape if 'ref_attr_shape' in locals() else None
            }
        }


def load_multiple_arrow_files(output_dir, pattern="seq*_task*.arrow"):
    """
    Load multiple Arrow files from a directory.
    
    Args:
        output_dir: Directory containing Arrow files
        pattern: Glob pattern to match Arrow files
        
    Returns:
        list: List of dictionaries containing loaded data
    """
    import glob
    
    arrow_files = glob.glob(os.path.join(output_dir, pattern))
    loaded_data = []
    
    print(f"Found {len(arrow_files)} Arrow files matching pattern '{pattern}'")
    
    for filepath in arrow_files:
        try:
            data = load_arrow_data(filepath)
            loaded_data.append(data)
            print(f"✓ Loaded: {os.path.basename(filepath)} (seq {data['seq_index']}, task {data['task_index']})")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully loaded {len(loaded_data)}/{len(arrow_files)} Arrow files")
    return loaded_data


def extract_essential_summary(loaded_data):
    """
    Extract a summary of essential data across multiple sequences.
    
    Args:
        loaded_data: List of loaded data dictionaries
        
    Returns:
        dict: Summary statistics and aggregated data
    """
    if not loaded_data:
        return {}
    
    # Aggregate reference cluster averages
    ref_cluster_avgs = [data['ref_cluster_avg'] for data in loaded_data]
    
    # Aggregate backgrounds
    backgrounds = [data['background'] for data in loaded_data]
    
    # Extract sequence indices and task indices
    seq_indices = [data['seq_index'] for data in loaded_data]
    task_indices = [data['task_index'] for data in loaded_data]
    
    # Get unique tasks
    unique_tasks = list(set(task_indices))
    
    # Aggregate MSM data
    msm_dfs = [data['msm_df'] for data in loaded_data]
    
    # Get shape information
    shapes_info = [data['shapes'] for data in loaded_data]
    
    # Extract reference cluster indices
    reference_cluster_indices = [data['reference_cluster_index'] for data in loaded_data]
    
    print(f"Summary: {len(loaded_data)} sequences across tasks {unique_tasks}")
    print(f"Sequence indices: {seq_indices}")
    print(f"MSM data: {len(msm_dfs)} DataFrames")
    print(f"Reference cluster indices: {reference_cluster_indices}")
    
    return {
        'num_sequences': len(loaded_data),
        'sequence_indices': seq_indices,
        'unique_tasks': unique_tasks,
        'reference_cluster_averages': ref_cluster_avgs,
        'backgrounds': backgrounds,
        'msm_dataframes': msm_dfs,
        'shapes_info': shapes_info,
        'reference_cluster_indices': reference_cluster_indices,
        'loaded_data': loaded_data
    }

    # Example usage (uncomment to test):
    # if __name__ == '__main__':
    #     # Load a single Arrow file
    #     # data = load_arrow_data('outputs_deepstarr_local_intgrad/seq463513_task1.arrow')
    #     # print(f"Loaded data for sequence {data['seq_index']}, task {data['task_index']}")
    #     # print(f"Reference cluster average shape: {data['ref_cluster_avg'].shape}")
    #     # print(f"MSM DataFrame shape: {data['msm_df'].shape}")
    #     # print(f"Reference cluster index: {data['reference_cluster_index']}")
    #     
    #     # Load multiple Arrow files
    #     # all_data = load_multiple_arrow_files('outputs_deepstarr_local_intgrad')
    #     # summary = extract_essential_summary(all_data)
    #     # print(f"Loaded {summary['num_sequences']} sequences across tasks {summary['unique_tasks']}")
    #     # print(f"MSM DataFrames: {len(summary['msm_dataframes'])}")
    #     # print(f"Reference cluster indices: {summary['reference_cluster_indices']}")


def run_comparison_for_sequence(seq_index, py_dir):
    """Run comparison for a specific sequence after all its parameter combinations are complete"""
    
    # Load reference data for this specific sequence
    reference_file = os.path.join(py_dir, f'seq{seq_index}_task1.arrow')
    if not os.path.exists(reference_file):
        print(f"⚠ Reference file not found: {reference_file}")
        print(f"⏳ Skipping comparisons for seq {seq_index} - reference not ready yet")
        return
        
    ref_data = load_arrow_data(reference_file)
    
    # Initialize CSV file with header if it doesn't exist
    csv_path = os.path.join(py_dir, f'parameter_sweep_results_seq{seq_index}.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("num_seqs,n_clusters,clustering_method,spearmanr_background,spearmanr_foreground,spearmanr_cluster_bg,runtime_seconds\n")

    # Run comparison for all parameter sets for this sequence
    for parameter_sweep_index in range(len(parameter_sweep_set)):
        parameter_set = parameter_sweep_set.iloc[parameter_sweep_index]
        num_seqs = parameter_set['num_seqs']
        n_clusters = parameter_set['n_clusters']
        clustering_method = parameter_set['clustering_method']
        
        # Check if this parameter set was already processed
        already_processed = False
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    if line.startswith(f"{num_seqs},{n_clusters},{clustering_method},"):
                        already_processed = True
                        print(f"  ✓ Parameter set already processed: {num_seqs} seqs, {n_clusters} clusters, {clustering_method}")
                        break
        
        if already_processed:
            continue
            
        save_path_essential = os.path.join(py_dir, f'parameter_set_{num_seqs}_{n_clusters}_{clustering_method}_results')

        # Check if parameter set results exist for this sequence
        param_file = os.path.join(save_path_essential, f'seq{seq_index}_task1.arrow')
        if not os.path.exists(param_file):
            print(f"⚠ Parameter set result not found: {param_file}")
            continue
            
        # Load parameter set result
        param_data = load_arrow_data(param_file)
        
        # Calculate Spearman correlations
        from scipy.stats import spearmanr
        
        # Load reference data - use the keys returned by load_arrow_data
        ref_bg = ref_data['meta_background']  # This is background_averaged_over_all_clusters
        ref_fg = ref_data['wt_cluster_foreground']
        ref_bg_cluster = ref_data['wt_cluster_background']  # This is wt_cluster_specific_background
        
        # Load parameter data - use the keys returned by load_arrow_data
        param_bg = param_data['meta_background']  # This is background_averaged_over_all_clusters
        param_fg = param_data['wt_cluster_foreground']
        param_bg_cluster = param_data['wt_cluster_background']  # This is wt_cluster_specific_background
        
        bg_corr, _ = spearmanr(
            ref_bg.flatten(), 
            param_bg.flatten()
        )
        
        fg_corr, _ = spearmanr(
            ref_fg.flatten(), 
            param_fg.flatten()
        )
        
        bg_cluster_corr, _ = spearmanr(
            ref_bg_cluster.flatten(), 
            param_bg_cluster.flatten()
        )
        
        # Get runtime from Arrow file metadata (with fallback for old format)
        runtime = param_data.get('runtime_seconds', 0.0)
        
        print(f"  Seq {seq_index}: {num_seqs} seqs, {n_clusters} clusters, {clustering_method} -> bg={bg_corr:.4f}, fg={fg_corr:.4f}, cluster_bg={bg_cluster_corr:.4f} (runtime: {runtime:.1f}s)")
        
        # Save line to CSV
        with open(csv_path, 'a') as f:
            f.write(f"{num_seqs},{n_clusters},{clustering_method},{bg_corr:.6f},{fg_corr:.6f},{bg_cluster_corr:.6f},{runtime:.2f}\n")


def load_arrow_data(filepath):
    """
    Load essential data from a saved Arrow file.
    
    Args:
        filepath: Path to the Arrow file
        
    Returns:
        dict: Dictionary containing all essential data arrays and metadata
    """
    print(f"Loading data from: {filepath}")
    
    # Read Arrow file using PyArrow's native reader to preserve metadata
    try:
        with pa.ipc.open_file(filepath) as reader:
            table = reader.read_all()
    except:
        # Fallback to feather reader
        table = feather.read_feather(filepath)
        # If it's a DataFrame, convert to Table
        if hasattr(table, 'to_arrow'):
            table = table.to_arrow()
    
    # Extract metadata
    metadata = table.schema.metadata
    seq_index = metadata[b'seq_index'].decode()
    task_index = metadata[b'task_index'].decode()
    description = metadata[b'description'].decode()
    
    # Extract runtime and parameter info from metadata (with fallback for old files)
    runtime_seconds = float(metadata.get(b'runtime_seconds', b'0.0').decode())
    num_seqs = int(metadata.get(b'num_seqs', b'100000').decode())
    n_clusters = int(metadata.get(b'n_clusters', b'50').decode())
    clustering_method = metadata.get(b'clustering_method', b'hierarchical').decode()
    
    # For reference files (old format), set default parameters
    if runtime_seconds == 0.0 and b'runtime_seconds' not in metadata:
        runtime_seconds = 0.0  # Unknown runtime for reference files
        num_seqs = 100000  # Reference uses 100K sequences
        n_clusters = 50  # Reference uses 50 clusters  
        clustering_method = 'hierarchical'  # Reference uses hierarchical clustering
    
    print(f"Sequence index: {seq_index}")
    print(f"Task index: {task_index}")
    print(f"Description: {description}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    print(f"Parameters: {num_seqs} seqs, {n_clusters} clusters, {clustering_method}")
    
    # Convert to pandas for easier data access
    df = table.to_pandas()
    
    # Parse array shapes and dtypes
    shapes_str = df['array_shapes'][0]
    shapes = shapes_str.split('|')
    
    # Parse array dtypes (with fallback for old files)
    try:
        dtypes_str = df['array_dtypes'][0]
        dtypes = dtypes_str.split('|')
        print("Using stored dtypes from Arrow file")
    except (KeyError, IndexError):
        # Fallback for old Arrow files without dtype information
        print("No dtype information found in Arrow file, using default dtypes...")
        dtypes = ['float16'] * len(shapes)
    
    # Handle both old and new format
    if len(shapes) >= 6:  # New format with 3 attribution maps + reference attribution
        meta_bg_shape = eval(shapes[0])
        wt_fg_shape = eval(shapes[1])
        wt_bg_shape = eval(shapes[2])
        wt_seq_shape = eval(shapes[3])  # wt_sequence_onehot shape
        ref_attr_shape = eval(shapes[4])  # reference_attribution shape
        bg_seq_shape = eval(shapes[5])
        
        meta_bg_dtype = np.dtype(dtypes[0])
        wt_fg_dtype = np.dtype(dtypes[1])
        wt_bg_dtype = np.dtype(dtypes[2])
        wt_seq_dtype = np.dtype(dtypes[3])  # wt_sequence_onehot is int8
        ref_attr_dtype = np.dtype(dtypes[4])  # reference_attribution is float16
        bg_seq_dtype = np.dtype(dtypes[5])
    elif len(shapes) >= 5:  # New format with 3 attribution maps (no reference attribution)
        meta_bg_shape = eval(shapes[0])
        wt_fg_shape = eval(shapes[1])
        wt_bg_shape = eval(shapes[2])
        ref_cluster_shape = eval(shapes[3])
        bg_seq_shape = eval(shapes[4])
        
        meta_bg_dtype = np.dtype(dtypes[0])
        wt_fg_dtype = np.dtype(dtypes[1])
        wt_bg_dtype = np.dtype(dtypes[2])
        ref_cluster_dtype = np.dtype(dtypes[3])
        bg_seq_dtype = np.dtype(dtypes[4])
        ref_attr_shape = None
        ref_attr_dtype = None
    else:  # Old format
        ref_cluster_shape = eval(shapes[0])
        background_shape = eval(shapes[1])
        bg_seq_shape = eval(shapes[2])
        
        ref_cluster_dtype = np.dtype(dtypes[0])
        background_dtype = np.dtype(dtypes[1])
        bg_seq_dtype = np.dtype(dtypes[2])
        meta_bg_shape = wt_fg_shape = wt_bg_shape = ref_attr_shape = None
        meta_bg_dtype = wt_fg_dtype = wt_bg_dtype = ref_attr_dtype = None
    
    print(f"Array shapes: meta_bg={meta_bg_shape}, wt_fg={wt_fg_shape}, wt_bg={wt_bg_shape}, wt_seq={wt_seq_shape}, ref_attr={ref_attr_shape}, bg_seq={bg_seq_shape}")
    print(f"Available columns: {list(df.columns)}")
    
    # Extract and reconstruct arrays
    if len(shapes) >= 6:  # New format with reference attribution
        meta_background = np.frombuffer(df['background_averaged_over_all_clusters'][0], dtype=meta_bg_dtype).reshape(meta_bg_shape)
        wt_cluster_foreground = np.frombuffer(df['wt_cluster_foreground'][0], dtype=wt_fg_dtype).reshape(wt_fg_shape)
        wt_cluster_background = np.frombuffer(df['wt_cluster_specific_background'][0], dtype=wt_bg_dtype).reshape(wt_bg_shape)
        wt_sequence_onehot = np.frombuffer(df['wt_sequence_onehot'][0], dtype=wt_seq_dtype).reshape(wt_seq_shape)
        reference_attribution = np.frombuffer(df['reference_attribution'][0], dtype=ref_attr_dtype).reshape(ref_attr_shape)
        background_sequence = np.frombuffer(df['background_sequence_onehot'][0], dtype=bg_seq_dtype).reshape(bg_seq_shape)
        
        # Reconstruct ref_cluster_avg from wt_cluster_foreground + wt_cluster_specific_background
        ref_cluster_avg = wt_cluster_foreground + wt_cluster_background
        
        # For backward compatibility
        background = meta_background
    elif len(shapes) >= 5:  # New format without reference attribution
        meta_background = np.frombuffer(df['background_averaged_over_all_clusters'][0], dtype=meta_bg_dtype).reshape(meta_bg_shape)
        wt_cluster_foreground = np.frombuffer(df['wt_cluster_foreground'][0], dtype=wt_fg_dtype).reshape(wt_fg_shape)
        wt_cluster_background = np.frombuffer(df['wt_cluster_specific_background'][0], dtype=wt_bg_dtype).reshape(wt_bg_shape)
        ref_cluster_avg = np.frombuffer(df['reference_cluster_average'][0], dtype=ref_cluster_dtype).reshape(ref_cluster_shape)
        background_sequence = np.frombuffer(df['background_sequence_onehot'][0], dtype=bg_seq_dtype).reshape(bg_seq_shape)
        reference_attribution = None
        
        # For backward compatibility
        background = meta_background
    else:  # Old format
        ref_cluster_avg = np.frombuffer(df['reference_cluster_average'][0], dtype=ref_cluster_dtype).reshape(ref_cluster_shape)
        background = np.frombuffer(df['average_background'][0], dtype=background_dtype).reshape(background_shape)
        background_sequence = np.frombuffer(df['background_sequence_onehot'][0], dtype=bg_seq_dtype).reshape(bg_seq_shape)
        
        # Set new variables to None for old format
        meta_background = background
        wt_cluster_foreground = None
        wt_cluster_background = None
        reference_attribution = None
    
    return {
        'seq_index': seq_index,
        'task_index': task_index,
        'ref_cluster_avg': ref_cluster_avg,
        'background': background,
        'background_sequence': background_sequence,
        'meta_background': meta_background,
        'wt_cluster_foreground': wt_cluster_foreground,
        'wt_cluster_background': wt_cluster_background,
        'reference_attribution': reference_attribution,
        'runtime_seconds': runtime_seconds,
        'num_seqs': num_seqs,
        'n_clusters': n_clusters,
        'clustering_method': clustering_method
    }


def load_multiple_arrow_files(output_dir, pattern="seq*_task*.arrow"):
    """
    Load multiple Arrow files from a directory.
    
    Args:
        output_dir: Directory containing Arrow files
        pattern: Glob pattern to match Arrow files
        
    Returns:
        list: List of dictionaries containing loaded data
    """
    import glob
    
    arrow_files = glob.glob(os.path.join(output_dir, pattern))
    loaded_data = []
    
    print(f"Found {len(arrow_files)} Arrow files matching pattern '{pattern}'")
    
    for filepath in arrow_files:
        try:
            data = load_arrow_data(filepath)
            loaded_data.append(data)
            print(f"✓ Loaded: {os.path.basename(filepath)} (seq {data['seq_index']}, task {data['task_index']})")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully loaded {len(loaded_data)}/{len(arrow_files)} Arrow files")
    return loaded_data


if __name__ == "__main__":
    py_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process by sequence order: all parameter combinations for seq0, then seq1, etc.
    # This allows starting comparisons as soon as reference seq0 is ready
    seq_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # All 15 reference sequences
    
    for seq_index in seq_indices:
        print(f"\n{'='*60}")
        print(f"PROCESSING ALL PARAMETER COMBINATIONS FOR SEQ {seq_index}")
        print(f"{'='*60}")
        
        for parameter_sweep_index in range(len(parameter_sweep_set)):
            print(f"Running parameter sweep {parameter_sweep_index + 1} of {len(parameter_sweep_set)} for seq {seq_index}")
            runtime = run_parameter_sweep(parameter_sweep_index, seq_index)
            print(f"Runtime: {runtime} seconds")
        
        print(f"✓ Completed all parameter combinations for seq {seq_index}")
        
        # Run comparison for this sequence immediately after all its parameter sets are done
        print(f"\nChecking if reference is ready for seq {seq_index}...")
        reference_file = os.path.join(py_dir, f'seq{seq_index}_task1.arrow')
        if os.path.exists(reference_file):
            print(f"✓ Reference found - running comparisons for seq {seq_index}...")
            run_comparison_for_sequence(seq_index, py_dir)
            print(f"✓ Completed comparisons for seq {seq_index}")
        else:
            print(f"⏳ Reference not ready for seq {seq_index} - skipping comparisons for now")
            print(f"   (Reference file: {reference_file})")
            print(f"   You can run comparisons later when references are complete")

    print("Parameter sweep complete, all arrow files saved")




