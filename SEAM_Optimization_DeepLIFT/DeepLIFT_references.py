## full imports for one-time load
import os, sys
import torch, torch.nn as nn, torch.fx as fx, copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import random
from scipy import stats
import tangermeme
from tangermeme.predict import predict
import time
from tqdm.auto import tqdm
from tangermeme.plot import plot_logo
torch.cuda.empty_cache()

# Then retry with device='cuda'
import gc
gc.collect()

# Then retry with device='cuda'
torch.cuda.empty_cache()
# Run FIRST in the notebook (after a kernel restart), before importing/using Torch/TF
SEED = 113

# Env vars that must be set before CUDA libraries initialize
os.environ['PYTHONHASHSEED'] = str(SEED)


# Python/NumPy
random.seed(SEED)
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# CuPy (if used)
try:
    import cupy as cp
    cp.random.seed(SEED)
except Exception:
    pass

#==============================================================================
#Initialize the model
#==============================================================================
model_dir  = "/grid/wsbs/home_norepl/pmantill/Trained_Model_Zoo/EvoAug_Distilled_Student_Model"
model_path = os.path.join(model_dir, "EvoAug_student_model.pt")

# Make sure distill_EvoAug2.py is in model_dir
sys.path.insert(0, model_dir)
import distill_EvoAug2 as distill_mod  # avoid shadowing 'model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_model = torch.load(model_path, map_location=device).to(device).eval()


py_directory = os.getcwd()
test_library_path = os.path.join(py_directory, '../../test_library_df.csv')

# Load test dataframe
test_library_df = pd.read_csv(test_library_path)

# Parse sequences directly from your script's method
reftestseqs = []
for i, row in test_library_df.iterrows():
    seq_str = row['seq_one_hot']
    clean_str = seq_str.replace('[', '').replace(']', '').replace('\n', ' ').replace('  ', ' ')
    values = [float(x) for x in clean_str.split() if x.strip()]
    seq_onehot = np.array(values, dtype=np.float32).reshape(4, -1).T
    reftestseqs.append(seq_onehot)
PERSISTENT_XDATASET = np.array(reftestseqs)
PERSISTENT_XDATASET = PERSISTENT_XDATASET.transpose(0, 2, 1)

print(PERSISTENT_XDATASET.shape)
first_seq = PERSISTENT_XDATASET[0:1]
print(first_seq.shape)

pytorch_model(torch.tensor(first_seq, dtype=torch.float32).to(device))
y = predict(pytorch_model, torch.tensor(first_seq, dtype=torch.float32).to(device), batch_size=256, func=lambda x: x['mean'])

#==============================================================================
#Set up SEAM
#==============================================================================
#SEAM set up
import os, sys

# Allow GPU usage for most operations, but k-means will be forced to CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Commented out to allow SLURM to detect best GPU

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import random
import numpy as np
import tensorflow as tf
# TensorFlow
tf.random.set_seed(SEED)


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from itertools import product

tf.get_logger().setLevel('ERROR')
# Suppress TensorRT warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
PERSISTENT_NUM_SEQS_SWEEP = [100000]  # number of sequences to generate (5K for debugging)
PERSISTENT_N_CLUSTERS_SWEEP = [50]  # number of clusters for  clustering
PERSISTENT_CLUSTERING_METHOD_SWEEP = ['hierarchical']  # clustering method (hierarchical already done by hierarchal_sweep.py)
PERSISTENT_N_PCA_COMPONENTS = 10  # number of PCA components for dimensionality reduction
PERSISTENT_SEQ_LEN = 249  # hardcoded sequence length for DeepSTARR (all sequences are 249bp)
PERSISTENT_ATTRIBUTION_METHOD = 'DeepLIFT'  # {saliency, smoothgrad, intgrad, ism}
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
    global PERSISTENT_MODEL, PERSISTENT_X_DATASET, PERSISTENT_MEMORY_POOL
    # Avoid re-initializing large pools on every call
    if (
        PERSISTENT_MEMORY_POOL is not None
        and PERSISTENT_MODEL is not None
        and PERSISTENT_X_DATASET is not None
    ):
        return
    
    PERSISTENT_X_DATASET = PERSISTENT_XDATASET

    PERSISTENT_MODEL = pytorch_model
    
    print("Initializing memory pool for 100K sequences...")
    PERSISTENT_MEMORY_POOL = MemoryPool(max_seqs=PERSISTENT_NUM_SEQS, max_seq_length=PERSISTENT_SEQ_LEN, max_pca_components=PERSISTENT_N_PCA_COMPONENTS)

#==============================================================================
# Parameter sweep function
#==============================================================================
def run_parameter_sweep(parameter_sweep_index, seq_index, task_index):
    """Run parameter sweep for a specific parameter combination and sequence"""
    start_time = time.time()
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================================================================
    # Check if this parameter set has already been processed
    # =============================================================================
    py_dir = py_directory
    save_path_essential = os.path.join(py_dir, f'Reference_arrows/parameter_set_{num_seqs}_{n_clusters}_{clustering_method}_results')
    
    seq_folder = f"Reference_arrows/Seq{seq_index}"
    arrow_filename = f"seq{seq_index}_task{task_index}.arrow"
    arrow_filepath = os.path.join(seq_folder, arrow_filename)
    if os.path.exists(arrow_filepath):
        print(f"✓ Sequence {seq_index} already processed - Arrow file exists: {arrow_filename}")
        return 0.0

    # =============================================================================
    # Use persistent model and data
    # =============================================================================
    # Use the persistent model and SEAM wrapper instead of loading them again
    X_dataset = PERSISTENT_X_DATASET.transpose(0, 2, 1)
    alphabet = PERSISTENT_ALPHABET
    x_ref = X_dataset[seq_index]
    x_ref = np.expand_dims(x_ref, 0)  # Already in SEAM format (batch, length, channels)

    # Define mutagenesis window for sequence
    seq_length = x_ref.shape[1]  # SEAM format: sequence length is second dimension
    mut_window = [0, seq_length]  # [start_position, stop_position]
    
    # Forward pass to get output for the specific head using SEAM wrapper
    output = predict(pytorch_model, torch.tensor(first_seq, dtype=torch.float32).to(device), batch_size=256, func=lambda x: x['mean'])
    print(f"Output shape: {output.shape}")
    print(f"Output type: {type(output)}")
    # Don't select task here - let compress_fun handle it
    # pred = output[:, task_index:task_index+1]  # Keep full batch dimension

    print(f"\nWild-type prediction: {output}")
    print(f"Prediction shape: {output.shape}")
    print(f"Prediction type: {type(output)}")
    print(f"Prediction type: {type(x_ref)}")

    # =============================================================================
    # SQUID API
    # Create in silico mutagenesis library
    # =============================================================================
    time_start = time.time()
    print(f"========== STARTING SEQ {seq_index} TASK {task_index} ==========")
    print(f'num seqs: {num_seqs}')
    print(f"n clusters: {n_clusters}")
    print(f"clustering method: {clustering_method}")
    print(f"mut rate: {mut_rate}")
    print(f"attribution method: {attribution_method}")

    def torch_pred_fun(x_np):
        # x_np: (L, A) or (B, L, A), np.ndarray or torch.Tensor
        if isinstance(x_np, torch.Tensor):
            x_t = x_np.to(device=device, dtype=torch.float32)
        else:
            if x_np.ndim == 2:
                x_np = x_np[None, ...]
            x_t = torch.from_numpy(np.ascontiguousarray(x_np)).to(device=device, dtype=torch.float32)
        # -> (B, A, L)
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)
        x_t = x_t.permute(0, 2, 1).contiguous()

        with torch.inference_mode():
            out = pytorch_model(x_t)
            y = out["mean"].detach().cpu().numpy()
        return y.T
    # Set up predictor class for in silico MAVE
    pred_generator = squid.predictor.ScalarPredictor(
        pred_fun=torch_pred_fun,
        task_idx=task_index,  # Match the actual task index we're using
        batch_size=512
    )

    # Set up mutagenizer class for in silico MAVE
    mut_generator = squid.mutagenizer.RandomMutagenesis(
        mut_rate=mut_rate,
        seed=42
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
    y_mut[:] = y_mut_temp.astype(np.float16).reshape(-1, 1)  # Convert to float16 for memory efficiency
    print(x_mut.shape)
    print(y_mut.shape)
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
    print(mave_df.shape)

    from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear

    # Wrap to expose only the mean head (B,2)
    class MeanOnlyModel(nn.Module):
        def __init__(self, student_model):
            super().__init__()
            self.student = student_model
        def forward(self, X):
            out = self.student(X)   # dict
            return out["mean"]      # (B,2)

    def uniquify_relu_calls(model: nn.Module) -> nn.Module:
        gm = fx.symbolic_trace(model)
        modules = dict(gm.named_modules())
        counter = 0
        for node in gm.graph.nodes:
            if node.op == 'call_module' and isinstance(modules[node.target], nn.ReLU):
                new_name = f"_relu_callsite_{counter}"
                counter += 1
                setattr(gm, new_name, nn.ReLU(inplace=False))
                node.target = new_name
        gm.recompile()
        return gm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PERSISTENT_MODEL.to(device).eval()

    attr_model = MeanOnlyModel(pytorch_model).eval()
    attr_model = copy.deepcopy(attr_model).eval()
    attr_model = uniquify_relu_calls(attr_model).to(device).eval()

    # Disable TF32 for DeepLIFT - more accurate but slower
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False


    def deeplift_attr_with_progress(x_np_BLA, target_idx, batch_size, n_shuffles):
        B, L, A = x_np_BLA.shape
        out = np.empty((B, L, A), dtype=np.float16)
        for s in range(0, B, batch_size):
            e = min(s + batch_size, B)
            x_batch = x_np_BLA[s:e]  # (b, L, A)
            x_t = torch.from_numpy(np.ascontiguousarray(x_batch)).to(device, dtype=torch.float32)
            x_t = x_t.permute(0, 2, 1).contiguous()          # -> (b, A, L)
            x_t.requires_grad_(True)                         # CRITICAL
            #print(target_idx)
            # IMPORTANT: do NOT wrap this in no_grad/inference_mode
            attr_BAL = deep_lift_shap(
                attr_model,
                x_t,
                target=target_idx,                # 0=Dev, 1=HK
                device='cuda',
                batch_size=x_t.shape[0],
                n_shuffles=n_shuffles,
                print_convergence_deltas=False,
                additional_nonlinear_ops={nn.ReLU: _nonlinear},
                verbose=True
            )

            attr_BLA = attr_BAL.permute(0, 2, 1).contiguous().cpu().detach().numpy()  # -> (b, L, A)
            out[s:e] = attr_BLA.astype(np.float16)

        return out

    #New (DeepLIFT):
    print(f'Starting DeepLIFT attribution...')
    attributions = deeplift_attr_with_progress(
        x_np_BLA=x_mut,               # (N, L, A) np/int8 is fine
        target_idx=task_index,        # keep consistent with your Dev/HK
        batch_size=5096,              # increased for H100 80GB
        n_shuffles=64                 # fewer shuffles for faster runtime
    ).astype(np.float16)

    attributions_pool[:] = attributions.astype(np.float16)

    t2 = time.time() - start_time
    print('Attribution time:', t2)
    tf.keras.backend.clear_session()
    #tf.config.experimental.reset_memory_stats('GPU:0')
    gc.collect()

    # =============================================================================
    # SEAM API
    # Cluster attribution maps using Hierarchical Clustering (References)
    # =============================================================================
    if clustering_method == 'kmeans':  # Direct K-Means clustering on original feature space
        
        clusterer = Clusterer(
            attributions_pool,  # Use pre-allocated pool for memory efficiency
            gpu=gpu  # Use GPU if available
        )
        
        # Perform k-means clustering directly on original attribution maps
        # Flatten attribution maps for k-means: (N, L, A) -> (N, L*A)
        attributions_flat = attributions_pool.reshape(attributions_pool.shape[0], -1)
        print(f"Flattened attribution shape: {attributions_flat.shape}")
        
        cluster_labels = clusterer.cluster(
            embedding=attributions_flat,  # Pass flattened attribution maps
            method='kmeans',
            n_clusters=n_clusters,  # This will be 20 from your parameter sweep
        )
        
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
        
        

        # Get cluster labels (for k-means, these are already the final labels)
        labels_n = clusterer.cluster_labels
        
        # Set up membership dataframe that MetaExplainer expects
        clusterer.membership_df = pd.DataFrame({
            'Cluster': cluster_labels,
            'Cluster_Sorted': cluster_labels
        })
        #print(clusterer.membership_df.head(20))
    elif clustering_method == 'hierarchical':  # Hierarchical clustering with Ward linkage (ENABLED)
        clusterer = Clusterer(
            attributions,
            gpu=gpu
        )

        # Perform hierarchical clustering directly on attribution maps
        link_method = 'ward'
        linkage = clusterer.cluster(
            method='hierarchical',
            link_method=link_method
        )
        
        # Get cluster labels from hierarchical clustering
        labels_n, cut_level = clusterer.get_cluster_labels(
            linkage,
            criterion='maxclust',
            n_clusters=n_clusters
        )
        
        # Use the hierarchical clustering labels
        cluster_labels = labels_n
        
        # Store hierarchical clustering results in clusterer for compatibility with SEAM API
        clusterer.cluster_labels = cluster_labels
        clusterer.linkage = linkage
        
        # Set up membership dataframe that MetaExplainer expects
        clusterer.membership_df = pd.DataFrame({
            'Cluster': cluster_labels,
            'Cluster_Sorted': cluster_labels
        })
        #print(clusterer.membership_df.head(20))
    # =============================================================================
    # SEAM API
    # Generate meta-explanations and related statistics
    # =============================================================================
    sort_method = 'median' # sort clusters by median DNN prediction (default)

    # Initialize MetaExplainer
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method=sort_method,
        ref_idx=0,
        mut_rate=mut_rate
    )

    # Generate Mechanism Summary Matrix (MSM) - essential for TFBS identification
    msm = meta.generate_msm(
        gpu=gpu
    )

    # Manually create Cluster_Sorted column if sort_method is specified
    if sort_method is not None and meta.cluster_order is not None:
        mapping_dict = {old_k: new_k for new_k, old_k in enumerate(meta.cluster_order)}
        meta.membership_df["Cluster_Sorted"] = meta.membership_df["Cluster"].map(mapping_dict)

    cluster_order = meta.get_cluster_order()  # array of original labels in sorted order
    label_to_sorted = {old: i for i, old in enumerate(cluster_order)}
    
    # 1) membership mapping is consistent
    assert (meta.membership_df["Cluster_Sorted"].to_numpy()
        == meta.membership_df["Cluster"].map(label_to_sorted).to_numpy()).all()
 

    # =============================================================================
    # SEAM API
    # Background separation
    # =============================================================================
    entropy_multiplier = 0.5  # default threshold factor for background separation
    

    meta.compute_background(mut_rate=mut_rate, entropy_multiplier=entropy_multiplier, adaptive_background_scaling = True, process_logos=False)

    # Always use original cluster ID for scaling
    ref_cluster_original = meta.membership_df.loc[ref_index, 'Cluster']
    
    # Map original cluster ID -> position in cluster_order
    cluster_order = meta.get_cluster_order()
    label_to_sorted = {old: i for i, old in enumerate(cluster_order)}
    ref_sorted_idx = label_to_sorted[ref_cluster_original]   # scalar int



    # Get average background attribution
    average_background_attribution = meta.background.astype(np.float16)


    # Get sequences in WT cluster (use original cluster label)
    k_idxs = meta.mave['Cluster'] == ref_cluster_original
    # Compute WT cluster average 
    WT_cluster_avg_raw = np.mean(meta.attributions[k_idxs], axis=0)

    # Seperate out the two types of backgrounds
    WT_cluster_specific_foreground = WT_cluster_avg_raw - (meta.cluster_backgrounds[ref_sorted_idx])
    WT_cluster_scaled_foreground = WT_cluster_avg_raw - (meta.background * meta.background_scaling[ref_sorted_idx])

    # save both backgrounds
    scaled_background = meta.background * meta.background_scaling[ref_sorted_idx]
    cluster_background = meta.cluster_backgrounds[ref_sorted_idx]

    mask = meta.mave['Hamming'].eq(0)
    wt_pos = np.flatnonzero(mask)[0]  # raises if none found
    WT_sequence_attribution = meta.attributions[wt_pos]
    
    # =============================================================================
    # Visualize with BatchLogo, save plots
    # =============================================================================

    all_wt_attributions = np.stack([
        WT_sequence_attribution,      # Individual WT sequence attribution
        WT_cluster_avg_raw,           # WT cluster attribution (before background subtraction)
        cluster_background,                # WT cluster-specific background
        WT_cluster_specific_foreground,        # WT foreground (cluster-specific background subtraction)
        scaled_background,            # WT scaled background attribution
        WT_cluster_scaled_foreground, # WT foreground (scaled background)
        average_background_attribution            # Average background across all clusters
    ])
    
    # Create BatchLogo for all WT attributions
    wt_logos = BatchLogo(
        all_wt_attributions,
        alphabet=meta.alphabet,
        figsize=[10, 2],
        center_values=True,
        batch_size=7,
        font_name='sans'
    )

    wt_logos.process_all()
    
    # Create a 7x1 subplot layout (vertical stacking)
    fig, axes = plt.subplots(7, 1, figsize=(20, 21))
    fig.suptitle(f'Reference Attributions Seq{seq_index} - Hierarchical clustering with 50 clusters on 100K sequences', fontsize=12, y=0.99)
    
    # Adjust subplot spacing to prevent overlap with main title
    plt.subplots_adjust(top=0.88, hspace=0.3)
    
    # Plot each logo
    titles = [
        'WT Sequence Attribution',
        'WT Cluster Attribution',
        'WT Cluster-Specific Background',
        'WT Foreground (Cluster-Specific)',
        'WT Scaled Background Attribution',
        'WT Foreground (Scaled Background)',
        'Average Background (All Clusters)'
    ]
    
    for i, (title, ax) in enumerate(zip(titles, axes.flat)):
        wt_logos.draw_single(i, ax=ax, fixed_ylim=False)
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    # Save the combined plot
    seq_folder = f'Reference_arrows/Seq{seq_index}'
    if not os.path.exists(seq_folder):
        os.makedirs(seq_folder)
    combined_logo_filename = f'SEAM_Optimization_Reference(Hierarchical clustering with 50 clusters on 100K sequences).png'
    combined_logo_path = os.path.join(seq_folder, combined_logo_filename)
    plt.savefig(combined_logo_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined WT attribution maps to: {combined_logo_path}")

    total_execution_time = time.time() - start_time
    print(f"Total execution time: {total_execution_time:.2f} seconds")


    # =============================================================================
    # Save all attribution maps
    # =============================================================================

     # Convert arrays to bytes for PyArrow compatibility
    table = pa.table({
        # Your 4 specific attribution maps
        'background_averaged_over_all_clusters': [average_background_attribution.tobytes()],
        'wt_cluster_foreground': [WT_cluster_specific_foreground.tobytes()],
        'wt_cluster_scaled_foreground': [WT_cluster_scaled_foreground.tobytes()],
        'wt_cluster_specific_background': [cluster_background.tobytes()],
        'scaled_background': [scaled_background.tobytes()],
        
        
        # Array shapes and dtypes for reconstruction
        'array_shapes': [str(average_background_attribution.shape) + '|' + str(WT_cluster_specific_foreground.shape) + '|' + str(WT_cluster_scaled_foreground.shape) + '|' + str(cluster_background.shape) + '|' + str(scaled_background.shape)],
        'array_dtypes': [str(average_background_attribution.dtype) + '|' + str(WT_cluster_specific_foreground.dtype) + '|' + str(WT_cluster_scaled_foreground.dtype) + '|' + str(cluster_background.dtype) + '|' + str(scaled_background.dtype)],
        'cluster_order': [meta.cluster_order.tolist() if meta.cluster_order is not None else None],
        'sort_method': [sort_method],
        'reference_cluster_index': [ref_cluster_original]
    })
    
    # Create new schema with metadata
    metadata = {
        b'seq_index': str(seq_index).encode(),
        b'task_index': str(task_index).encode(),
        b'description': b'SEAM attribution maps: average_background_attribution, WT_cluster_specific_foreground, WT_cluster_scaled_foreground, cluster_background, scaled_background',
        b'runtime_seconds': str(total_execution_time).encode(),
        b'num_seqs': str(num_seqs).encode(),
        b'n_clusters': str(n_clusters).encode(),
        b'clustering_method': clustering_method.encode()
    }
    table = table.replace_schema_metadata(metadata)
    
     # Save Arrow file with compression
    print("DEBUG: Saving Arrow file...")
    filename = f'seq{seq_index}_task{task_index}.arrow'
    save_path_essential = f'Reference_arrows/Seq{seq_index}'
    os.makedirs(save_path_essential, exist_ok=True)
    filepath = os.path.join(save_path_essential, filename)
    feather.write_feather(table, filepath, compression='lz4')

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
    print(f"TOTAL EXECUTION TIME: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    # Always release memory pool arrays, even if processing fails
    PERSISTENT_MEMORY_POOL.release_arrays(slots)
    
    # Force garbage collection to free up memory
    gc.collect()
    tf.keras.backend.clear_session()

#==============================================================================
# Main function to sweep through parameters
#==============================================================================
## Run through all sequences, check for existing and skip if found
task_index = 1
parameter_sweep_index = 0
parameter_sweep_current = parameter_sweep_set.iloc[0]
for seq_index in range(len(test_library_df)):
    seq_folder = f'Reference_arrows/Seq{seq_index}'
    arrow_path = os.path.join(seq_folder, f'seq{seq_index}_task{task_index}.arrow')
    if os.path.exists(arrow_path):
        print(f"Skipping existing file: {arrow_path}")
        continue
    os.makedirs(seq_folder, exist_ok=True)
    print(f"No file found for Seq{seq_index} - parameter set {parameter_sweep_current}, running parameter sweep")
    run_parameter_sweep(0, seq_index, task_index)
print("All Reference Attributions Complete")