#!/usr/bin/env python3
"""
SEAM Parameter Sweep Analysis

This script performs a comprehensive parameter sweep for SEAM optimization:
- Library size optimization (100 to 100,000 sequences)
- Clustering method comparison (4 different methods as per paper)
- Number of clusters optimization (10 to 50 clusters)

Uses pre-computed reference attributions from 15 sequences with 100k sequences + hierarchical clustering.
"""

import os
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from typing import Tuple, Dict, Any, Optional, List
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import json
import pickle
import glob

# Try to import UMAP and Leiden
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ UMAP not available. Install with: pip install umap-learn")

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("⚠ Leiden clustering not available. Install with: pip install leidenalg python-igraph")

class SEAMParameterSweep:
    """
    SEAM parameter sweep analysis using pre-computed reference attributions.
    """
    
    def __init__(self, gpu_id: int = 3):
        self.gpu_id = gpu_id
        self.keras_model = None
        self.alphabet = ['A', 'C', 'G', 'T']
        self.reference_attributions = {}  # Store reference attributions
        self.results = {}  # Store sweep results
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU configuration"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        conda_prefix = '/grid/wsbs/home_norepl/pmantill/miniconda3/envs/EvoAug2_env'
        cuda_lib_path = f'{conda_prefix}/lib'
        
        os.environ['LD_LIBRARY_PATH'] = f'{cuda_lib_path}:{os.environ.get("LD_LIBRARY_PATH", "")}'
        os.environ['CUDA_HOME'] = conda_prefix
        os.environ['CUDA_ROOT'] = conda_prefix
        os.environ['PATH'] = f'{cuda_lib_path}:{os.environ.get("PATH", "")}'
        
        print(f"✓ GPU {self.gpu_id} configured")
    
    def load_keras_model(self, model_path: str):
        """Load the converted Keras model and apply SEAM patches"""
        
        print(f"Loading Keras model from: {model_path}")
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ TensorFlow configured for GPU {self.gpu_id} with memory growth enabled")
        
        # Apply SEAM format patch - SAME AS seam_optimization_analysis.py
        print("Applying SEAM format patch...")
        from seam_format_patch import patch_seam_format
        patch_seam_format()
        
        self.keras_model = tf.keras.models.load_model(model_path)
        
        # Test model
        test_input = tf.random.normal((1, 4, 249))
        test_output = self.keras_model(test_input)
        print(f"✓ Keras model loaded successfully")
        print(f"✓ Model test: input shape {test_input.shape} -> output shape {test_output.shape}")
        print(f"✓ Model has both Dev and Hk tasks: {test_output.shape[1] == 2}")
        
        print(f"✓ Model loaded successfully - ready for SEAM attribution computation")
    
    def load_reference_sequences(self, csv_path: str, num_sequences: int = 15):
        """Load 15 reference sequences from CSV file"""
        import pandas as pd
        
        print(f"Loading {num_sequences} reference sequences from: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Convert DNA sequences to one-hot encoding using tangermeme
        from tangermeme.utils import one_hot_encode
        
        def dna_to_onehot(dna_seq):
            # tangermeme's one_hot_encode returns (alphabet, seq_length) format as PyTorch tensor
            onehot = one_hot_encode(dna_seq)
            # Convert PyTorch tensor to numpy (no transpose needed)
            return onehot.numpy()
        
        # Convert sequences
        X_test = np.array([dna_to_onehot(seq) for seq in df['sequence']])
        
        # Create y_test from true_dev_label (we'll use Dev activity for both tasks for now)
        y_test = np.column_stack([df['true_dev_label'].values, df['true_dev_label'].values])
        
        print(f"✓ Loaded {X_test.shape[0]} sequences with shape {X_test.shape}")
        print(f"✓ Dev activity range: {y_test[:, 0].min():.3f} to {y_test[:, 0].max():.3f}")
        print(f"✓ Activity bins: {df['activity_bin'].value_counts().to_dict()}")
        print(f"✓ Sequences loaded in CSV order (indices 0-{X_test.shape[0]-1})")
        
        return X_test, y_test
    
    def load_reference_attributions(self, reference_dir: str, task_index: int = 0):
        """Load pre-computed reference attributions for all sequences (Dev task only)"""
        print(f"Loading reference attributions from: {reference_dir}")
        print(f"Task: {task_index} (Dev)")
        
        reference_attributions = {}
        
        # Look for reference attribution files for Dev task only
        # First, look for problematic (new) files
        problematic_pattern = f"temp_reference_attributions_task{task_index}_seq*_problematic.pkl"
        problematic_files = glob.glob(os.path.join(reference_dir, problematic_pattern))
        
        if not problematic_files:
            # Try current directory if reference_dir doesn't exist
            problematic_files = glob.glob(problematic_pattern)
        
        # Then look for regular files
        pattern = f"temp_reference_attributions_task{task_index}_seq*.pkl"
        regular_files = glob.glob(os.path.join(reference_dir, pattern))
        
        if not regular_files:
            # Try current directory if reference_dir doesn't exist
            regular_files = glob.glob(pattern)
        
        # Combine files, prioritizing problematic (new) ones
        ref_files = []
        used_sequences = set()
        
        # Add problematic files first (these are the new regenerated ones)
        # Include seq1 problematic file now
        for file in problematic_files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            seq_idx = None
            for i, part in enumerate(parts):
                if part.startswith('seq'):
                    seq_part = part.replace('seq', '')
                    # Handle cases like "seq1_problematic.pkl" by taking only the number part
                    seq_num = seq_part.split('_')[0]  # Split on underscore and take first part
                    seq_num = seq_num.split('.')[0]  # Remove file extension
                    seq_idx = int(seq_num)
                    break
            if seq_idx is not None:  # Include all problematic files including seq1
                ref_files.append(file)
                used_sequences.add(seq_idx)
        
        # No special handling needed - use problematic file for seq1
        
        # Add regular files for sequences not already covered by problematic files
        for file in regular_files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            seq_idx = None
            for i, part in enumerate(parts):
                if part.startswith('seq'):
                    seq_part = part.replace('seq', '')
                    # Handle cases like "seq1_BAD.pkl" by taking only the number part
                    seq_num = seq_part.split('_')[0]  # Split on underscore and take first part
                    seq_num = seq_num.split('.')[0]  # Remove file extension
                    seq_idx = int(seq_num)
                    break
            if seq_idx is not None and seq_idx not in used_sequences:
                ref_files.append(file)
        
        if not ref_files:
            raise FileNotFoundError(f"No reference attribution files found for task {task_index}")
        
        print(f"Found {len(ref_files)} reference attribution files")
        
        # Sort files by sequence number, not alphabetically
        def get_seq_number(filename):
            basename = os.path.basename(filename)
            # Extract sequence number from filename like "temp_reference_attributions_task0_seq0_problematic.pkl"
            # or "temp_reference_attributions_task0_seq0.pkl"
            parts = basename.split('_')
            for part in parts:
                if part.startswith('seq'):
                    # Extract number after 'seq' and before any suffix
                    seq_part = part.replace('seq', '')
                    # Handle cases like "seq1_problematic" or "seq1_BAD" by taking only the number part
                    seq_num = seq_part.split('_')[0]  # Split on underscore and take first part
                    # Remove any file extension
                    seq_num = seq_num.split('.')[0]
                    return int(seq_num)
            return 0  # fallback
        
        for ref_file in sorted(ref_files, key=get_seq_number):
            # Extract sequence index from filename
            filename = os.path.basename(ref_file)
            # Extract sequence number from filename like "temp_reference_attributions_task0_seq0_problematic.pkl"
            parts = filename.split('_')
            seq_idx = None
            for part in parts:
                if part.startswith('seq'):
                    # Extract number after 'seq' and before any suffix
                    seq_part = part.replace('seq', '')
                    # Handle cases like "seq1_problematic" or "seq1_BAD" by taking only the number part
                    seq_num = seq_part.split('_')[0]  # Split on underscore and take first part
                    # Remove any file extension
                    seq_num = seq_num.split('.')[0]
                    seq_idx = int(seq_num)
                    break
            
            print(f"  Loading sequence {seq_idx} from: {os.path.basename(ref_file)}")
            try:
                with open(ref_file, 'rb') as f:
                    ref_data = pickle.load(f)
                
        # No special transpose handling needed for problematic files
                
                reference_attributions[seq_idx] = ref_data
                print(f"    ✓ Loaded reference attribution for sequence {seq_idx}")
            except Exception as e:
                print(f"    ✗ Error loading sequence {seq_idx}: {e}")
                continue
        
        print(f"✓ Loaded reference attributions for {len(reference_attributions)} sequences")
        print(f"✓ Available sequence indices: {sorted(reference_attributions.keys())}")
        
        return reference_attributions
    
    def generate_mutagenesis_library(self, reference_sequence: np.ndarray, 
                                   num_sequences: int, 
                                   mut_rate: float = 0.1,
                                   task_index: int = 0):
        """Generate mutagenesis library using SQUID API"""
        import squid
        
        print(f"  Generating {num_sequences} sequences...")
        
        # Ensure reference has batch dimension
        x_ref = np.expand_dims(reference_sequence, 0)
        seq_length = x_ref.shape[2]
        # SQUID expects [start, stop] where stop is exclusive
        mut_window = [0, seq_length]
        
        # Set up predictor class for in silico MAVE
        # SQUID generates (seq_length, alphabet) but model expects (alphabet, seq_length)
        def predict_wrapper(x_batch):
            # Transpose from (batch, seq_length, alphabet) to (batch, alphabet, seq_length)
            x_transposed = np.transpose(x_batch, (0, 2, 1))
            # Convert uint8 to float32
            x_transposed = x_transposed.astype(np.float32)
            predictions = self.keras_model.predict_on_batch(x_transposed)
            # Return predictions for the specific task as a 1D array
            return predictions[:, task_index].flatten()
        
        pred_generator = squid.predictor.ScalarPredictor(
            pred_fun=predict_wrapper,
            task_idx=None,  # Don't use task_idx, we handle it in the wrapper
            batch_size=500  # Use a smaller batch size to avoid the last batch issue
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
        
        # Generate mutagenesis data
        # SQUID expects (seq_length, alphabet) but we have (alphabet, seq_length)
        x_ref_transposed = x_ref[0].T  # Transpose from (4, 249) to (249, 4)
        
        # Generate sequences and predictions using SQUID MAVE
        x_mut, y_mut = mave.generate(x_ref_transposed, num_sim=num_sequences)
        
        # Fix y_mut shape - SQUID returns (1, num_batches, batch_size) but we need (num_sequences,)
        if y_mut.ndim == 3:
            y_mut = y_mut.flatten()
        elif y_mut.ndim == 2:
            y_mut = y_mut.flatten()
        
        print(f"    SQUID output shapes:")
        print(f"      x_mut: {x_mut.shape}, dtype: {x_mut.dtype}")
        print(f"      y_mut: {y_mut.shape}, dtype: {y_mut.dtype}")
        
        # Fix y_mut shape - SQUID returns (1, num_batches, batch_size) but we need (num_sequences,)
        if y_mut.ndim == 3:
            y_mut = y_mut.flatten()
        elif y_mut.ndim == 2:
            y_mut = y_mut.flatten()
        
        print(f"    After reshape:")
        print(f"      y_mut: {y_mut.shape}, dtype: {y_mut.dtype}")
        
        # SQUID generates (seq_length, alphabet) but SEAM expects (alphabet, seq_length)
        # Transpose back: (num_sequences, seq_length, alphabet) -> (num_sequences, alphabet, seq_length)
        x_mut_transposed = np.transpose(x_mut, (0, 2, 1))
        
        # Convert to float32 for SEAM compatibility
        x_mut_transposed = x_mut_transposed.astype(np.float32)
        
        print(f"    After transpose:")
        print(f"      x_mut_transposed: {x_mut_transposed.shape}, dtype: {x_mut_transposed.dtype}")
        print(f"      y_mut: {y_mut.shape}, dtype: {y_mut.dtype}")
        
        return x_mut_transposed, y_mut
    
    def compute_attributions(self, x_mut: np.ndarray, y_mut: np.ndarray, x_ref: np.ndarray, 
                           task_index: int = 0, num_steps: int = 10):
        """Compute attribution maps using SEAM API with our patches"""
        from seam import Attributer, Compiler, Clusterer
        
        print(f"  Computing attributions...")
        print(f"    x_mut shape: {x_mut.shape}")
        print(f"    y_mut shape: {y_mut.shape}")
        print(f"    x_ref shape: {x_ref.shape}")
        
        # SEAM API - Compile sequence analysis data
        # x_mut is now in (batch, alphabet, seq_length) format from generate_mutagenesis_library
        # No transpose needed for Compiler
        x_mut_for_compiler = x_mut
        
        compiler = Compiler(
            x=x_mut_for_compiler,
            y=y_mut,
            x_ref=x_ref,
            y_bg=None,
            alphabet=self.alphabet,
            gpu=self.gpu_id
        )
        
        mave_df = compiler.compile()
        
        # Use SEAM's patched _intgrad_gpu method directly (same as seam_optimization_analysis.py)
        attributer = Attributer(
            self.keras_model,
            method='intgrad',
            task_index=task_index,
            compress_fun=lambda x: tf.reduce_sum(x)  # Explicit scalar reduction for gradient computation
        )
        
        # Transpose data from (batch, alphabet, seq_length) to (batch, seq_length, alphabet) for SEAM
        x_mut_for_seam = np.transpose(x_mut_for_compiler, (0, 2, 1))  # (batch, alphabet, seq_length) -> (batch, seq_length, alphabet)
        x_ref_for_seam = np.transpose(x_ref, (0, 2, 1))  # (batch, alphabet, seq_length) -> (batch, seq_length, alphabet)
        
        # Compute attribution maps using SEAM's compute method (same as seam_optimization_analysis.py)
        attributions = attributer.compute(
            x=x_mut_for_seam,
            x_ref=x_ref_for_seam,
            save_window=None,
            batch_size=32,  # Use same batch size as seam_optimization_analysis.py
            gpu=self.gpu_id,
            num_steps=num_steps
        )
        
        print(f"✓ Attributions computed: shape {attributions.shape}, dtype {attributions.dtype}")
        print(f"✓ Attribution range: {attributions.min():.6f} to {attributions.max():.6f}")
        
        return attributions, mave_df
    
    def cluster_attributions(self, attributions: np.ndarray, 
                           method: str = 'kmeans',
                           n_clusters: int = 50,
                           embedding_method: str = None):
        """
        Cluster attribution maps using SEAM's built-in clustering methods.
        
        Methods:
        - leiden: Leiden clustering on k-nearest neighbor graph with modularity vertex partition
        - kmeans: K-means on PCA embedding using SEAM's GPU-optimized method
        - kmeans_umap: K-means on UMAP embedding using SEAM's built-in UMAP method
        - kmeans_tsne: K-means on t-SNE embedding using SEAM's built-in t-SNE method
        """
        from seam import Clusterer
        
        print(f"  Clustering with {method} (k={n_clusters})...")
        
        if method == 'leiden':
            if not LEIDEN_AVAILABLE:
                raise ValueError("Leiden clustering not available. Install with: pip install leidenalg python-igraph")
            
            # Leiden clustering on k-nearest neighbor graph with modularity vertex partition
            attributions_flat = attributions.reshape(attributions.shape[0], -1)
            
            # Build k-nearest neighbor graph
            n_neighbors = min(15, attributions_flat.shape[0] - 1)
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(attributions_flat)
            knn_graph = knn.kneighbors_graph(attributions_flat, mode='connectivity')
            
            # Convert to igraph
            sources, targets = knn_graph.nonzero()
            weights = knn_graph.data
            edges = list(zip(sources, targets))
            
            # Create igraph graph
            g = ig.Graph(edges, directed=False)
            g.es['weight'] = weights
            
            # Leiden clustering with modularity optimization
            # This automatically optimizes the number of clusters
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, 
                                              resolution_parameter=1.0, random_state=42)
            cluster_labels = np.array(partition.membership)
            
            # If we need a specific number of clusters, we can adjust the resolution parameter
            # or use a different approach, but Leiden typically finds the optimal number automatically
            
        elif method == 'kmeans':
            # Use SEAM's GPU-optimized K-means clustering with PCA embedding
            clusterer = Clusterer(
                attributions,
                method='pca',
                gpu=True
            )
            
            # Compute PCA embedding first
            n_components = min(50, attributions.shape[1] * attributions.shape[2])
            pca_embedding = clusterer.embed(
                n_components=n_components,
                plot_eigenvalues=False,
                save_path=None
            )
            
            # Perform k-means clustering on PCA space
            cluster_labels = clusterer.cluster(
                embedding=pca_embedding,
                method='kmeans',
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
        elif method == 'kmeans_umap':
            # Use SEAM's built-in UMAP method + K-means
            clusterer = Clusterer(
                attributions,
                method='umap',
                gpu=True
            )
            
            # Compute UMAP embedding using SEAM's method
            n_components = min(50, attributions.shape[1] * attributions.shape[2])
            umap_embedding = clusterer.embed(
                n_components=n_components,
                plot_eigenvalues=False,
                save_path=None
            )
            
            # Perform k-means clustering on UMAP space using SEAM's method
            cluster_labels = clusterer.cluster(
                embedding=umap_embedding,
                method='kmeans',
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
        elif method == 'kmeans_tsne':
            # Use SEAM's built-in t-SNE method + K-means
            clusterer = Clusterer(
                attributions,
                method='tsne',
                gpu=True
            )
            
            # Compute t-SNE embedding using SEAM's method
            n_components = min(50, attributions.shape[1] * attributions.shape[2])
            tsne_embedding = clusterer.embed(
                n_components=n_components,
                plot_eigenvalues=False,
                save_path=None
            )
            
            # Perform k-means clustering on t-SNE space using SEAM's method
            cluster_labels = clusterer.cluster(
                embedding=tsne_embedding,
                method='kmeans',
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return cluster_labels
    
    def generate_background_attribution(self, attributions: np.ndarray, 
                                      cluster_labels: np.ndarray):
        """Generate background attribution map (reference)"""
        # Find the cluster most similar to the average background
        num_clusters = len(np.unique(cluster_labels))
        cluster_averages = []
        
        for cluster_idx in range(num_clusters):
            cluster_mask = cluster_labels == cluster_idx
            if np.any(cluster_mask):
                cluster_avg = np.mean(attributions[cluster_mask], axis=0)
                cluster_averages.append(cluster_avg)
            else:
                cluster_averages.append(np.zeros_like(attributions[0]))
        
        # Calculate average background
        overall_background = np.mean(attributions, axis=0)
        
        # Find cluster closest to overall background
        distances = []
        for cluster_avg in cluster_averages:
            distance = np.linalg.norm(cluster_avg - overall_background)
            distances.append(distance)
        
        closest_cluster_idx = np.argmin(distances)
        background_attribution = cluster_averages[closest_cluster_idx]
        
        return background_attribution, closest_cluster_idx
    
    def evaluate_configuration(self, reference_sequence: np.ndarray,
                             task_index: int,
                             num_sequences: int,
                             clustering_method: str,
                             n_clusters: int,
                             reference_attribution: np.ndarray,
                             seq_idx: int):
        """
        Evaluate a specific configuration against the reference.
        Returns Spearman correlation, runtime, and background attribution.
        """
        print(f"  Evaluating: seq{seq_idx}, {num_sequences} seqs, {clustering_method}, {n_clusters} clusters")
        
        start_time = time.time()
        
        try:
            # Generate mutagenesis library
            x_mut, y_mut = self.generate_mutagenesis_library(
                reference_sequence, num_sequences, task_index=task_index
            )
            
            # Compute attributions
            x_ref = np.expand_dims(reference_sequence, 0)
            attributions, mave_df = self.compute_attributions(x_mut, y_mut, x_ref, task_index)
            
            # Cluster attributions
            cluster_labels = self.cluster_attributions(
                attributions, method=clustering_method, n_clusters=n_clusters
            )
            
            # Generate background attribution
            background_attribution, background_cluster = self.generate_background_attribution(
                attributions, cluster_labels
            )
            
            # Calculate Spearman correlation
            correlation, _ = spearmanr(
                reference_attribution.flatten(),
                background_attribution.flatten()
            )
            
            total_time = time.time() - start_time
            
            print(f"    ✓ Correlation: {correlation:.6f}, Time: {total_time:.2f}s")
            
            return {
                'correlation': correlation,
                'runtime': total_time,
                'success': True,
                'background_attribution': background_attribution,
                'background_cluster': background_cluster,
                'attributions': attributions,
                'cluster_labels': cluster_labels
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"    ✗ Error: {e}")
            return {
                'correlation': np.nan,
                'runtime': total_time,
                'success': False,
                'error': str(e),
                'background_attribution': None,
                'background_cluster': None,
                'attributions': None,
                'cluster_labels': None
            }
    
    def optimize_library_size(self, sequences: np.ndarray, task_index: int, reference_attributions: Dict):
        """
        C.1 Optimizing number of sequences
        Test library sizes: 100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000
        """
        print(f"\n{'='*60}")
        print(f"LIBRARY SIZE OPTIMIZATION")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"{'='*60}")
        
        # Test different library sizes
        library_sizes = [100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000]
        results = []
        
        # Create CSV file for incremental saving
        csv_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/seam_parameter_sweep_results/library_size_sweep_results.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Check if CSV file already exists
        if os.path.exists(csv_path):
            print(f"✓ Found existing CSV file: {csv_path}")
            print(f"  → Will append new results to existing file")
        else:
            # Initialize CSV with header if it doesn't exist
            with open(csv_path, 'w') as f:
                f.write("phase,sequence,library_size,clustering_method,n_clusters,correlation,runtime_seconds\n")
            print(f"✓ Created new CSV file: {csv_path}")
        
        # Only use sequences that have NEW problematic reference attributions
        # Focus on seq1 only as requested
        problematic_sequences = [1]  # Focus on seq1 only
        
        if not problematic_sequences:
            print("⚠ No problematic sequences found with new reference attributions")
            print("  → Skipping library size optimization")
            return []
        
        available_sequences = problematic_sequences
        
        print(f"✓ Focusing on seq1 only: {sorted(available_sequences)}")
        print(f"✓ Using the same approach that worked for seq0")
        
        for num_sequences in library_sizes:
            print(f"\n--- Testing {num_sequences} sequences ---")
            seq_results = []
            
            for seq_idx in available_sequences:
                # Use the SAME sequence for both test and reference
                reference_sequence = sequences[seq_idx]
                reference_attribution = reference_attributions[seq_idx]['background_attribution']
                print(f"    Testing seq{seq_idx}: using sequences[{seq_idx}] (shape: {reference_sequence.shape})")
                print(f"      Reference attribution range: {reference_attribution.min():.6f} to {reference_attribution.max():.6f}")
                
                result = self.evaluate_configuration(
                    reference_sequence, task_index, num_sequences,
                    'kmeans', 50, reference_attribution, seq_idx
                )
                
                seq_results.append({
                    'seq_idx': seq_idx,
                    'num_sequences': num_sequences,
                    'clustering_method': 'kmeans',
                    'n_clusters': 50,
                    'correlation': result['correlation'],
                    'runtime': result['runtime'],
                    'success': result['success']
                })
                
                # Save to CSV immediately after each result
                with open(csv_path, 'a') as f:
                    f.write(f"library_size,seq{seq_idx},{num_sequences},kmeans,50,{result['correlation']:.6f},{result['runtime']:.2f}\n")
                print(f"    ✓ Saved result to CSV: seq{seq_idx}, {num_sequences} seqs, correlation={result['correlation']:.6f}")
            
            results.extend(seq_results)
        
        return results
    
    def optimize_clustering_method(self, sequences: np.ndarray, task_index: int, reference_attributions: Dict):
        """
        C.2 Optimizing cluster analysis
        Test methods: leiden, kmeans, kmeans_umap, kmeans_tsne
        """
        print(f"\n{'='*60}")
        print(f"CLUSTERING METHOD OPTIMIZATION")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"{'='*60}")
        
        # Test different clustering methods
        clustering_methods = ['leiden', 'kmeans', 'kmeans_umap', 'kmeans_tsne']
        results = []
        
        for method in clustering_methods:
            print(f"\n--- Testing {method} ---")
            seq_results = []
            
            for seq_idx in reference_attributions.keys():
                # Use the SAME sequence for both test and reference
                reference_sequence = sequences[seq_idx]
                reference_attribution = reference_attributions[seq_idx]['background_attribution']
                
                result = self.evaluate_configuration(
                    reference_sequence, task_index, 10000,  # Use 10,000 sequences
                    method, 50, reference_attribution, seq_idx
                )
                
                seq_results.append({
                    'seq_idx': seq_idx,
                    'num_sequences': 10000,
                    'clustering_method': method,
                    'n_clusters': 50,
                    'correlation': result['correlation'],
                    'runtime': result['runtime'],
                    'success': result['success']
                })
            
            results.extend(seq_results)
        
        return results
    
    def optimize_number_of_clusters(self, sequences: np.ndarray, task_index: int, reference_attributions: Dict):
        """
        C.3 Optimizing number of clusters
        Test cluster numbers: 10, 20, 30, 40, 50
        """
        print(f"\n{'='*60}")
        print(f"NUMBER OF CLUSTERS OPTIMIZATION")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"{'='*60}")
        
        # Test different numbers of clusters
        n_clusters_list = [10, 20, 30, 40, 50]
        results = []
        
        for n_clusters in n_clusters_list:
            print(f"\n--- Testing {n_clusters} clusters ---")
            seq_results = []
            
            for seq_idx in reference_attributions.keys():
                # Use the SAME sequence for both test and reference
                reference_sequence = sequences[seq_idx]
                reference_attribution = reference_attributions[seq_idx]['background_attribution']
                
                result = self.evaluate_configuration(
                    reference_sequence, task_index, 10000,  # Use 10,000 sequences
                    'kmeans', n_clusters, reference_attribution, seq_idx  # Use kmeans
                )
                
                seq_results.append({
                    'seq_idx': seq_idx,
                    'num_sequences': 10000,
                    'clustering_method': 'kmeans',
                    'n_clusters': n_clusters,
                    'correlation': result['correlation'],
                    'runtime': result['runtime'],
                    'success': result['success']
                })
            
            results.extend(seq_results)
        
        return results
    
    def run_full_optimization(self, sequences: np.ndarray, reference_attributions: Dict,
                             task_index: int = 0, 
                             output_path: str = None, resume: bool = True):
        """
        Run the complete SEAM parameter sweep analysis.
        
        This analysis uses:
        - Pre-computed reference attributions (100k sequences + hierarchical clustering)
        - 4 clustering methods: leiden, kmeans, kmeans_umap, kmeans_tsne
        - SEAM API for clustering and background generation
        - SQUID API for mutagenesis library generation
        """
        print(f"\n{'='*60}")
        print(f"SEAM PARAMETER SWEEP ANALYSIS")
        print(f"Using: Pre-computed reference attributions + 4 clustering methods")
        print(f"Sequences: {len(sequences)}")
        print(f"Task: {task_index} (Dev)")
        print(f"{'='*60}")
        
        # Check if results already exist and resume if requested
        if resume and output_path and os.path.exists(output_path):
            print(f"✓ Found existing results file: {output_path}")
            try:
                with open(output_path, 'rb') as f:
                    all_results = pickle.load(f)
                print("✓ Loaded existing results. Use resume=False to recompute from scratch.")
                return all_results
            except Exception as e:
                print(f"⚠ Error loading existing results: {e}")
                print("→ Starting fresh computation...")
        
        all_results = {}
        
        print(f"\n{'='*60}")
        print(f"DEV TASK PARAMETER SWEEP")
        print(f"{'='*60}")
        
        task_results = {}
        
        # Use first sequence for optimization analysis
        reference_sequence = sequences[0]
        
        # C.1: Library size optimization
        print(f"\n--- C.1: Library Size Optimization ---")
        library_results = self.optimize_library_size(
            sequences, task_index, reference_attributions
        )
        task_results['library_size'] = library_results
        
        # Save intermediate results after library size optimization
        if output_path:
            temp_results = {f'task_{task_index}': task_results}
            temp_output = output_path.replace('.pkl', '_temp_library_size.pkl')
            self.save_results(temp_results, temp_output)
            print(f"✓ Intermediate results saved to: {temp_output}")
        
        # C.2: Clustering method optimization
        print(f"\n--- C.2: Clustering Method Optimization ---")
        clustering_results = self.optimize_clustering_method(
            sequences, task_index, reference_attributions
        )
        task_results['clustering_method'] = clustering_results
        
        # Save intermediate results after clustering method optimization
        if output_path:
            temp_results = {f'task_{task_index}': task_results}
            temp_output = output_path.replace('.pkl', '_temp_clustering_method.pkl')
            self.save_results(temp_results, temp_output)
            print(f"✓ Intermediate results saved to: {temp_output}")
        
        # C.3: Number of clusters optimization
        print(f"\n--- C.3: Number of Clusters Optimization ---")
        clusters_results = self.optimize_number_of_clusters(
            sequences, task_index, reference_attributions
        )
        task_results['number_of_clusters'] = clusters_results
        
        all_results[f'task_{task_index}'] = task_results
        all_results[f'task_{task_index}_reference_attributions'] = reference_attributions
        
        # Force memory cleanup after each task
        import gc
        gc.collect()
        tf.keras.backend.clear_session()
        print(f"\n✓ Task {task_index} completed successfully!")
        
        return all_results
    
    def save_results(self, results: Dict, output_path: str):
        """Save optimization results"""
        print(f"\nSaving results to: {output_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        
        # Save as pickle for full data preservation
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Also save as JSON for easy viewing (without numpy arrays)
        json_results = {}
        for task_key, task_data in results.items():
            json_results[task_key] = {}
            for analysis_key, analysis_data in task_data.items():
                if analysis_key == 'reference':
                    # Skip reference data for JSON
                    continue
                json_results[task_key][analysis_key] = analysis_data
        
        json_path = output_path.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
        print(f"✓ JSON summary saved to {json_path}")
    
    def print_summary(self, results: Dict):
        """Print optimization summary"""
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP SUMMARY (DEV TASK)")
        print(f"{'='*60}")
        
        for task_key, task_data in results.items():
            if not task_key.startswith('task_') or task_key.endswith('_reference_attributions'):
                continue
                
            task_idx = int(task_key.split('_')[1])
            print(f"\nDev Task Results:")
            
            # Library size results
            library_results = task_data['library_size']
            successful_library = [r for r in library_results if r['success']]
            if successful_library:
                best_library = max(successful_library, key=lambda x: x['correlation'])
                avg_correlation = np.mean([r['correlation'] for r in successful_library])
                print(f"  C.1 Library size: {len(successful_library)}/{len(library_results)} successful")
                print(f"  Best correlation: {best_library['correlation']:.6f} "
                      f"(seq{best_library['seq_idx']}, {best_library['num_sequences']} seqs)")
                print(f"  Average correlation: {avg_correlation:.6f}")
            
            # Clustering method results
            clustering_results = task_data['clustering_method']
            successful_clustering = [r for r in clustering_results if r['success']]
            if successful_clustering:
                best_clustering = max(successful_clustering, key=lambda x: x['correlation'])
                avg_correlation = np.mean([r['correlation'] for r in successful_clustering])
                print(f"  C.2 Clustering methods: {len(successful_clustering)}/{len(clustering_results)} successful")
                print(f"  Best correlation: {best_clustering['correlation']:.6f} "
                      f"(seq{best_clustering['seq_idx']}, {best_clustering['clustering_method']})")
                print(f"  Average correlation: {avg_correlation:.6f}")
            
            # Number of clusters results
            clusters_results = task_data['number_of_clusters']
            successful_clusters = [r for r in clusters_results if r['success']]
            if successful_clusters:
                best_clusters = max(successful_clusters, key=lambda x: x['correlation'])
                avg_correlation = np.mean([r['correlation'] for r in successful_clusters])
                print(f"  C.3 Cluster numbers: {len(successful_clusters)}/{len(clusters_results)} successful")
                print(f"  Best correlation: {best_clusters['correlation']:.6f} "
                      f"(seq{best_clusters['seq_idx']}, {best_clusters['n_clusters']} clusters)")
                print(f"  Average correlation: {avg_correlation:.6f}")


def main():
    """Main function to run SEAM parameter sweep analysis"""
    
    # Configuration
    model_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/EvoAug_keras_model"
    csv_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/evoaug_15seq_results_updated.csv"
    reference_dir = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment"  # Directory with reference attribution files
    output_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/seam_parameter_sweep_results/seam_parameter_sweep_results.pkl"
    gpu_id = 3
    task_index = 0  # Dev task only
    
    # Create SEAM parameter sweep analyzer
    seam_sweep = SEAMParameterSweep(gpu_id=gpu_id)
    
    # Load model and data
    seam_sweep.load_keras_model(model_path)
    X_test, y_test = seam_sweep.load_reference_sequences(csv_path, num_sequences=15)
    
    # Load reference attributions (Dev task only)
    reference_attributions = seam_sweep.load_reference_attributions(reference_dir, task_index=task_index)
    
    # Run parameter sweep
    results = seam_sweep.run_full_optimization(
        X_test, reference_attributions, 
        task_index=task_index, 
        output_path=output_path, 
        resume=True
    )
    
    # Print summary
    seam_sweep.print_summary(results)
    
    # Save results
    seam_sweep.save_results(results, output_path)
    
    print(f"\n✓ SEAM parameter sweep complete!")
    print(f"✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()