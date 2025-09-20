#!/usr/bin/env python3
"""
SEAM Optimization Analysis

This script replicates the SEAM optimization analysis from the paper:
- Library size optimization (100 to 100,000 sequences)
- Clustering method comparison (5 different methods)
- Number of clusters optimization (10 to 50 clusters)
- Reference: Background attribution map with 100,000 sequences + hierarchical clustering
- Evaluation: Spearman correlation between reference and alternative hyperparameters
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
# UMAP import will be handled conditionally in functions that need it
UMAP_AVAILABLE = False
import json
import pickle

class SEAMOptimizationAnalysis:
    """
    SEAM optimization analysis following the paper's methodology.
    """
    
    def __init__(self, gpu_id: int = 3):
        self.gpu_id = gpu_id
        self.keras_model = None
        self.alphabet = ['A', 'C', 'G', 'T']
        self.reference_attributions = {}  # Store reference attributions
        self.results = {}  # Store optimization results
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
        
        # Apply minimal SEAM format patch
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
        
        print("✓ SEAM format patch applied successfully")
    
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
        
        return X_test, y_test
    
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
        compiler = Compiler(
            x=x_mut,
            y=y_mut,
            x_ref=x_ref,
            y_bg=None,
            alphabet=self.alphabet,
            gpu=self.gpu_id
        )
        
        mave_df = compiler.compile()
        
        # SEAM API - Compute attribution maps (uses our patched intgrad)
        attributer = Attributer(
            self.keras_model,
            method='intgrad',
            task_index=task_index,
            compress_fun=lambda x: tf.reduce_sum(x)  # Explicit scalar reduction for gradient computation
        )
        
        # Transpose data from (batch, alphabet, seq_length) to (batch, seq_length, alphabet) for SEAM
        x_mut_for_seam = np.transpose(x_mut, (0, 2, 1))  # (batch, alphabet, seq_length) -> (batch, seq_length, alphabet)
        x_ref_for_seam = np.transpose(x_ref, (0, 2, 1))  # (batch, alphabet, seq_length) -> (batch, seq_length, alphabet)
        
        # Compute attribution maps
        attributions = attributer.compute(
            x=x_mut_for_seam,
            x_ref=x_ref_for_seam,
            save_window=None,
            batch_size=32,
            gpu=self.gpu_id,
            num_steps=num_steps
        )
        
        return attributions, mave_df
    
    def cluster_attributions(self, attributions: np.ndarray, 
                           method: str = 'hierarchical',
                           n_clusters: int = 50,
                           embedding_method: str = None):
        """
        Cluster attribution maps using SEAM's GPU-optimized methods.
        
        Methods:
        - hierarchical: Ward linkage hierarchical clustering (GPU-optimized)
        - leiden: Leiden clustering (if available)
        - kmeans: K-means on original feature space
        - kmeans_umap: K-means on UMAP embedding
        - kmeans_tsne: K-means on t-SNE embedding
        """
        from seam import Clusterer
        
        print(f"  Clustering with {method} (k={n_clusters})...")
        
        if method == 'hierarchical':
            # Use SEAM's GPU-optimized hierarchical clustering
            clusterer = Clusterer(attributions, gpu=True)
            
            # Perform hierarchical clustering
            linkage_matrix = clusterer.cluster(method='hierarchical', link_method='ward')
            
            # Get cluster labels
            cluster_labels, _ = clusterer.get_cluster_labels(
                linkage_matrix, criterion='maxclust', n_clusters=n_clusters
            )
            
        elif method == 'kmeans':
            # Use SEAM's GPU-optimized K-means clustering
            clusterer = Clusterer(attributions, gpu=True)
            cluster_labels = clusterer.cluster(method='kmeans', n_clusters=n_clusters, random_state=42)
            
        elif method == 'kmeans_umap':
            if not UMAP_AVAILABLE:
                raise ValueError("UMAP not available. Install with: pip install umap-learn")
            
            # K-means on UMAP embedding
            attributions_flat = attributions.reshape(attributions.shape[0], -1)
            
            # UMAP embedding
            reducer = umap.UMAP(n_components=min(50, attributions_flat.shape[1]), 
                              random_state=42)
            embedding = reducer.fit_transform(attributions_flat)
            
            # K-means on embedding
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding)
            
        elif method == 'kmeans_tsne':
            # K-means on t-SNE embedding
            attributions_flat = attributions.reshape(attributions.shape[0], -1)
            
            # t-SNE embedding
            tsne = TSNE(n_components=min(50, attributions_flat.shape[1]), 
                       random_state=42, perplexity=min(30, attributions_flat.shape[0]-1))
            embedding = tsne.fit_transform(attributions_flat)
            
            # K-means on embedding
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding)
            
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
    
    def generate_reference_attribution(self, reference_sequence: np.ndarray,
                                     task_index: int = 0,
                                     num_sequences: int = 100000,
                                     n_clusters: int = 50):
        """
        Generate reference attribution using 100,000 sequences + hierarchical clustering.
        This is the reference against which all other configurations are compared.
        """
        print(f"\n--- Generating Reference Attribution ---")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"Sequences: {num_sequences}")
        print(f"Clustering: Hierarchical (Ward linkage)")
        print(f"Clusters: {n_clusters}")
        
        start_time = time.time()
        
        # Generate mutagenesis library
        x_mut, y_mut = self.generate_mutagenesis_library(
            reference_sequence, num_sequences, task_index=task_index
        )
        
        # Compute attributions
        x_ref = np.expand_dims(reference_sequence, 0)
        attributions, mave_df = self.compute_attributions(x_mut, y_mut, x_ref, task_index)
        
        # Cluster attributions (hierarchical)
        cluster_labels = self.cluster_attributions(
            attributions, method='hierarchical', n_clusters=n_clusters
        )
        
        # Generate background attribution
        background_attribution, background_cluster = self.generate_background_attribution(
            attributions, cluster_labels
        )
        
        total_time = time.time() - start_time
        
        print(f"✓ Reference attribution generated in {total_time:.2f}s")
        print(f"✓ Background cluster: {background_cluster}")
        print(f"✓ Attribution range: {background_attribution.min():.6f} to {background_attribution.max():.6f}")
        
        return {
            'background_attribution': background_attribution,
            'attributions': attributions,
            'cluster_labels': cluster_labels,
            'background_cluster': background_cluster,
            'total_time': total_time,
            'num_sequences': num_sequences,
            'n_clusters': n_clusters,
            'method': 'hierarchical'
        }
    
    def evaluate_configuration(self, reference_sequence: np.ndarray,
                             task_index: int,
                             num_sequences: int,
                             clustering_method: str,
                             n_clusters: int,
                             reference_attribution: np.ndarray):
        """
        Evaluate a specific configuration against the reference.
        Returns Spearman correlation and runtime.
        """
        print(f"  Evaluating: {num_sequences} seqs, {clustering_method}, {n_clusters} clusters")
        
        start_time = time.time()
        
        try:
            # Generate mutagenesis library
            x_mut, y_mut = self.generate_mutagenesis_library(
                reference_sequence, num_sequences, task_index=task_index
            )
            
            # Compute attributions
            x_ref = np.expand_dims(reference_sequence, 0)
            attributions, mave_df = self.compute_attributions(x_mut, x_ref, task_index)
            
            # Cluster attributions
            cluster_labels = self.cluster_attributions(
                attributions, method=clustering_method, n_clusters=n_clusters
            )
            
            # Generate background attribution
            background_attribution, _ = self.generate_background_attribution(
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
                'success': True
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"    ✗ Error: {e}")
            return {
                'correlation': np.nan,
                'runtime': total_time,
                'success': False,
                'error': str(e)
            }
    
    def optimize_library_size(self, reference_sequence: np.ndarray, task_index: int):
        """
        C.1 Optimizing number of sequences
        Test library sizes: 100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000
        """
        print(f"\n{'='*60}")
        print(f"LIBRARY SIZE OPTIMIZATION")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"{'='*60}")
        
        # Generate reference attribution (100,000 sequences + hierarchical)
        reference_result = self.generate_reference_attribution(
            reference_sequence, task_index, num_sequences=100000, n_clusters=50
        )
        reference_attribution = reference_result['background_attribution']
        
        # Test different library sizes
        library_sizes = [100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000]
        results = []
        
        for num_sequences in library_sizes:
            print(f"\n--- Testing {num_sequences} sequences ---")
            
            result = self.evaluate_configuration(
                reference_sequence, task_index, num_sequences,
                'hierarchical', 50, reference_attribution
            )
            
            results.append({
                'num_sequences': num_sequences,
                'clustering_method': 'hierarchical',
                'n_clusters': 50,
                'correlation': result['correlation'],
                'runtime': result['runtime'],
                'success': result['success']
            })
        
        return results, reference_result
    
    def optimize_clustering_method(self, reference_sequence: np.ndarray, task_index: int):
        """
        C.2 Optimizing cluster analysis
        Test methods: hierarchical, leiden, kmeans, kmeans_umap, kmeans_tsne
        """
        print(f"\n{'='*60}")
        print(f"CLUSTERING METHOD OPTIMIZATION")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"{'='*60}")
        
        # Generate reference attribution (100,000 sequences + hierarchical)
        reference_result = self.generate_reference_attribution(
            reference_sequence, task_index, num_sequences=100000, n_clusters=50
        )
        reference_attribution = reference_result['background_attribution']
        
        # Test different clustering methods
        clustering_methods = ['hierarchical', 'kmeans', 'kmeans_tsne']
        if UMAP_AVAILABLE:
            clustering_methods.append('kmeans_umap')
        results = []
        
        for method in clustering_methods:
            print(f"\n--- Testing {method} ---")
            
            result = self.evaluate_configuration(
                reference_sequence, task_index, 10000,  # Use 10,000 sequences (optimized)
                method, 50, reference_attribution
            )
            
            results.append({
                'num_sequences': 10000,
                'clustering_method': method,
                'n_clusters': 50,
                'correlation': result['correlation'],
                'runtime': result['runtime'],
                'success': result['success']
            })
        
        return results, reference_result
    
    def optimize_number_of_clusters(self, reference_sequence: np.ndarray, task_index: int):
        """
        C.3 Optimizing number of clusters
        Test cluster numbers: 10, 20, 30, 40, 50
        """
        print(f"\n{'='*60}")
        print(f"NUMBER OF CLUSTERS OPTIMIZATION")
        print(f"Task: {task_index} ({'Dev' if task_index == 0 else 'Hk'})")
        print(f"{'='*60}")
        
        # Generate reference attribution (100,000 sequences + hierarchical)
        reference_result = self.generate_reference_attribution(
            reference_sequence, task_index, num_sequences=100000, n_clusters=50
        )
        reference_attribution = reference_result['background_attribution']
        
        # Test different numbers of clusters
        n_clusters_list = [10, 20, 30, 40, 50]
        results = []
        
        for n_clusters in n_clusters_list:
            print(f"\n--- Testing {n_clusters} clusters ---")
            
            result = self.evaluate_configuration(
                reference_sequence, task_index, 10000,  # Use 10,000 sequences (optimized)
                'kmeans', n_clusters, reference_attribution  # Use kmeans (optimized)
            )
            
            results.append({
                'num_sequences': 10000,
                'clustering_method': 'kmeans',
                'n_clusters': n_clusters,
                'correlation': result['correlation'],
                'runtime': result['runtime'],
                'success': result['success']
            })
        
        return results, reference_result
    
    def run_full_optimization(self, sequences: np.ndarray, task_indices: List[int] = [0, 1], 
                             output_path: str = None, resume: bool = True):
        """
        Run the complete SEAM optimization analysis.
        
        This analysis uses:
        - Our new Keras model with both Dev and Hk tasks
        - Patched integrated gradients with proper task-aware attributions
        - SEAM API for clustering and background generation
        - SQUID API for mutagenesis library generation
        """
        print(f"\n{'='*60}")
        print(f"SEAM OPTIMIZATION ANALYSIS")
        print(f"Using: New Keras model + Patched integrated gradients")
        print(f"Sequences: {len(sequences)}")
        print(f"Tasks: {task_indices}")
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
        
        for task_idx in task_indices:
            print(f"\n{'='*60}")
            print(f"TASK {task_idx} ({'Dev' if task_idx == 0 else 'Hk'})")
            print(f"{'='*60}")
            
            task_results = {}
            
            # Generate reference attributions for ALL 15 sequences
            print(f"\n--- Generating Reference Attributions for All {len(sequences)} Sequences ---")
            reference_attributions = {}
            
            # Only process problematic sequences that need regeneration
            problematic_sequences = [0, 1, 4, 5, 6, 7, 8]  # Sequences that need regeneration
            
            for seq_idx, reference_sequence in enumerate(sequences):
                # Skip sequences that don't need regeneration
                if seq_idx not in problematic_sequences:
                    print(f"\nSequence {seq_idx + 1}/{len(sequences)} - SKIPPING (not problematic)")
                    continue
                    
                print(f"\nSequence {seq_idx + 1}/{len(sequences)} - PROBLEMATIC (regenerating)")
                
                # Check if this sequence's problematic reference attribution already exists
                temp_save_path = f"temp_reference_attributions_task{task_idx}_seq{seq_idx}_problematic.pkl"
                if os.path.exists(temp_save_path):
                    print(f"  ✓ Found existing PROBLEMATIC reference attribution for sequence {seq_idx}")
                    try:
                        with open(temp_save_path, 'rb') as f:
                            reference_attributions[seq_idx] = pickle.load(f)
                        print(f"  ✓ Loaded existing PROBLEMATIC reference attribution for sequence {seq_idx}")
                        continue
                    except Exception as e:
                        print(f"  ⚠ Error loading existing file: {e}")
                        print(f"  → Recomputing sequence {seq_idx}...")
                
                # Compute reference attribution if not found or loading failed
                reference_attributions[seq_idx] = self.generate_reference_attribution(
                    reference_sequence, task_idx, num_sequences=100000, n_clusters=50
                )
                
                # Save after each sequence to avoid losing progress
                with open(temp_save_path, 'wb') as f:
                    pickle.dump(reference_attributions[seq_idx], f)
                print(f"✓ Saved reference attribution for sequence {seq_idx} to {temp_save_path}")
            
            # Use first sequence for optimization analysis
            reference_sequence = sequences[0]
            
            # C.1: Library size optimization
            print(f"\n--- C.1: Library Size Optimization ---")
            library_results, reference_result = self.optimize_library_size(
                reference_sequence, task_idx
            )
            task_results['library_size'] = library_results
            task_results['reference'] = reference_result
            
            # C.2: Clustering method optimization
            print(f"\n--- C.2: Clustering Method Optimization ---")
            clustering_results, _ = self.optimize_clustering_method(
                reference_sequence, task_idx
            )
            task_results['clustering_method'] = clustering_results
            
            # C.3: Number of clusters optimization
            print(f"\n--- C.3: Number of Clusters Optimization ---")
            clusters_results, _ = self.optimize_number_of_clusters(
                reference_sequence, task_idx
            )
            task_results['number_of_clusters'] = clusters_results
            
            all_results[f'task_{task_idx}'] = task_results
            all_results[f'task_{task_idx}_reference_attributions'] = reference_attributions
        
        return all_results
    
    def save_results(self, results: Dict, output_path: str):
        """Save optimization results"""
        print(f"\nSaving results to: {output_path}")
        
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
        
        # Also save reference attributions separately for easy access
        for task_idx in [0, 1]:
            ref_key = f'task_{task_idx}_reference_attributions'
            if ref_key in results:
                ref_output_path = output_path.replace('.pkl', f'_task{task_idx}_reference_attributions.pkl')
                with open(ref_output_path, 'wb') as f:
                    pickle.dump(results[ref_key], f)
                print(f"✓ Reference attributions for task {task_idx} saved to: {ref_output_path}")
    
    def print_summary(self, results: Dict):
        """Print optimization summary"""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        for task_key, task_data in results.items():
            task_idx = int(task_key.split('_')[1])
            print(f"\nTask {task_idx} ({'Dev' if task_idx == 0 else 'Hk'}):")
            
            # Library size results
            library_results = task_data['library_size']
            best_library = max([r for r in library_results if r['success']], 
                             key=lambda x: x['correlation'])
            print(f"  Best library size: {best_library['num_sequences']} "
                  f"(correlation: {best_library['correlation']:.6f})")
            
            # Clustering method results
            clustering_results = task_data['clustering_method']
            best_clustering = max([r for r in clustering_results if r['success']], 
                                key=lambda x: x['correlation'])
            print(f"  Best clustering: {best_clustering['clustering_method']} "
                  f"(correlation: {best_clustering['correlation']:.6f})")
            
            # Number of clusters results
            clusters_results = task_data['number_of_clusters']
            best_clusters = max([r for r in clusters_results if r['success']], 
                              key=lambda x: x['correlation'])
            print(f"  Best clusters: {best_clusters['n_clusters']} "
                  f"(correlation: {best_clusters['correlation']:.6f})")


def main():
    """Main function to run SEAM optimization analysis"""
    
    # Configuration
    model_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/EvoAug_keras_model"
    csv_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/evoaug_15seq_results_updated.csv"
    output_path = "/grid/wsbs/home_norepl/pmantill/Motif_swap_experiment/seam_optimization_results.pkl"
    gpu_id = 3
    
    # Create SEAM optimization analyzer
    seam_opt = SEAMOptimizationAnalysis(gpu_id=gpu_id)
    
    # Load model and data
    seam_opt.load_keras_model(model_path)
    X_test, y_test = seam_opt.load_reference_sequences(csv_path, num_sequences=15)
    
    # Run full optimization analysis
    results = seam_opt.run_full_optimization(X_test, task_indices=[0, 1], 
                                            output_path=output_path, resume=True)
    
    # Print summary
    seam_opt.print_summary(results)
    
    # Save results
    seam_opt.save_results(results, output_path)
    
    print(f"\n✓ SEAM optimization analysis complete!")
    print(f"✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()