# Motif_Swap_Experiments
## Oracle Model
Uses EvoAug student model as an oracle model for motif context swap experiments.

## SEAM Optimization with Integrated Gradients

`SEAM_Optimization` Finds the optimial parameter set to replicate reference Background, Foreground, and Cluster-specific Background attributions.

- Clustering Technique: [kmeans, tnse+kmeans, umap+kmeans]
  
- Muatgenesis Library Size: [100, 500, 1K, 5K, 10K, 25K, 50K, 75K, 100K]
  
- Number of Clusters: [10, 20, 30, 40, 50]

Generate 15 sequences to run optimization:
```
python Generate_sequence_library.py
```

Make Reference (100K sequence library, 50 Clusters, Hierarchical clustering): 

```
python Generate_SEAM_reference_attributions.py <start> <end> <task>
```

Sweep through parameters: 
```
python Parameter_sweep_comparison.py
```

Plot results:
```
python plot_parameter_sweep_results.ipynb
```

## SEAM Optimization with DeepLIFT/SHAP 

`SEAM_Optimization_DeepLIFT` Finds the optimial parameter set to replicate reference Background, Foreground, and Cluster-specific Background attributions.

- Clustering Technique: [kmeans]
  
- Muatgenesis Library Size: [100, 500, 1K, 5K, 10K, 50K, 75K, 100K]
  
- Number of Clusters: [10, 20, 30, 40, 50, 75]


Make Reference (100K sequence library, 50 Clusters, Hierarchical clustering): 

```
python DeepLIFT_references.py
```

Sweep through parameters: 
```
python DeepLIFT_kmeans_param_sweep.py
```

Plot results with the `parameter_comparison.ipynb` notebook


