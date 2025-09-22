# Motif_Swap_Experiments
## Oracle Model
Uses EvoAug student model as an oracle model for motif context swap experiments.

## SEAM Optimization

`SEAM_Optimization` Finds the optimial parameter set to replicate reference Background, Foreground, and Cluster-specific Background attributions.

- Clustering Technique: [kmeans, tnse+kmeans, umap+kmeans]
  
- Muatgenesis Library Size: [100, 500, 1K, 5K, 10K, 25K, 50K, 75K, 100K]
  
- Number of Clusters: [10, 20, 30, 40, 50]

Generate 15 sequences to run optimization:
```
python Generate_sequence_library.py
```

Make Reference (100K sequence library, 50 Clusters, K-means: 

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
