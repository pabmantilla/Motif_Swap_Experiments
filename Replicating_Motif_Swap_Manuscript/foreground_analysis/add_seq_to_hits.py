#!/usr/bin/env python
"""
Add str_seq column to clean_hits CSVs for low, mid, and high bins.
Creates new dataframes with sequences from Dev_full_data and Hk_full_data.
"""

import pandas as pd

def add_seq_to_hits(hits_dir, seq_dir, hits_prefix="clean_hits", output_suffix="_seq"):
    """
    Add str_seq column to hits CSVs.
    
    Args:
        hits_dir: Directory containing the hits CSVs
        seq_dir: Directory containing the df CSVs with sequences
        hits_prefix: Prefix for the hits files (e.g., 'clean_hits' or 'hits')
        output_suffix: Suffix to add to output files
    """
    bins = ['low', 'mid', 'high']
    
    for bin_name in bins:
        # Read hits CSV
        hits_file = f"{hits_dir}/{bin_name}_{hits_prefix}.csv"
        hits_df = pd.read_csv(hits_file)
        
        # Read df CSV with sequences
        seq_file = f"{seq_dir}/{bin_name}_df.csv"
        seq_df = pd.read_csv(seq_file)
        
        # Create sequence lookup dictionary (row position -> str_seq)
        # sequence_name corresponds to row position (0, 1, 2...), not the 'index' column
        seq_dict = dict(enumerate(seq_df['str_seq']))
        
        # Add str_seq column
        hits_seq = hits_df.copy()
        hits_seq['str_seq'] = hits_seq['sequence_name'].map(seq_dict)
        
        # Get the first 110 unique sequences in this file
        first_110_seqs = hits_seq['sequence_name'].unique()[:110]
        hits_seq = hits_seq[hits_seq['sequence_name'].isin(first_110_seqs)]
        
        # Save
        output_file = f"{hits_dir}/{bin_name}_{hits_prefix}{output_suffix}.csv"
        hits_seq.to_csv(output_file, index=False)
        
        print(f"{bin_name}_{hits_prefix}{output_suffix} shape: {hits_seq.shape}")
        print(f"  Saved to: {output_file}")

# Process Dev
print("=" * 50)
print("Processing Dev...")
print("=" * 50)
add_seq_to_hits(
    hits_dir="TF-Modisco-lite_results/Dev_manual_hits",
    seq_dir="../experimental_library_generation/Binned_libraries/Dev_full_data",
    hits_prefix="clean_hits"
)

# Process Hk
print("\n" + "=" * 50)
print("Processing Hk...")
print("=" * 50)
add_seq_to_hits(
    hits_dir="TF-Modisco-lite_results/Hk_manual_hits",
    seq_dir="../experimental_library_generation/Binned_libraries/Hk_full_data",
    hits_prefix="hits"
)

