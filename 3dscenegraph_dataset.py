#!/usr/bin/env python3
"""
Extract and display all relationship arrays from 3D Scene Graph .npz files
in a single directory, without loading any meshes or images.
"""
import os
import numpy as np
import argparse

def load_relations(npz_path):
    """
    Load all arrays whose keys contain 'relation' (case-insensitive)
    from the 'output' dict inside the given .npz.
    Returns a dict mapping key → numpy array.
    """
    with np.load(npz_path, allow_pickle=True) as archive:
        data = archive['output'].item()
    relations = {k: v for k, v in data.items() if 'relation' in k.lower()}
    return relations

def main(folder_path):
    """
    For each .npz in folder_path, load and print any relation arrays.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")

    npz_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.npz'))
    if not npz_files:
        print(f"No .npz files found in {folder_path}")
        return

    for fname in npz_files:
        npz_path = os.path.join(folder_path, fname)
        relations = load_relations(npz_path)

        print(f"\n=== {fname} ===")
        if not relations:
            print("  (no relation arrays found)")
            continue

        for name, arr in relations.items():
            print(f"  {name}: shape {arr.shape}")
            # Show up to first 5 entries
            sample = arr[:5]
            print(f"    sample:\n{sample}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Print 3D Scene Graph relations from .npz files in a folder"
    )
    parser.add_argument(
        'folder',
        nargs='?',
        default='/home/tristan/3dscenegraph/3DSceneGraph/3dscenegraph_data/verified_graph',
        help="Path to folder containing .npz scene files"
    )
    args = parser.parse_args()
    main(args.folder)
