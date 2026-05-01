# Copyright (c) 2026 Harshil Malhotra. All rights reserved.
# This code is subject to the terms of the Custom Non-Commercial & Attribution License 
# found in the LICENSE.md file in the root directory of this source tree.
# Commercial use requires a paid license.

import os

# Configuration
DATASET_PATH = "Earlobes.v11i.yolov8"
SUBSETS = ["train", "valid", "test"]

# Mapping
# Old index 3 (wholeear) -> New index 0
# Old index 0 (earlobe) -> New index 1
# Remove old index 1 (eye) and 2 (nose)
mapping = {
    "3": "0",  # wholeear
    "0": "1"   # earlobe
}

def process_labels():
    total_files = 0
    total_lines_kept = 0
    total_lines_removed = 0

    for subset in SUBSETS:
        label_dir = os.path.join(DATASET_PATH, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"Directory not found: {label_dir}")
            continue

        print(f"Processing subset: {subset}")
        
        for filename in os.listdir(label_dir):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(label_dir, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                
                old_cls = parts[0]
                if old_cls in mapping:
                    parts[0] = mapping[old_cls]
                    new_lines.append(" ".join(parts) + "\n")
                    total_lines_kept += 1
                else:
                    total_lines_removed += 1

            # Save modified labels
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            total_files += 1

    print("\n" + "="*30)
    print("Label Refinement Complete")
    print(f"Total files processed: {total_files}")
    print(f"Total lines kept: {total_lines_kept}")
    print(f"Total lines removed: {total_lines_removed}")
    print("="*30)

if __name__ == "__main__":
    process_labels()