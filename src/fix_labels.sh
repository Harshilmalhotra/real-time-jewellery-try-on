#!/bin/bash

BASE_DIR="Earlobes.v11i.yolov8"

echo "Starting Label Refinement (Bash)..."
date

# Process train, valid, test directories
for subset in "train" "valid" "test"; do
    echo "Processing $subset labels..."
    LABEL_DIR="$BASE_DIR/$subset/labels"
    
    if [ ! -d "$LABEL_DIR" ]; then
        echo "Directory $LABEL_DIR not found, skipping."
        continue
    fi

    # 1. Remove lines starting with 1 (eye) or 2 (nose)
    # 2. Replace 0 (earlobe) with 1
    # 3. Replace 3 (wholeear) with 0
    # Note: We must be careful about order. Since we're replacing the class ID at the start of the line.
    
    find "$LABEL_DIR" -maxdepth 1 -name "*.txt" -exec sed -i '/^[12] /d; s/^0 /1 /; s/^3 /0 /' {} +
done

echo "Label Refinement Complete!"
date
