import os

# Configuration
DATASET_PATH = "Earlobes.v11i.yolov8"
SUBSETS = ["train", "valid", "test"]

# Threshold for geometry-based classification
# Area (width * height) in normalized coordinates (0 to 1)
# Whole ears are typically > 0.02 of image area
# Earlobes are typically < 0.02
AREA_THRESHOLD = 0.02

def restore_labels():
    total_files = 0
    total_ears = 0
    total_earlobes = 0

    for subset in SUBSETS:
        label_dir = os.path.join(DATASET_PATH, subset, "labels")
        if not os.path.exists(label_dir):
            continue

        print(f"Repairing subset: {subset}")
        
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
                
                # We ignore the current class ID and re-classify based on geometry
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
                
                if area > AREA_THRESHOLD:
                    parts[0] = "0"  # wholeear
                    total_ears += 1
                else:
                    parts[0] = "1"  # earlobe
                    total_earlobes += 1
                
                new_lines.append(" ".join(parts) + "\n")

            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            total_files += 1

    print("\n" + "="*30)
    print("Label Restoration Complete")
    print(f"Total files repaired: {total_files}")
    print(f"Detections mapped to wholeear (0): {total_ears}")
    print(f"Detections mapped to earlobe (1): {total_earlobes}")
    print("="*30)

if __name__ == "__main__":
    restore_labels()
