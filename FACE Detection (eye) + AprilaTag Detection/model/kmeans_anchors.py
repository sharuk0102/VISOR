import os
import glob
import numpy as np

def iou(box, clusters):
    """Calculate IoU between a box and centroid clusters.
    
    box: (2,) representing (width, height)
    clusters: (N, 2) representing width/height of N centroids
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    
    union = box_area + cluster_area - intersection
    return intersection / union

def kmeans(boxes, k, max_iters=200):
    """Run K-means clustering using 1 - IoU as distance metric."""
    num_boxes = boxes.shape[0]
    
    # Initialize clusters randomly from the boxes
    indices = np.random.choice(num_boxes, k, replace=False)
    clusters = boxes[indices].copy()
    
    last_assignments = np.zeros(num_boxes)
    
    for _ in range(max_iters):
        # Calculate distances (1 - IoU) for all boxes against all clusters
        distances = np.zeros((num_boxes, k))
        for idx in range(num_boxes):
            distances[idx] = 1 - iou(boxes[idx], clusters)
            
        # Assign boxes to closest cluster
        assignments = np.argmin(distances, axis=1)
        
        if np.array_equal(assignments, last_assignments):
            break
            
        # Update clusters to be the mean of assigned boxes
        for c in range(k):
            assigned_boxes = boxes[assignments == c]
            if len(assigned_boxes) > 0:
                clusters[c] = np.mean(assigned_boxes, axis=0)
                
        last_assignments = assignments.copy()
        
    return clusters

def run_anchor_clustering():
    labels_dir = "processed_dataset/train/labels"
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print(f"Error: No label files found in {labels_dir}!")
        return
        
    # Extracted widths and heights
    box_dims = []
    
    # Image size
    IMG_W, IMG_H = 224, 160
    
    for f_path in label_files:
        with open(f_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO label format: cls_idx cx cy w h
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    
                    # Convert normalized coords back to absolute pixels
                    w_px = w_norm * IMG_W
                    h_px = h_norm * IMG_H
                    box_dims.append([w_px, h_px])
                    
    box_dims = np.array(box_dims)
    print(f"Extracted {len(box_dims)} bounding boxes for anchor clustering.")
    
    # Run K-means to get 6 anchors
    num_anchors = 6
    clusters = kmeans(box_dims, num_anchors)
    
    # Sort clusters by area (width * height) in ascending order
    areas = clusters[:, 0] * clusters[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_clusters = clusters[sorted_indices]
    
    # Print results
    print("\nCalculated Anchors (Width, Height in Pixels):")
    for i, cluster in enumerate(sorted_clusters):
        print(f"  Anchor {i+1}: {cluster[0]:.2f}, {cluster[1]:.2f}")
        
    # Scale 1 (stride 16) gets the smaller 3 anchors, Scale 2 (stride 32) gets the larger 3 anchors
    print("\nScale Allocation:")
    print("  Scale 1 (Stride 16):", [list(np.round(c, 2)) for c in sorted_clusters[:3]])
    print("  Scale 2 (Stride 32):", [list(np.round(c, 2)) for c in sorted_clusters[3:]])
    
    # Save to a format readable by the model script
    os.makedirs("model", exist_ok=True)
    with open("model/anchors.txt", "w") as out:
        anchors_str = ",".join([f"{int(round(c[0]))},{int(round(c[1]))}" for c in sorted_clusters])
        out.write(anchors_str)
    print("\nSaved anchors to model/anchors.txt")

if __name__ == "__main__":
    # Fix random seed for reproducibility
    np.random.seed(42)
    run_anchor_clustering()
