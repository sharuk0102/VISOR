import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from yolov3_lite import create_yolov3_lite_model

# Constants
IMG_W, IMG_H = 224, 160
NUM_CLASSES = 1

# Load anchors from file
with open("model/anchors.txt", "r") as f:
    anchors_raw = [float(x) for x in f.read().strip().split(",")]
ANCHORS = np.array(anchors_raw).reshape(-1, 2)
# Stride 16 anchors (first 3), Stride 32 anchors (last 3)
ANCHORS_S16 = ANCHORS[:3]
ANCHORS_S32 = ANCHORS[3:]

# Grid sizes
GRID_S16_W, GRID_S16_H = 14, 10
GRID_S32_W, GRID_S32_H = 7, 5

def load_data_split(split_name):
    """Loads images and labels from split folder into memory."""
    img_dir = os.path.join("processed_dataset", split_name, "images")
    lbl_dir = os.path.join("processed_dataset", split_name, "labels")
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    images = []
    labels = []
    
    for img_path in img_files:
        # Load grayscale image and normalize to [0, 1]
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale")
        img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img_arr)
        
        # Load corresponding label file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: cls, cx, cy, w, h (normalized)
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append([cx, cy, w, h])
        labels.append(boxes)
        
    return np.array(images, dtype=np.float32), labels

def calculate_iou_wh(w1, h1, w2, h2):
    """Calculate IoU between box size and anchor size (centered at 0)."""
    inter_w = np.minimum(w1, w2)
    inter_h = np.minimum(h1, h2)
    inter_area = inter_w * inter_h
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area

def encode_targets(labels_list, batch_size):
    """Encodes list of boxes into YOLO stride-16 and stride-32 target tensors."""
    num_samples = len(labels_list)
    
    # Target shapes: [batch, grid_h, grid_w, num_anchors, 4 + 1 + 1]
    # Class index 0 is 'no_helmet', we have 1 class
    y_true_s16 = np.zeros((num_samples, GRID_S16_H, GRID_S16_W, 3, 6), dtype=np.float32)
    y_true_s32 = np.zeros((num_samples, GRID_S32_H, GRID_S32_W, 3, 6), dtype=np.float32)
    
    for idx, boxes in enumerate(labels_list):
        for box in boxes:
            cx, cy, w, h = box
            w_px = w * IMG_W
            h_px = h * IMG_H
            
            # Find best anchor match among all 6 anchors
            ious = [calculate_iou_wh(w_px, h_px, anchor[0], anchor[1]) for anchor in ANCHORS]
            best_anchor_idx = np.argmax(ious)
            
            # Stride 16 Scale (Anchors 1, 2, 3)
            if best_anchor_idx < 3:
                grid_w, grid_h = GRID_S16_W, GRID_S16_H
                anchor_w, anchor_h = ANCHORS_S16[best_anchor_idx]
                
                # Grid cell index
                gx = int(cx * grid_w)
                gy = int(cy * grid_h)
                
                # Ensure coordinates are within grid boundaries
                gx = max(0, min(grid_w - 1, gx))
                gy = max(0, min(grid_h - 1, gy))
                
                # Relative box center in grid cell
                tx = cx * grid_w - gx
                ty = cy * grid_h - gy
                
                # Log scale width and height relative to anchor
                tw = np.log(w_px / anchor_w)
                th = np.log(h_px / anchor_h)
                
                # Anchor offset inside scale
                a_idx = best_anchor_idx
                
                # Write to target
                y_true_s16[idx, gy, gx, a_idx, 0:4] = [tx, ty, tw, th]
                y_true_s16[idx, gy, gx, a_idx, 4] = 1.0  # Objectness confidence
                y_true_s16[idx, gy, gx, a_idx, 5] = 1.0  # Class probability (no_helmet)
                
            # Stride 32 Scale (Anchors 4, 5, 6)
            else:
                grid_w, grid_h = GRID_S32_W, GRID_S32_H
                anchor_w, anchor_h = ANCHORS_S32[best_anchor_idx - 3]
                
                gx = int(cx * grid_w)
                gy = int(cy * grid_h)
                
                gx = max(0, min(grid_w - 1, gx))
                gy = max(0, min(grid_h - 1, gy))
                
                tx = cx * grid_w - gx
                ty = cy * grid_h - gy
                
                tw = np.log(w_px / anchor_w)
                th = np.log(h_px / anchor_h)
                
                a_idx = best_anchor_idx - 3
                
                y_true_s32[idx, gy, gx, a_idx, 0:4] = [tx, ty, tw, th]
                y_true_s32[idx, gy, gx, a_idx, 4] = 1.0
                y_true_s32[idx, gy, gx, a_idx, 5] = 1.0
                
    return y_true_s16, y_true_s32

class YOLOv3LiteTrainer(tf.keras.Model):
    """Wrapper class to implement custom training step with separate YOLO losses."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        
    def call(self, inputs):
        return self.base_model(inputs)
        
    def compute_scale_loss(self, y_true, y_pred_raw):
        # y_true: [batch, grid_h, grid_w, 3, 6]
        # y_pred_raw: [batch, grid_h, grid_w, 18] -> Reshape to [batch, grid_h, grid_w, 3, 6]
        grid_h, grid_w = y_true.shape[1], y_true.shape[2]
        y_pred = tf.reshape(y_pred_raw, (-1, grid_h, grid_w, 3, 6))
        
        # 1. Bounding Box Coordinates Loss (MSE)
        # Only compute loss for grid cells containing target objects (objectness == 1)
        obj_mask = y_true[..., 4:5]  # [batch, grid_h, grid_w, 3, 1]
        
        pred_xy = tf.sigmoid(y_pred[..., 0:2])
        pred_wh = y_pred[..., 2:4]
        
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        
        xy_loss = tf.reduce_sum(obj_mask * tf.square(true_xy - pred_xy))
        wh_loss = tf.reduce_sum(obj_mask * tf.square(true_wh - pred_wh))
        coord_loss = 5.0 * (xy_loss + wh_loss)  # High weight on bounding coordinates
        
        # 2. Objectness Loss (Binary Cross Entropy)
        # Apply weighting: low weight on background cells, high weight on foreground cells
        pred_conf = tf.sigmoid(y_pred[..., 4:5])
        true_conf = y_true[..., 4:5]
        
        # Binary Cross Entropy loss
        conf_loss_all = tf.keras.losses.binary_crossentropy(true_conf, pred_conf, from_logits=False)
        conf_loss_all = tf.expand_dims(conf_loss_all, axis=-1)
        
        # Filter background vs foreground
        noobj_mask = 1.0 - obj_mask
        conf_loss = tf.reduce_sum(obj_mask * conf_loss_all) + 0.5 * tf.reduce_sum(noobj_mask * conf_loss_all)
        
        # 3. Class Classification Loss (Binary Cross Entropy)
        pred_class = tf.sigmoid(y_pred[..., 5:6])
        true_class = y_true[..., 5:6]
        
        class_loss_all = tf.keras.losses.binary_crossentropy(true_class, pred_class, from_logits=False)
        class_loss_all = tf.expand_dims(class_loss_all, axis=-1)
        class_loss = tf.reduce_sum(obj_mask * class_loss_all)
        
        # Return summed losses scaled by batch size
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        return (coord_loss + conf_loss + class_loss) / batch_size
        
    def train_step(self, data):
        x, (y_true_s16, y_true_s32) = data
        
        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            y_pred_s16 = y_pred[:, :2520]
            y_pred_s32 = y_pred[:, 2520:]
            
            # Compute loss for both scales
            loss_s16 = self.compute_scale_loss(y_true_s16, y_pred_s16)
            loss_s32 = self.compute_scale_loss(y_true_s32, y_pred_s32)
            total_loss = loss_s16 + loss_s32
            
        gradients = tape.gradient(total_loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))
        
        return {"loss": total_loss, "loss_s16": loss_s16, "loss_s32": loss_s32}
        
    def test_step(self, data):
        x, (y_true_s16, y_true_s32) = data
        
        y_pred = self.base_model(x, training=False)
        y_pred_s16 = y_pred[:, :2520]
        y_pred_s32 = y_pred[:, 2520:]
        
        loss_s16 = self.compute_scale_loss(y_true_s16, y_pred_s16)
        loss_s32 = self.compute_scale_loss(y_true_s32, y_pred_s32)
        total_loss = loss_s16 + loss_s32
        
        return {"loss": total_loss, "loss_s16": loss_s16, "loss_s32": loss_s32}

def run_training():
    print("Loading datasets...")
    X_train, y_train_raw = load_data_split("train")
    X_val, y_val_raw = load_data_split("val")
    
    print(f"Loaded {X_train.shape[0]} training and {X_val.shape[0]} validation images.")
    
    print("Encoding bounding boxes into grid-cell target tensors...")
    y_train_s16, y_train_s32 = encode_targets(y_train_raw, batch_size=6)
    y_val_s16, y_val_s32 = encode_targets(y_val_raw, batch_size=6)
    
    print("Instantiating YOLOv3-Lite Model...")
    base_model = create_yolov3_lite_model(input_shape=(IMG_H, IMG_W, 1), num_classes=NUM_CLASSES)
    trainer = YOLOv3LiteTrainer(base_model)
    
    # Initialize Adam optimizer with initial learning rate from paper (0.0001)
    # Use weight decay through L2 if needed, otherwise rely on optimizer constraints
    opt = optimizers.Adam(learning_rate=0.0001)
    trainer.compile(optimizer=opt)
    
    print("\nStarting Training (10 Epochs, Batch Size = 6)...")
    
    # Add simple learning rate scheduler and checkpoints
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    
    history = trainer.fit(
        X_train, (y_train_s16, y_train_s32),
        validation_data=(X_val, (y_val_s16, y_val_s32)),
        epochs=10,
        batch_size=6,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed successfully!")
    
    # Save the base model weights and complete model in .h5 format
    os.makedirs("model", exist_ok=True)
    base_model.save("model/yolov3_lite_model.h5")
    print("Saved Keras model to model/yolov3_lite_model.h5")
    
if __name__ == "__main__":
    run_training()
