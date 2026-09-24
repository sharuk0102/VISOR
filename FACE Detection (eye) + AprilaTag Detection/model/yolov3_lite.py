import tensorflow as tf
from tensorflow.keras import layers, Model

def relu6(x):
    return tf.nn.relu6(x)

def bottleneck_block(inputs, out_channels, stride, expansion, block_id):
    """MobileNetV2 Bottleneck block."""
    in_channels = inputs.shape[-1]
    x = inputs
    prefix = f"block_{block_id}_"
    
    # 1. Expansion phase
    if expansion != 1:
        expanded_channels = in_channels * expansion
        x = layers.Conv2D(expanded_channels, kernel_size=1, padding='same', use_bias=False, name=prefix + "expand")(x)
        x = layers.BatchNormalization(name=prefix + "expand_BN")(x)
        x = layers.Activation(relu6, name=prefix + "expand_relu")(x)
    else:
        expanded_channels = in_channels

    # 2. Depthwise Convolution phase
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False, name=prefix + "depthwise")(x)
    x = layers.BatchNormalization(name=prefix + "depthwise_BN")(x)
    x = layers.Activation(relu6, name=prefix + "depthwise_relu")(x)

    # 3. Linear Projection phase
    x = layers.Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False, name=prefix + "project")(x)
    x = layers.BatchNormalization(name=prefix + "project_BN")(x)

    # 4. Residual Connection
    if in_channels == out_channels and stride == 1:
        x = layers.Add(name=prefix + "add")([inputs, x])
        
    return x

def build_backbone(input_tensor):
    """Builds a lightweight MobileNetV2-Lite backbone."""
    # Step 1: Entry Conv Layer (stride 2)
    # Output: 80x112x16
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same', use_bias=False, name="conv_init")(input_tensor)
    x = layers.BatchNormalization(name="conv_init_BN")(x)
    x = layers.Activation(relu6, name="conv_init_relu")(x)
    
    # Bottleneck 1: 80x112x16, stride 1, exp 1
    x = bottleneck_block(x, out_channels=16, stride=1, expansion=1, block_id=1)
    
    # Bottleneck 2: 40x56x24, stride 2, exp 6
    x = bottleneck_block(x, out_channels=24, stride=2, expansion=6, block_id=2)
    # Bottleneck 3: 40x56x24, stride 1, exp 6
    x = bottleneck_block(x, out_channels=24, stride=1, expansion=6, block_id=3)
    
    # Bottleneck 4: 20x28x32, stride 2, exp 6
    x = bottleneck_block(x, out_channels=32, stride=2, expansion=6, block_id=4)
    # Bottleneck 5: 20x28x32, stride 1, exp 6
    x = bottleneck_block(x, out_channels=32, stride=1, expansion=6, block_id=5)
    
    # Bottleneck 6: 10x14x64, stride 2, exp 6
    # This is the stride-16 scale output
    x_stride_16 = bottleneck_block(x, out_channels=64, stride=2, expansion=6, block_id=6)
    # Bottleneck 7: 10x14x64, stride 1, exp 6
    x_stride_16 = bottleneck_block(x_stride_16, out_channels=64, stride=1, expansion=6, block_id=7)
    
    # Bottleneck 8: 5x7x128, stride 2, exp 6
    # This is the stride-32 scale output
    x_stride_32 = bottleneck_block(x_stride_16, out_channels=128, stride=2, expansion=6, block_id=8)
    # Bottleneck 9: 5x7x128, stride 1, exp 6
    x_stride_32 = bottleneck_block(x_stride_32, out_channels=128, stride=1, expansion=6, block_id=9)
    
    return x_stride_16, x_stride_32

def create_yolov3_lite_model(input_shape=(160, 224, 1), num_classes=1):
    """Creates a 2-scale YOLOv3-Lite detection model."""
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # 1. Feature Extraction via Backbone
    feat_s16, feat_s32 = build_backbone(inputs)
    
    # --- SCALE 2 (Stride 32 - 5x7 grid) ---
    # Reduce channels
    s32_conv = layers.Conv2D(64, kernel_size=1, padding='same', use_bias=False, name="s32_conv_1x1")(feat_s32)
    s32_conv = layers.BatchNormalization(name="s32_conv_1x1_BN")(s32_conv)
    s32_conv = layers.Activation(relu6, name="s32_conv_1x1_relu")(s32_conv)
    
    s32_feat = layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False, name="s32_conv_3x3")(s32_conv)
    s32_feat = layers.BatchNormalization(name="s32_conv_3x3_BN")(s32_feat)
    s32_feat = layers.Activation(relu6, name="s32_conv_3x3_relu")(s32_feat)
    
    # YOLO prediction output layer for Scale 2 (Stride 32)
    # 3 anchors * (4 coords + 1 objectness + num_classes)
    num_filters = 3 * (4 + 1 + num_classes)
    output_s32 = layers.Conv2D(num_filters, kernel_size=1, padding='same', name="yolo_output_stride_32")(s32_feat)
    
    # --- SCALE 1 (Stride 16 - 10x14 grid) ---
    # Upsample Stride 32 features to concatenate with Stride 16 features
    s32_upsample = layers.Conv2D(64, kernel_size=1, padding='same', use_bias=False, name="s32_to_upsample_conv")(s32_conv)
    s32_upsample = layers.BatchNormalization(name="s32_to_upsample_conv_BN")(s32_upsample)
    s32_upsample = layers.Activation(relu6, name="s32_to_upsample_conv_relu")(s32_upsample)
    
    # 2x nearest neighbor upsampling (5x7 -> 10x14)
    upsampled = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="s32_upsampling")(s32_upsample)
    
    # Concatenate upsampled features with Stride 16 features
    s16_concat = layers.Concatenate(name="s16_concat")([upsampled, feat_s16])
    
    s16_conv = layers.Conv2D(64, kernel_size=1, padding='same', use_bias=False, name="s16_conv_1x1")(s16_concat)
    s16_conv = layers.BatchNormalization(name="s16_conv_1x1_BN")(s16_conv)
    s16_conv = layers.Activation(relu6, name="s16_conv_1x1_relu")(s16_conv)
    
    s16_feat = layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False, name="s16_conv_3x3")(s16_conv)
    s16_feat = layers.BatchNormalization(name="s16_conv_3x3_BN")(s16_feat)
    s16_feat = layers.Activation(relu6, name="s16_conv_3x3_relu")(s16_feat)
    
    # YOLO prediction output layer for Scale 1 (Stride 16)
    output_s16 = layers.Conv2D(num_filters, kernel_size=1, padding='same', name="yolo_output_stride_16")(s16_feat)
    
    # Flatten both outputs to ensure a single concatenated output layer for K210 KPU compatibility
    flat_s16 = layers.Reshape((2520,), name="s16_reshape")(output_s16)
    flat_s32 = layers.Reshape((630,), name="s32_reshape")(output_s32)
    output_concat = layers.Concatenate(axis=-1, name="yolo_output_concat")([flat_s16, flat_s32])
    
    # Return model with a single concatenated output
    model = Model(inputs=inputs, outputs=output_concat, name="YOLOv3_Lite_K210")
    return model

if __name__ == "__main__":
    model = create_yolov3_lite_model()
    model.summary()
    print("\nYOLOv3-Lite Model successfully created!")
    print(f"Total outputs: {len(model.outputs)}")
    print(f"Output shape: {model.outputs[0].shape}")
