import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy import ndimage


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# SwinTransformer model architecture
class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module (CBAM)"""
    
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM3D, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        
        return x_channel * spatial_weights

class SwinTransformerBlock3D(nn.Module):
    
    def __init__(self, dim, num_heads, window_size=(4, 4, 4), shift_size=0):
        super(SwinTransformerBlock3D, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=False,
            dropout=0.1
        )
        
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
    def window_partition(self, x, window_size):
       
        B, C, D, H, W = x.shape
        
        # Pad if necessary
        pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D_padded, H_padded, W_padded = D + pad_d, H + pad_h, W + pad_w
        else:
            D_padded, H_padded, W_padded = D, H, W
        
        # Window partition
        x = x.view(B, C, 
                  D_padded // window_size[0], window_size[0],
                  H_padded // window_size[1], window_size[1], 
                  W_padded // window_size[2], window_size[2])
        windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        windows = windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)
        return windows, (D_padded, H_padded, W_padded)
    
    def window_reverse(self, windows, window_size, D, H, W, original_dhw):
       
        D_orig, H_orig, W_orig = original_dhw
        B = int(windows.shape[0] / (D * H * W / (window_size[0] * window_size[1] * window_size[2])))
        
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                        window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, -1, D, H, W)
        
        # Remove padding
        if D != D_orig or H != H_orig or W != W_orig:
            x = x[:, :, :D_orig, :H_orig, :W_orig]
        
        return x
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        original_dhw = (D, H, W)
        
        #layer norm
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        #shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(2, 3, 4))
        else:
            shifted_x = x_norm
            
        # Window partition
        windows, padded_dhw = self.window_partition(shifted_x, self.window_size)
        
        # Self-attention
        windows_attended, _ = self.attention(windows, windows, windows)
        
        # Reverse window partition
        D_padded, H_padded, W_padded = padded_dhw
        if self.shift_size > 0:
            x_attended = self.window_reverse(windows_attended, self.window_size, D_padded, H_padded, W_padded, original_dhw)
            # Reverse shift
            x_attended = torch.roll(x_attended, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(2, 3, 4))
        else:
            x_attended = self.window_reverse(windows_attended, self.window_size, D_padded, H_padded, W_padded, original_dhw)
        
        #residual connection
        x = x + x_attended
        
        
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(B * D * H * W, C)
        x_mlp = self.mlp(self.norm2(x_flat))
        x = x_flat + x_mlp
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        return x

class SwinTransformerBottleneck3D(nn.Module):
    
    def __init__(self, in_channels, embed_dim, num_heads, depths=[2, 2], window_size=(4, 4, 4)):
        super(SwinTransformerBottleneck3D, self).__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        # input to embedding dimension
        self.input_projection = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        
         
        self.stages = nn.ModuleList()
        current_dim = embed_dim
        
        for i, depth in enumerate(depths):
            stage_blocks = nn.ModuleList()
            for j in range(depth):
                
                shift_size = 0 if j % 2 == 0 else window_size[0] // 2
                stage_blocks.append(
                    SwinTransformerBlock3D(
                        dim=current_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size
                    )
                )
            self.stages.append(stage_blocks)
        
        
        self.output_projection = nn.Conv3d(embed_dim, in_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        batch_size, channels, depth, height, width = x.size()
        
        
        x_proj = self.input_projection(x)  # (batch, embed_dim, depth, height, width)
        
        # Swin Transformer stages
        for stage in self.stages:
            for block in stage:
                x_proj = block(x_proj)
        
        # Project back to original channels
        x_out = self.output_projection(x_proj)
        
        return x_out

class DoubleConv3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_cbam=False):
        super(DoubleConv3D, self).__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM3D(out_channels)
        
    def forward(self, x):
        x = self.double_conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        return x

class Down3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(Down3D, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, use_cbam=use_cbam)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(Up3D, self).__init__()
        
        # Upsampling
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Double convolution with CBAM
        self.conv = DoubleConv3D(in_channels, out_channels, use_cbam=use_cbam)
        
    def forward(self, x1, x2):
        # Upsample
        x1 = self.up(x1)
        
        # Handle dimension mismatches
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        
        # Double convolution with CBAM
        return self.conv(x)

class SwinTransformerUNet3D(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512],
                 swin_embed_dim=512, swin_heads=8, swin_depths=[2, 2], window_size=(4, 4, 4)):
        super(SwinTransformerUNet3D, self).__init__()
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, features[0], use_cbam=True)
        self.down1 = Down3D(features[0], features[1], use_cbam=True)
        self.down2 = Down3D(features[1], features[2], use_cbam=True)
        self.down3 = Down3D(features[2], features[3], use_cbam=True)
        self.down4 = Down3D(features[3], features[4], use_cbam=True)
        
        # Bottleneck with Swin Transformer
        self.bottleneck_conv = DoubleConv3D(features[4], features[4], use_cbam=False)
        self.swin_transformer = SwinTransformerBottleneck3D(
            in_channels=features[4],
            embed_dim=swin_embed_dim,
            num_heads=swin_heads,
            depths=swin_depths,
            window_size=window_size
        )
        self.bottleneck_out = DoubleConv3D(features[4], features[4], use_cbam=True)
        
        # Decoder
        self.up1 = Up3D(features[4], features[3], use_cbam=True)  # 512 -> 256
        self.up2 = Up3D(features[3], features[2], use_cbam=True)  # 256 -> 128
        self.up3 = Up3D(features[2], features[1], use_cbam=True)  # 128 -> 64
        self.up4 = Up3D(features[1], features[0], use_cbam=True)  # 64 -> 32
        
        # Output - NO SIGMOID here (sigmoid will be applied in loss function during training)
        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder with CBAM
        x1 = self.inc(x)      # 32
        x2 = self.down1(x1)   # 64
        x3 = self.down2(x2)   # 128
        x4 = self.down3(x3)   # 256
        x5 = self.down4(x4)   # 512
        
        # Bottleneck with Swin Transformer
        x6 = self.bottleneck_conv(x5)  # 512 features
        x6 = self.swin_transformer(x6)  # Apply Swin Transformer
        x6 = self.bottleneck_out(x6)
        
        # Decoder with CBAM
        x = self.up1(x6, x4)  # 512 + 256 -> 256
        x = self.up2(x, x3)   # 256 + 128 -> 128
        x = self.up3(x, x2)   # 128 + 64 -> 64
        x = self.up4(x, x1)   # 64 + 32 -> 32
        
        
        return self.outc(x)


def load_model(model_path, model_class=SwinTransformerUNet3D):
    """Load the pre-trained SwinTransformer weights"""
    model = model_class(in_channels=1, out_channels=1)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    print(f"Model weight successfully loaded from {model_path}")
    return model

def zero_std_guard_normalize(image):
    
    mean_val = np.mean(image)
    std_val = np.std(image)
    
    # Zero-Std Guard
    epsilon = 1e-8
    if std_val < epsilon:
        normalized = np.zeros_like(image)
    else:
        normalized = (image - mean_val) / std_val
        
    return normalized

def extract_central_patch(image_data, patch_size=(128, 128, 128)):
    """
    Extract central patch from the image
    """
    d, h, w = image_data.shape
    patch_d, patch_h, patch_w = patch_size
    
    # Calculate central coordinates
    center_z = d // 2
    center_y = h // 2
    center_x = w // 2
    
    start_z = max(0, center_z - patch_d // 2)
    end_z = min(d, start_z + patch_d)
    start_y = max(0, center_y - patch_h // 2)
    end_y = min(h, start_y + patch_h)
    start_x = max(0, center_x - patch_w // 2)
    end_x = min(w, start_x + patch_w)
    
    # Extract central patch
    central_patch = image_data[start_z:end_z, start_y:end_y, start_x:end_x]
    
    if central_patch.shape != patch_size:
        pad_z = patch_d - central_patch.shape[0]
        pad_y = patch_h - central_patch.shape[1]
        pad_x = patch_w - central_patch.shape[2]
        
        pad_before_z = pad_z // 2
        pad_after_z = pad_z - pad_before_z
        pad_before_y = pad_y // 2
        pad_after_y = pad_y - pad_before_y
        pad_before_x = pad_x // 2
        pad_after_x = pad_x - pad_before_x
        
        central_patch = np.pad(central_patch, 
                             ((pad_before_z, pad_after_z), 
                              (pad_before_y, pad_after_y), 
                              (pad_before_x, pad_after_x)), 
                             mode='constant', constant_values=0)
    
    return central_patch, (start_z, end_z, start_y, end_y, start_x, end_x)

def predict_central_patch(model, image_path, patch_size=(128, 128, 128), threshold=0.5):
    
    # Load image
    img = nib.load(image_path)
    image_data = img.get_fdata()
    affine = img.affine
    original_shape = image_data.shape
    
    # Apply Zero-Std Guard normalization (from transformer pipeline)
    image_data = zero_std_guard_normalize(image_data)
    
    # Extract central patch
    central_patch, patch_coords = extract_central_patch(image_data, patch_size)
    start_z, end_z, start_y, end_y, start_x, end_x = patch_coords
    
    # Convert to tensor
    patch_tensor = torch.FloatTensor(central_patch).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        patch_tensor = patch_tensor.cuda()
    
    # Predict on central patch only
    with torch.no_grad():
        output_logits = model(patch_tensor)
        # Apply sigmoid
        output_patch = torch.sigmoid(output_logits)
    
    # Convert to numpy
    pred_patch = output_patch.squeeze().cpu().numpy()
    
    # Create full output mask
    output_mask = np.zeros(original_shape, dtype=np.float32)
    
    # Get the actual predicted region
    pred_height = min(pred_patch.shape[0], end_z - start_z)
    pred_width = min(pred_patch.shape[1], end_y - start_y)
    pred_depth = min(pred_patch.shape[2], end_x - start_x)
    
    
    output_mask[start_z:start_z+pred_height, 
                start_y:start_y+pred_width, 
                start_x:start_x+pred_depth] = pred_patch[:pred_height, :pred_width, :pred_depth]
    
    #threshold to create binary mask
    output_binary = (output_mask > threshold).astype(np.uint8)
    
    #NIfTI file
    output_nifti = nib.Nifti1Image(output_binary, affine)
    
    return output_nifti, output_mask, img, output_binary

def predict_direct(model, image_data, threshold=0.5):
    
    #normalization
    image_data = zero_std_guard_normalize(image_data)
    
    #Convert to tensor
    image_tensor = torch.FloatTensor(image_data).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    #Predict
    with torch.no_grad():
        output_logits = model(image_tensor)
        # Apply sigmoid here for inference
        pred_mask = torch.sigmoid(output_logits).squeeze().cpu().numpy()
    
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    return pred_binary, pred_mask

def crop_segmented_region(original_img, segmentation_mask, margin=8):
    
    #data arrays
    original_data = original_img.get_fdata()
    segmentation_data = segmentation_mask.get_fdata()
    affine = original_img.affine
    
    binary_mask = (segmentation_data > 0).astype(np.uint8)
    
    #bounding box of the segmentation
    if np.sum(binary_mask) == 0:
        return original_img, (0, original_data.shape[0], 0, original_data.shape[1], 0, original_data.shape[2])
    
    non_zero_indices = np.where(binary_mask > 0)
    
    # Calculate bounding box with margin
    z_start = max(0, np.min(non_zero_indices[0]) - margin)
    z_end = min(original_data.shape[0], np.max(non_zero_indices[0]) + margin + 1)
    y_start = max(0, np.min(non_zero_indices[1]) - margin)
    y_end = min(original_data.shape[1], np.max(non_zero_indices[1]) + margin + 1)
    x_start = max(0, np.min(non_zero_indices[2]) - margin)
    x_end = min(original_data.shape[2], np.max(non_zero_indices[2]) + margin + 1)
    
    #Crop
    cropped_data = original_data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    #affine matrix for the cropped region
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.array([x_start, y_start, z_start])
    
    cropped_img = nib.Nifti1Image(cropped_data, new_affine)
    
    bbox = (z_start, z_end, y_start, y_end, x_start, x_end)
    
    return cropped_img, bbox

def combine_masks(mask1, mask2, label1=1, label2=2):
    
    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)
    
    #combined mask
    combined_mask = np.zeros_like(mask1_binary, dtype=np.uint8)
    
    #labels
    combined_mask[mask1_binary > 0] = label1
    combined_mask[mask2_binary > 0] = label2
    
    return combined_mask

def predict_all_structures(test_image_dir, 
                          lat_ventricle_model_path, 
                          lat_chp_model_path,
                          third_ventricle_model_path,
                          third_chp_model_path,
                          output_base_dir='prediction',
                          threshold=0.5,
                          margin=8):
    """
    Main function to predict all ChP
    """
    
    #output directories
    output_dirs = {
        'original': os.path.join(output_base_dir, 'original_file'),
        'lat_ventricle': os.path.join(output_base_dir, 'lat_ventricle_mask'),
        'lat_chp': os.path.join(output_base_dir, 'lat_chp_mask'),
        'third_ventricle': os.path.join(output_base_dir, 'third_ventricle_mask'),
        'third_chp': os.path.join(output_base_dir, 'third_chp_mask'),
        'combined_ventricles': os.path.join(output_base_dir, 'combined_ventricles'),
        'combined_chp': os.path.join(output_base_dir, 'combined_chp')
    }
    
    
    for dir_path in output_dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Loading model weights
    print("Loading weights...")
    lat_ventricle_model = load_model(lat_ventricle_model_path)
    lat_chp_model = load_model(lat_chp_model_path)
    third_ventricle_model = load_model(third_ventricle_model_path)
    third_chp_model = load_model(third_chp_model_path)
    
    # Get input images
    test_images = [os.path.join(test_image_dir, f)
                  for f in os.listdir(test_image_dir) 
                  if f.endswith(('.nii', '.nii.gz'))]
    
    print(f"Found {len(test_images)} input images")
    
    # Process each image
    for img_path in test_images:
        filename = os.path.basename(img_path)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print('='*60)
        
        try:
            # Generate base name
            if filename.endswith('.nii.gz'):
                base_name = filename.replace('.nii.gz', '')
                extension = '.nii.gz'
            else:
                base_name = filename.replace('.nii', '')
                extension = '.nii'
            
            # Load original image
            original_img = nib.load(img_path)
            original_affine = original_img.affine
            original_shape = original_img.shape
            
            # STEP 0: Save original image
            original_filename = f"{base_name}{extension}"
            original_output_path = os.path.join(output_dirs['original'], original_filename)
            nib.save(original_img, original_output_path)
            print(f"Saved original image")
            
            
            #PART 1: LATERAL VENTRICLE AND LATERAL CHP
            
            print("\n Lateral Ventricle and Lateral ChP")
            
            # Step 1: Predict lateral ventricle
            print("Predicting lateral ventricle...")
            lat_vent_pred_nifti, _, _, _ = predict_central_patch(
                lat_ventricle_model, 
                img_path, 
                patch_size=(128, 128, 128),
                threshold=threshold
            )
            
            # Save lateral ventricle mask
            lat_vent_filename = f"{base_name}{extension}"
            lat_vent_output_path = os.path.join(output_dirs['lat_ventricle'], lat_vent_filename)
            nib.save(lat_vent_pred_nifti, lat_vent_output_path)
            print(f"✓Lateral ventricle mask saved")
            
            # Step 2: Crop region around lateral ventricle
            print("Processing lateral ventricle region...")
            cropped_lat_img, lat_bbox = crop_segmented_region(
                original_img, 
                lat_vent_pred_nifti, 
                margin=margin
            )
            
            # Step 3: Predict lateral ChP on cropped region
            print("Predicting lateral ChP...")
            cropped_lat_data = cropped_lat_img.get_fdata()
            lat_chp_pred_binary, _ = predict_direct(
                lat_chp_model,
                cropped_lat_data,
                threshold=threshold
            )
            
            # Create full-size lateral ChP mask
            print("Reverting full-size lateral ChP mask...")
            full_lat_chp_mask = np.zeros(original_shape, dtype=np.uint8)
            
            # Map predictions back to original coordinates
            z_start, z_end, y_start, y_end, x_start, x_end = lat_bbox
            full_lat_chp_mask[z_start:z_end, y_start:y_end, x_start:x_end] = lat_chp_pred_binary
            
            # Save lateral ChP mask
            lat_chp_full_filename = f"{base_name}{extension}"
            lat_chp_full_output_path = os.path.join(output_dirs['lat_chp'], lat_chp_full_filename)
            lat_chp_full_nifti = nib.Nifti1Image(full_lat_chp_mask, original_affine)
            nib.save(lat_chp_full_nifti, lat_chp_full_output_path)
            print(f"✓ Lateral ChP mask saved")
            
        
            # PART 2: 3RD VENTRICLE AND 3RD VENTRICLE CHP
           
            print("\n3rd Ventricle and 3rd Ventricle ChP")
            
            # Step 5: Predict 3rd ventricle
            print("Predicting 3rd ventricle...")
            third_vent_pred_nifti, _, _, _ = predict_central_patch(
                third_ventricle_model, 
                img_path, 
                patch_size=(128, 128, 128),
                threshold=threshold
            )
            
            # Save 3rd ventricle mask
            third_vent_filename = f"{base_name}{extension}"
            third_vent_output_path = os.path.join(output_dirs['third_ventricle'], third_vent_filename)
            nib.save(third_vent_pred_nifti, third_vent_output_path)
            print(f"✓ 3rd ventricle mask saved")
            
            # Crop region around 3rd ventricle (for processing only)
            print("Processing 3rd ventricle region...")
            cropped_third_img, third_bbox = crop_segmented_region(
                original_img, 
                third_vent_pred_nifti, 
                margin=margin
            )
            
            # Step 7: Predict 3rd ventricle ChP on cropped region
            print("Predicting 3rd ventricle ChP...")
            cropped_third_data = cropped_third_img.get_fdata()
            third_chp_pred_binary, _ = predict_direct(
                third_chp_model,
                cropped_third_data,
                threshold=threshold
            )
            
            # Create full-size 3rd ventricle ChP mask
            print("Reverting full-size 3rd ventricle ChP mask...")
            full_third_chp_mask = np.zeros(original_shape, dtype=np.uint8)
            
            # Map predictions back to original coordinates
            z_start, z_end, y_start, y_end, x_start, x_end = third_bbox
            full_third_chp_mask[z_start:z_end, y_start:y_end, x_start:x_end] = third_chp_pred_binary
            
            # Save 3rd ventricle ChP mask
            third_chp_full_filename = f"{base_name}{extension}"
            third_chp_full_output_path = os.path.join(output_dirs['third_chp'], third_chp_full_filename)
            third_chp_full_nifti = nib.Nifti1Image(full_third_chp_mask, original_affine)
            nib.save(third_chp_full_nifti, third_chp_full_output_path)
            print(f"✓ 3rd ventricle ChP mask saved")
            
            
            # PART 3: COMBINE MASKS
            
            print("\nCombining Masks")
            
            # Load masks for combining
            lat_vent_mask_data = lat_vent_pred_nifti.get_fdata()
            third_vent_mask_data = third_vent_pred_nifti.get_fdata()
            
            # Step 9: Combine ventricle masks
            print("Combining ventricle masks...")
            combined_ventricles = combine_masks(
                lat_vent_mask_data, 
                third_vent_mask_data, 
                label1=1, 
                label2=2
            )
            
            combined_vent_filename = f"{base_name}{extension}"
            combined_vent_output_path = os.path.join(output_dirs['combined_ventricles'], combined_vent_filename)
            combined_vent_nifti = nib.Nifti1Image(combined_ventricles, original_affine)
            nib.save(combined_vent_nifti, combined_vent_output_path)
            print(f"✓ Combined ventricles mask saved")
            
            # Step 10: Combine ChP masks
            print("Combining ChP masks...")
            combined_chp = combine_masks(
                full_lat_chp_mask, 
                full_third_chp_mask, 
                label1=1, 
                label2=2
            )
            
            combined_chp_filename = f"{base_name}{extension}"
            combined_chp_output_path = os.path.join(output_dirs['combined_chp'], combined_chp_filename)
            combined_chp_nifti = nib.Nifti1Image(combined_chp, original_affine)
            nib.save(combined_chp_nifti, combined_chp_output_path)
            print(f"✓ Combined ChP mask saved")
            
            print(f"\n✓ Successfully processed {filename}")
            
        except Exception as e:
            print(f"✗ Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print('='*60)
    print(f"\nOutput directories created in '{output_base_dir}':")
    for dir_name, dir_path in output_dirs.items():
        print(f"  • {dir_name}: {dir_path}")
    
    return output_dirs

# MAIN Function
if __name__ == "__main__":
    #model paths (weights)
    LAT_VENTRICLE_MODEL_PATH = "weights/lventricle_weight.pth"
    LAT_CHP_MODEL_PATH = "weights/lchp_weight.pth"
    THIRD_VENTRICLE_MODEL_PATH = "weights/3ventricle_weight.pth"
    THIRD_CHP_MODEL_PATH = "weights/3chp_weight.pth"
    
    # Path to directory containing input NIfTI files
    TEST_IMAGE_DIR = "input_data"
    
    # Prediction directory
    OUTPUT_BASE_DIR = "prediction"
    
    # Parameters
    THRESHOLD = 0.5
    MARGIN = 8
    
    # Check if paths exist
    model_paths = [
        (LAT_VENTRICLE_MODEL_PATH, "Lateral ventricle model"),
        (LAT_CHP_MODEL_PATH, "Lateral ChP model"),
        (THIRD_VENTRICLE_MODEL_PATH, "3rd ventricle model"),
        (THIRD_CHP_MODEL_PATH, "3rd ventricle ChP model"),
        (TEST_IMAGE_DIR, "Test image directory")
    ]
    
    for path, description in model_paths:
        if not os.path.exists(path):
            print(f"ERROR: {description} not found at {path}")
            exit(1)
    
    # Run  prediction
    output_dirs = predict_all_structures(
        test_image_dir=TEST_IMAGE_DIR,
        lat_ventricle_model_path=LAT_VENTRICLE_MODEL_PATH,
        lat_chp_model_path=LAT_CHP_MODEL_PATH,
        third_ventricle_model_path=THIRD_VENTRICLE_MODEL_PATH,
        third_chp_model_path=THIRD_CHP_MODEL_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        threshold=THRESHOLD,
        margin=MARGIN
    )
