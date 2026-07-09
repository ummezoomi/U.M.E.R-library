# Umer_Library/modules/perception.py

import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

# =====================================================================
# THE PERCEPTUAL ENCODER
# Converts raw 2D pixel matrices into 111-D Abstract Spatial Concepts
# =====================================================================

def extract_quadrant_features(img_gray):
    """
    Extracts 12 Abstract Concepts per spatial zone (Texture + Edge Density).
    Uses Gray-Level Co-occurrence Matrices (GLCM) and Sobel Gradients.
    """
    if np.max(img_gray) == 0: 
        return np.zeros(12, dtype=np.float32)
    
    mean_val = np.mean(img_gray)
    std_val = np.std(img_gray)
    
    if std_val < 1e-5:
        skw, krt = 0.0, 0.0
    else:
        skw = skew(img_gray.flatten())
        krt = kurtosis(img_gray.flatten())
    
    # Gradient/Edge Tracking
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(edge_mag)
    edge_var = np.std(edge_mag)
    
    # Textural Weave (GLCM)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [mean_val, std_val, skw, krt, edge_density, edge_var, 
            contrast, homogeneity, energy, correlation, np.max(edge_mag), np.median(img_gray)]

def abstract_image(img):
    """
    The Master Extractor: 9-Zone Textures + Global Holistic Morphology.
    Output: 111-D Vector (9 zones * 12 features + 3 global features).
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge-Crop to remove scanner/border artifacts (10%)
    h_orig, w_orig = img_gray.shape
    crop_y, crop_x = int(h_orig * 0.10), int(w_orig * 0.10)
    img_gray = img_gray[crop_y : h_orig - crop_y, crop_x : w_orig - crop_x]
    
    # 2. Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # ==========================================
    # STAGE 1: GLOBAL DETERMINISTIC CONTOUR
    # ==========================================
    heavy_blur = cv2.GaussianBlur(img_clahe, (15, 15), 0)
    _, thresh = cv2.threshold(heavy_blur, np.percentile(heavy_blur, 35), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circularity, solidity, aspect_ratio = 0.0, 0.0, 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
            
        x, y, w_box, h_box = cv2.boundingRect(c)
        if h_box > 0:
            aspect_ratio = float(w_box) / h_box
            
    # ==========================================
    # STAGE 2: 9-ZONE SLICING (3x3 Grid)
    # ==========================================
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0) 
    h, w = img_blur.shape
    step_y, step_x = h // 3, w // 3
    features = []
    
    for row in range(3):
        for col in range(3):
            patch = img_blur[row*step_y : (row+1)*step_y, col*step_x : (col+1)*step_x]
            features.extend(extract_quadrant_features(patch))
            
    # Append the 3 Macro features to the end of the 108 micro-features
    features.extend([circularity, solidity, aspect_ratio])
    
    return np.nan_to_num(np.array(features, dtype=np.float32))

def get_feature_names():
    """Returns the ordered labels for the 111-D vector, useful for PCA and Autopsies."""
    feature_names = []
    base_names = ["Mean", "Std", "Skew", "Kurt", "EdgeDens", "EdgeVar", "Contrast", "Homogen", "Energy", "Correl", "MaxEdge", "Median"]
    for row in range(3):
        for col in range(3):
            for name in base_names:
                feature_names.append(f"Z{row}{col}_{name}")
                
    feature_names.extend(["GLOBAL_Circularity", "GLOBAL_Solidity", "GLOBAL_AspectRatio"])
    return feature_names