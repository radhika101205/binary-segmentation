import cv2
import numpy as np
import maxflow
import matplotlib.pyplot as plt

def segment_with_histograms(image_path, bins=32):
    # 1. Load Image
    img = cv2.imread("/home/radhika/Documents/CV-assignment-2/images/object.jpeg")
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Scale down if too large
    h, w = img.shape[:2]
    if max(h, w) > 800:
        img = cv2.resize(img, (int(w * 800/max(h,w)), int(h * 800/max(h,w))))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 2. Get User Annotation (Bounding Box)
    print("Select Foreground ROI and press ENTER.")
    roi = cv2.selectROI("Select ROI", img, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, bw, bh = roi

    # Create masks
    fg_mask = np.zeros((h, w), dtype=bool)
    fg_mask[y:y+bh, x:x+bw] = True
    bg_mask = ~fg_mask

    # 3. Build 3D Color Histograms
    # Quantize the image into fewer bins (e.g., 32x32x32 instead of 256x256x256)
    quantized_img = (img_rgb / (256 / bins)).astype(int)
    
    fg_pixels = quantized_img[fg_mask]
    bg_pixels = quantized_img[bg_mask]

    # Calculate 3D histograms
    fg_hist, _ = np.histogramdd(fg_pixels, bins=(bins, bins, bins), range=((0, bins), (0, bins), (0, bins)))
    bg_hist, _ = np.histogramdd(bg_pixels, bins=(bins, bins, bins), range=((0, bins), (0, bins), (0, bins)))

    # Add epsilon to prevent log(0) and normalize to create probability distributions
    epsilon = 1e-7
    fg_prob = (fg_hist + epsilon) / (np.sum(fg_hist) + epsilon * (bins**3))
    bg_prob = (bg_hist + epsilon) / (np.sum(bg_hist) + epsilon * (bins**3))

    # 4. Unary Costs (Data Term)
    # Map every pixel to its corresponding probability bin
    r_idx, g_idx, b_idx = quantized_img[:,:,0], quantized_img[:,:,1], quantized_img[:,:,2]
    
    # Calculate negative log-likelihoods
    unary_fg = -np.log(fg_prob[r_idx, g_idx, b_idx]).astype(np.float32)
    unary_bg = -np.log(bg_prob[r_idx, g_idx, b_idx]).astype(np.float32)

    # 5. Graph Construction
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((h, w))

    # Add Terminal Edges (Unary Costs)
    g.add_grid_tedges(nodes, unary_bg, unary_fg)

    # Add Neighbor Edges (Pairwise Costs / Smoothness)
    lambda_smooth = 50
    sigma = 5.0
    img_float = img_rgb.astype(np.float32)

    # Horizontal edges
    diff_x = img_float[:, 1:, :] - img_float[:, :-1, :]
    weight_x_partial = lambda_smooth * np.exp(-np.sum(diff_x**2, axis=2) / (2 * sigma**2))
    weight_x = np.zeros((h, w), dtype=np.float32)
    weight_x[:, :-1] = weight_x_partial

    # Vertical edges
    diff_y = img_float[1:, :, :] - img_float[:-1, :, :]
    weight_y_partial = lambda_smooth * np.exp(-np.sum(diff_y**2, axis=2) / (2 * sigma**2))
    weight_y = np.zeros((h, w), dtype=np.float32)
    weight_y[:-1, :] = weight_y_partial

    g.add_grid_edges(nodes, weight_x, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=True)
    g.add_grid_edges(nodes, weight_y, np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=True)

    # Hard Constraints: Force area outside box to be Sink (Background)
    inf_cap = 1e10
    bg_enforcement = np.where(bg_mask, inf_cap, 0).astype(np.float32)
    g.add_grid_tedges(nodes, 0, bg_enforcement)

    # 6. Min-Cut / Max-Flow
    print("Calculating Min-Cut (Histogram)...")
    g.maxflow()
    segments = g.get_grid_segments(nodes)
    
    # Invert so FG=1, BG=0
    final_mask = np.logical_not(segments).astype(np.uint8)

    # Visualization
    extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.title("Histogram Unary Costs (BG Probability)"), plt.imshow(unary_bg, cmap='hot')
    plt.subplot(1, 2, 2), plt.title("Extracted Object"), plt.imshow(extracted)
    plt.show()

# Run the function
if __name__ == "__main__":
    segment_with_histograms("your_image.jpg") # Replace with your image