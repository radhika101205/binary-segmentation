import cv2
import numpy as np
import maxflow
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

class GraphCutSegmenter:
    def __init__(self, image_path, n_components=3):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Resize if image is too large to keep graph-cut computationally feasible
        h, w = self.original_image.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            self.image = cv2.resize(self.original_image, (int(w * scale), int(h * scale)))
        else:
            self.image = self.original_image.copy()

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.h, self.w, _ = self.image.shape
        self.n_components = n_components

# ==========================================
    # IF I WANT TO USE BOUNDING BOX FOR ANNOTATING IMAGE
# ==========================================

        
    def annotate_image(self):
        """Step 1: User annotation via bounding box."""
        print("Drag a box around the foreground object. Press ENTER to confirm, or 'c' to cancel.")
        roi = cv2.selectROI("Annotate Bounding Box", self.image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Annotate Bounding Box")

        x, y, w, h = roi
        if w == 0 and h == 0:
            raise ValueError("No bounding box selected. Exiting.")

        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.mask[y:y+h, x:x+w] = 1 # 1 = Probable Foreground, 0 = Definite Background
        self.bbox_mask = self.mask.copy()
        print("Annotation complete.")

    def train_gmms(self):
        """Step 2: Foreground-Background Modeling using GMMs."""
        print("Training Gaussian Mixture Models...")
        
        pixels = self.image_rgb.reshape(-1, 3).astype(np.float64)
        mask_flat = self.mask.flatten()

        fg_pixels = pixels[mask_flat == 1]
        bg_pixels = pixels[mask_flat == 0]

        self.gmm_fg = GaussianMixture(n_components=self.n_components, covariance_type='full', reg_covar=1e-4)
        self.gmm_bg = GaussianMixture(n_components=self.n_components, covariance_type='full', reg_covar=1e-4)

        self.gmm_fg.fit(fg_pixels)
        self.gmm_bg.fit(bg_pixels)
        
        self.unary_fg = -self.gmm_fg.score_samples(pixels).reshape(self.h, self.w)
        self.unary_bg = -self.gmm_bg.score_samples(pixels).reshape(self.h, self.w)

        # The Naive Segmentation: Pixel-by-pixel thresholding
        # If foreground score is lower (better) than background, mark as 1 (Foreground)
        self.naive_segmentation = (self.unary_fg < self.unary_bg).astype(np.uint8)

    def construct_graph_and_cut(self, lambda_smooth=50, sigma=5.0):
        """Step 3 & 4: Graph Construction and Min-Cut / Max-Flow."""
        print("Constructing graph and calculating min-cut...")
        
        # Create the graph
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes((self.h, self.w))

        # 1. Add Terminal Edges (Unary Costs / Data Term)
        # Cost to connect to Source (FG) is the likelihood it belongs to BG
        # Cost to connect to Sink (BG) is the likelihood it belongs to FG
        g.add_grid_tedges(nodes, self.unary_bg, self.unary_fg)

        # 2. Add Neighbor Edges (Pairwise Costs / Smoothness Term)
        # Calculate gradients (differences between neighboring pixels)
        img_float = self.image_rgb.astype(np.float32)
        
        # Horizontal edges (x-direction)
        diff_x = img_float[:, 1:, :] - img_float[:, :-1, :]
        weight_x_partial = lambda_smooth * np.exp(-np.sum(diff_x**2, axis=2) / (2 * sigma**2))
        # Pad right side to match image dimensions (H, W)
        weight_x = np.zeros((self.h, self.w), dtype=np.float32)
        weight_x[:, :-1] = weight_x_partial
        
        # Vertical edges (y-direction)
        diff_y = img_float[1:, :, :] - img_float[:-1, :, :]
        weight_y_partial = lambda_smooth * np.exp(-np.sum(diff_y**2, axis=2) / (2 * sigma**2))
        # Pad bottom to match image dimensions (H, W)
        weight_y = np.zeros((self.h, self.w), dtype=np.float32)
        weight_y[:-1, :] = weight_y_partial

        # Adding edges to graph structure (right and down)
        structure_x = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, 0, 0]])
        structure_y = np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0]])

        g.add_grid_edges(nodes, weight_x, structure_x, symmetric=True)
        g.add_grid_edges(nodes, weight_y, structure_y, symmetric=True)

        # 3. Applying Hard Constraints based on bounding box
        # Enforce pixels outside the bounding box to strictly be background
        inf_cap = 1e10
        bg_enforcement = np.where(self.mask == 0, inf_cap, 0).astype(np.float32)
        g.add_grid_tedges(nodes, 0, bg_enforcement)
        # 4. Compute Max Flow / Min Cut
        g.maxflow()
        
        # Get the segments (0 for source/FG, 1 for sink/BG in PyMaxflow convention)
        segments = g.get_grid_segments(nodes)
        
        # Invert so 1 is FG, 0 is BG
        self.raw_segmentation = np.logical_not(segments).astype(np.uint8)

# ==========================================
    # IF I WANT TO USE SCRIBBLES FOR ANNOTATING IMAGE
# ==========================================

    # def annotate_image(self):
    #     """Step 1: User annotation via scribbles."""
    #     print("--- Scribble Annotation Mode ---")
    #     print("LEFT CLICK & DRAG : Draw Foreground (Red)")
    #     print("RIGHT CLICK & DRAG: Draw Background (Blue)")
    #     print("Press 'ENTER' when finished.")

    #     # Initialize mask with 2 (Unknown)
    #     self.mask = np.full((self.h, self.w), 2, dtype=np.uint8)
    #     drawing_vis = self.image.copy()
        
    #     drawing = False
    #     mode = -1 # 1 for FG, 0 for BG

    #     def draw_circle(event, x, y, flags, param):
    #         nonlocal drawing, mode
    #         # Brush size
    #         radius = 5 
            
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             drawing = True
    #             mode = 1
    #         elif event == cv2.EVENT_RBUTTONDOWN:
    #             drawing = True
    #             mode = 0
    #         elif event == cv2.EVENT_MOUSEMOVE:
    #             if drawing:
    #                 if mode == 1:
    #                     cv2.circle(drawing_vis, (x, y), radius, (0, 0, 255), -1) # Red for FG
    #                     cv2.circle(self.mask, (x, y), radius, 1, -1)
    #                 elif mode == 0:
    #                     cv2.circle(drawing_vis, (x, y), radius, (255, 0, 0), -1) # Blue for BG
    #                     cv2.circle(self.mask, (x, y), radius, 0, -1)
    #         elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
    #             drawing = False

    #     cv2.namedWindow('Annotate Scribbles')
    #     cv2.setMouseCallback('Annotate Scribbles', draw_circle)

    #     while True:
    #         cv2.imshow('Annotate Scribbles', drawing_vis)
    #         k = cv2.waitKey(1) & 0xFF
    #         if k == 13: # Enter key
    #             break
                
    #     cv2.destroyAllWindows()
        
        
    #     # Save the original scribbles for hard constraints during iterations
    #     self.bbox_mask = self.mask.copy() 
    #     print("Annotation complete.")

    # def train_gmms(self):
    #     """Step 2: Foreground-Background Modeling using GMMs."""
    #     print("Training Gaussian Mixture Models...")
    #     pixels = self.image_rgb.reshape(-1, 3).astype(np.float64)
    #     mask_flat = self.mask.flatten()

    #     # ONLY train on the specific pixels the user scribbled on
    #     fg_pixels = pixels[mask_flat == 1]
    #     bg_pixels = pixels[mask_flat == 0]

    #     if len(fg_pixels) == 0 or len(bg_pixels) == 0:
    #         raise ValueError("You must draw BOTH foreground and background scribbles!")

    #     self.gmm_fg = GaussianMixture(n_components=self.n_components, covariance_type='full', reg_covar=1e-4)
    #     self.gmm_bg = GaussianMixture(n_components=self.n_components, covariance_type='full', reg_covar=1e-4)

    #     self.gmm_fg.fit(fg_pixels)
    #     self.gmm_bg.fit(bg_pixels)
        
    #     self.unary_fg = -self.gmm_fg.score_samples(pixels).reshape(self.h, self.w)
    #     self.unary_bg = -self.gmm_bg.score_samples(pixels).reshape(self.h, self.w)

    # def construct_graph_and_cut(self, lambda_smooth=50, sigma=5.0):
    #     """Step 3 & 4: Graph Construction and Min-Cut."""
    #     print("Constructing graph and calculating min-cut...")
    #     g = maxflow.Graph[float]()
    #     nodes = g.add_grid_nodes((self.h, self.w))

    #     # 1. Base Unary Costs (Data Term from GMMs)
    #     source_caps = self.unary_bg.copy() 
    #     sink_caps = self.unary_fg.copy()

    #     # 2. Apply Hard Constraints based on original scribbles
    #     inf_cap = 1e10
        
    #     # Force RED scribbles to be Foreground
    #     source_caps[self.bbox_mask == 1] = inf_cap
    #     sink_caps[self.bbox_mask == 1] = 0
        
    #     # Force BLUE scribbles to be Background
    #     source_caps[self.bbox_mask == 0] = 0
    #     sink_caps[self.bbox_mask == 0] = inf_cap

    #     # Add the terminal edges
    #     g.add_grid_tedges(nodes, source_caps, sink_caps)

    #     # 3. Add Neighbor Edges (Pairwise Costs / Smoothness Term)
    #     img_float = self.image_rgb.astype(np.float32)
        
    #     diff_x = img_float[:, 1:, :] - img_float[:, :-1, :]
    #     weight_x_partial = lambda_smooth * np.exp(-np.sum(diff_x**2, axis=2) / (2 * sigma**2))
    #     weight_x = np.zeros((self.h, self.w), dtype=np.float32)
    #     weight_x[:, :-1] = weight_x_partial
        
    #     diff_y = img_float[1:, :, :] - img_float[:-1, :, :]
    #     weight_y_partial = lambda_smooth * np.exp(-np.sum(diff_y**2, axis=2) / (2 * sigma**2))
    #     weight_y = np.zeros((self.h, self.w), dtype=np.float32)
    #     weight_y[:-1, :] = weight_y_partial

    #     structure_x = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    #     structure_y = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

    #     g.add_grid_edges(nodes, weight_x, structure_x, symmetric=True)
    #     g.add_grid_edges(nodes, weight_y, structure_y, symmetric=True)

    #     # 4. Compute Max Flow / Min Cut
    #     g.maxflow()
    #     segments = g.get_grid_segments(nodes)
    #     self.raw_segmentation = np.logical_not(segments).astype(np.uint8)

    def refine_segmentation(self):
        """Step 5: Artifact Mitigation & Refinement using Morphological Operations."""
        print("Refining segmentation boundaries...")
        
        # Use a 5x5 elliptical kernel for smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Morphological OPENING: removes small noise points in the background
        refined_mask = cv2.morphologyEx(self.raw_segmentation, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Morphological CLOSING: fills small holes inside the foreground object
        self.final_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    def visualize_results(self):
        """Display the final output and overlays."""
        # Create an overlay (e.g., Red tint over foreground)
        overlay = self.image_rgb.copy()
        overlay[self.final_mask == 1] = [255, 0, 0] # Red mask
        
        # Blended output
        blended = cv2.addWeighted(self.image_rgb, 0.6, overlay, 0.4, 0)

        # Extract just the object with a black background
        extracted = cv2.bitwise_and(self.image_rgb, self.image_rgb, mask=self.final_mask)

        # Plotting

        # Top left (if using bounding box for annotating)

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.title("Original Image + Bounding Box")
        annotated_vis = self.image_rgb.copy()
        # Find bbox to draw it for visualization
        y_indices, x_indices = np.where(self.mask == 1)
        if len(y_indices) > 0:
            cv2.rectangle(annotated_vis, (min(x_indices), min(y_indices)), 
                          (max(x_indices), max(y_indices)), (0, 255, 0), 3)
        plt.imshow(annotated_vis)

        # Top left (if using scribbles for annotating)

        # plt.title("Original Image + Scribbles")
        # annotated_vis = self.image_rgb.copy()
        
        # # Paint the foreground scribbles Red
        # annotated_vis[self.bbox_mask == 1] = [255, 0, 0] 
        # # Paint the background scribbles Blue
        # annotated_vis[self.bbox_mask == 0] = [0, 0, 255] 
        
        # plt.imshow(annotated_vis)

        plt.axis('off')

        # --- Top Middle: Naive Segmentation ---
        plt.subplot(2, 3, 2)
        plt.title("Naive Segmentation (No Graph Cut)")
        plt.imshow(self.naive_segmentation, cmap='gray')
        plt.axis('off')

        # --- Top Right: Raw Graph Cut Mask ---
        plt.subplot(2, 3, 3)
        plt.title("Raw Graph Cut Mask")
        plt.imshow(self.raw_segmentation, cmap='gray')
        plt.axis('off')

        # --- Bottom Left: Refined Overlay ---
        plt.subplot(2, 3, 4)
        plt.title("Refined Mask Overlay")
        plt.imshow(blended)
        plt.axis('off')

        # --- Bottom Middle: Final Extraction ---
        plt.subplot(2, 3, 5)
        plt.title("Extracted Object")
        plt.imshow(extracted)
        plt.axis('off')


        plt.tight_layout()

        # ==========================================
        # SAVING LOGIC
        # ==========================================
        import os
        base_filename = os.path.basename(self.image_path)
        
        # 2. Create a new output filename
        save_path = f"result2_{base_filename}"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSuccess! 4-pane figure saved to: {save_path}")

        # 4. Display the window
        plt.show()


# ==========================================
# Execution Block (Iterative Approach)
# ==========================================
if __name__ == "__main__":
    IMAGE_FILE = "/home/radhika/Documents/CV-assignment-2/images/brain-tomor-menigloma.jpg" 
    
    try:
        pipeline = GraphCutSegmenter(IMAGE_FILE, n_components=3)
        pipeline.annotate_image()
        
        # --- Iterative Optimization Loop ---
        iterations = 3
        print(f"\nStarting {iterations} iterations of optimization...")
        
        # We need to keep the original bounding box safe for hard constraints
        original_bbox_mask = pipeline.mask.copy() 
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1} ---")
            pipeline.train_gmms()
            
            # Apply graph cut. We must enforce that anything outside the 
            # ORIGINAL bounding box is always background.
            pipeline.construct_graph_and_cut(lambda_smooth=50, sigma=3.0)
            
            # The crucial iterative step: 
            # Update the training mask for the next iteration's GMMs.
            # We take the new segmentation, but force the outside of the bbox to remain 0
            pipeline.mask = np.where(original_bbox_mask == 0, 0, pipeline.raw_segmentation)
            
        # --- Final Refinement ---
        pipeline.refine_segmentation()
        pipeline.visualize_results()
        
    except Exception as e:
        print(f"Pipeline failed: {e}")