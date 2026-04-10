\title{\textbf{\huge Graph Cut Image Segmentation}}

## Project Overview
This project implements a robust binary image segmentation pipeline using the Graph Cut (Min-Cut/Max-Flow) methodology. It separates foreground objects from their backgrounds by formulating segmentation as an energy minimization problem, utilizing Gaussian Mixture Models (GMMs) for data likelihoods and pairwise pixel gradients for spatial smoothing.

### Key Features Implemented
* **Dual Annotation Modes:** Supports standard Bounding Box initialization as well as explicit Foreground/Background Scribble annotations for complex topologies (e.g., ring shapes or medical images).
* **Iterative Optimization:** Implements a "GrabCut" style iterative loop. The mask progressively updates to shed background pollution from the initial user annotation, leading to tighter, cleaner GMM models.
* **GMM-Based Data Term:** Fits full-covariance Gaussian Mixture Models to capture distinct, multi-modal color clusters in both the foreground and background.
* **Morphological Refinement:** Applies opening and closing structural operations to mitigate salt-and-pepper noise and smooth final extraction boundaries.
* **Algorithmic Comparison:** Automatically calculates and displays a "Naive Segmentation" (Data Term only) to visually demonstrate the critical impact of the Smoothness Term in the Graph Cut algorithm.

---

## Dependencies

This project relies on Python 3.x and the following libraries. You can install them via pip:

```bash
pip install opencv-python numpy matplotlib scikit-learn PyMaxflow
```
Note: PyMaxflow is used specifically for the optimized Min-Cut/Max-Flow graph traversal.

## How to Run the Code

1. Place the Python script and your target images in the same directory (or update the `image_path` in the main execution block).
2. Execute the script from the terminal:
   ```bash
   python graph_cut_segmenter.py
   ```
   ### Interaction Instructions
    Depending on the annotation mode enabled in the code, follow these steps when the OpenCV window appears:

    **Option A: Bounding Box Mode**
    * Click and drag to draw a rectangle encompassing the entire foreground object.
    * Try to minimize the amount of background floor/wall included in the box.
    * Press `ENTER` to confirm and start the algorithm, or `C` to cancel.

    **Option B: Scribble Mode**
    * **Left Click & Drag:** Draw RED lines on the definite foreground object.
    * **Right Click & Drag:** Draw BLUE lines on the definite background (floor, walls, shadows).
    * *Tip:* For objects with holes (like the arch of headphones), ensure you right-click to place a blue dot inside the empty space.
    * Press `ENTER` to confirm and start the pipeline.

---

## Output
Upon completion, the script will automatically generate and save a high-resolution 6-pane comparison figure (`result_<filename>.jpg`) in the working directory. The panes display:
1. **Original Image + Annotations:** Displays the user's initial bounding box or red/blue scribbles.
2. **Naive Segmentation:** Pixel-by-pixel classification ignoring spatial constraints (highlights noise).
3. **Raw Graph Cut Mask:** The raw output of the Min-Cut optimization.
4. **Refined Mask Overlay:** The final mask rendered as a semi-transparent red overlay on the original image.
5. **Extracted Object:** The isolated foreground object with a pure black background.

---

