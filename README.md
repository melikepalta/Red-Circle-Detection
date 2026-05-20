# Red Circle Detection

A Python script that scans a folder of images and identifies which ones contain red circles using OpenCV. Built to process a batch of 100 images and pull out the ones that match.

---

## How It Works

The script reads every image in the `test images` folder and runs each one through a detection pipeline with four filters:

1. **HSV Color Masking** — Converts the image to HSV color space and isolates red pixels using two hue ranges (0–8 and 170–179) to account for how red wraps around in HSV.
2. **Contour Area** — Ignores shapes that are too small to be meaningful (default minimum: 1000px²).
3. **Circularity** — Measures how close a shape is to a perfect circle using the formula `4π * area / perimeter²`. A value of 1.0 is a perfect circle. Shapes below 0.75 are rejected.
4. **Fill Ratio** — Compares the contour area to the area of its minimum enclosing circle. This filters out hollow rings and irregular blobs. Default minimum: 0.70.

If a shape passes all four checks, the image is flagged as containing a red circle and saved to the `detected_images` folder.

---

## Debug Mode

With `DEBUG = True`, the script prints detailed logs for every image explaining why each shape passed or failed:

- Number of red pixels found in the mask
- Area, circularity, and fill ratio for the largest contour
- Whether each threshold was met or not

This makes it easy to tune the parameters if detection is off.

---

## Tuning the Parameters

If the detection isn't working well on your images, here's what to adjust:

| Problem | What to try |
|---|---|
| Missing darker red circles | Lower the `V` minimum in `lower_red1` / `lower_red2` |
| Missing washed-out reds | Lower the `S` minimum in the HSV arrays |
| Off-shade reds not detected | Widen the hue (`H`) range |
| Distorted circles being filtered out | Lower `MIN_CIRCULARITY` (try 0.65) |
| Partially obscured circles missed | Lower `MIN_FILL_RATIO` |
| Small circles not detected | Lower `MIN_AREA` |
| Too many false positives | Raise any of the above thresholds |

---

## Project Structure

```
.
├── test images/        # Input images (place your 100 images here)
├── detected_images/    # Output folder (created automatically)
├── detection.py        # Main script
└── README.md
```

---

## Requirements

```
opencv-python
numpy
```

Install with:

```bash
pip install opencv-python numpy
```

---

## Usage

1. Put your images in the `test images` folder.
2. Run the script:

```bash
python detection.py
```

3. Images with red circles will be saved to `detected_images/` and flagged in the console output.
