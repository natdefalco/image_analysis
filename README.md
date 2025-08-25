Fractured Chalk Image Analysis

This repository contains Python tools for analyzing fluid flow and dye infiltration in fractured chalk experiments.
The scripts process image sequences of Brilliant Blue dye transport through fractured chalk cores, enabling quantification of fingering patterns, flow area, and temporal dynamics.

✨ Features

Image cropping & preprocessing (multiple_experiments_area.py, multiple_exps.py)

Interactive cropping of images from multiple experiment folders.

Conversion of pixel area to calibrated physical units (mm²).

Batch processing with results saved to Excel.

Grid-based heatmap analysis (crop_heat_grid.py)

Divides cropped images into a grid.

Detects blue-stained regions (dye presence).

Outputs binary maps, heatmaps, and CSV tables of pixel counts.

Fingering analysis (fingering.py)

Detects and quantifies preferential flow “fingers.”

Calculates finger width and depth in mm.

Produces heatmaps (fraction, intensity, area) and summary plots.

Exports detailed CSV metrics.

Experiment comparison tools

compare_fingers.py: Compare finger evolution (count, depth, width) across selected experiments.

comparing_fingers.py: Aggregates and visualizes average metrics across multiple runs (line plots, histograms, violin plots).

🛠 Requirements

Python 3.8+

Packages: numpy, pandas, matplotlib, seaborn, opencv-python, scipy

Install dependencies:

pip install numpy pandas matplotlib seaborn opencv-python scipy

🚀 Usage

Place experiment images in separate folders under a parent directory.

Run the relevant script, e.g.:

python fingering.py


Follow on-screen instructions for cropping the region of interest.

Outputs (CSV tables, plots, heatmaps) are saved in the experiment’s cropped_output/ subfolder.

📂 Repository structure
fingering.py              # Finger detection and metrics
compare_fingers.py        # Compare experiments (finger count/depth/width)
comparing_fingers.py      # Aggregate stats across experiments
crop_heat_grid.py         # Grid-based heatmaps of dye spread
multiple_experiments_area.py # Cropping + area analysis in mm²
multiple_exps.py          # Cropping + area analysis in pixels

📖 Context

These tools were developed as part of experimental hydrology research at the Zuckerberg Institute for Water Research (Ben-Gurion University).
They support visualization and quantification of contaminant flow through fractured chalk under unsaturated and saturated conditions.
