# --- 1. Configuration and Global Settings ---
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label, find_objects

# Paths and constants
input_folder = r"C:\Users\natde\OneDrive - BGU\Brilliant Blue Exp Pic\26" #change accordingly
output_folder = os.path.join(input_folder, "cropped_output")
subfolders = {
    "cropped": os.path.join(output_folder, "cropped_images"),
    "heatmap_fraction": os.path.join(output_folder, "heatmaps", "fraction"),
    "heatmap_intensity": os.path.join(output_folder, "heatmaps", "intensity"),
    "heatmap_area": os.path.join(output_folder, "heatmaps", "area"),
    "csv": os.path.join(output_folder, "csv"),
    "figures": os.path.join(output_folder, "figures")
}
for folder in subfolders.values():
    os.makedirs(folder, exist_ok=True)



core_width_mm = 50 #change accordingly
core_height_mm = 75  #change accordingly

grid_rows = 48 #change accordingly
grid_cols = 48 #change accordingly
time_interval_minutes = 1 #change accordingly

# Derived area per pixel per cell:
# total core area (mm²) = width * height
# each grid cell = core divided into 48x48 regions
# so area per cell = (core_width / cols) * (core_height / rows)
area_per_cell_mm2 = (core_width_mm / grid_cols) * (core_height_mm / grid_rows)
mm_per_col = core_width_mm / grid_cols
mm_per_row = core_height_mm / grid_rows

drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1
scale = 1.0


# --- 2. Utility Functions ---
def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, display_image, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_image = display_image.copy()
        cv2.rectangle(temp_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        print(f"Selected Coordinates (scaled): Start ({x_start}, {y_start}), End ({x_end}, {y_end})")

def get_scaled_coordinates():
    return int(x_start / scale), int(y_start / scale), int(x_end / scale), int(y_end / scale)

# --- 3. Core Processing Functions ---
def crop_image(image, coords):
    x1, y1, x2, y2 = coords
    return image[y1:y2, x1:x2]

def analyze_grid_fraction(image, grid_size):
    height, width = image.shape[:2]
    cell_h, cell_w = height // grid_size[0], width // grid_size[1]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 50, 40]), np.array([145, 255, 255]))
    intensity = hsv[..., 2]

    fraction_map = np.zeros(grid_size, dtype=float)
    intensity_map = np.zeros(grid_size, dtype=float)
    area_map = np.zeros(grid_size, dtype=float)
    grid_data = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x1, y1 = j * cell_w, i * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cell_mask = mask[y1:y2, x1:x2]
            cell_intensity = intensity[y1:y2, x1:x2]

            blue_pixels = cv2.countNonZero(cell_mask)
            total_pixels = cell_h * cell_w
            fraction = blue_pixels / total_pixels
            area = blue_pixels * (area_per_cell_mm2 / total_pixels)

            avg_intensity = np.mean(cell_intensity[cell_mask > 0]) if blue_pixels > 0 else 0.0

            fraction_map[i, j] = fraction
            intensity_map[i, j] = avg_intensity
            area_map[i, j] = area

            grid_data.append([i, j, blue_pixels, fraction, area, avg_intensity])

    return grid_data, fraction_map, intensity_map, area_map

def detect_fingers(fraction_map, threshold=0.05):
    binary_map = (fraction_map > threshold).astype(int)
    labeled, num = label(binary_map)
    return labeled, num

def summarize_fingers(labeled):
    results = []
    for i, slc in enumerate(find_objects(labeled)):
        if slc is None:
            continue
        depth = slc[0].stop - slc[0].start
        width = slc[1].stop - slc[1].start
        results.append({
            'FingerID': i + 1,
            'Width': width,
            'Depth': depth,
            'Width_mm': width * mm_per_col,
            'Depth_mm': depth * mm_per_row
        })
    return results

def plot_heatmap(data, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Grid Columns")
    plt.ylabel("Grid Rows")
    plt.savefig(path)
    plt.close()

def process_images():
    global scale
    all_grid_data = []
    all_finger_data = []
    x1, y1, x2, y2 = get_scaled_coordinates()

    for idx, file_name in enumerate(sorted(os.listdir(input_folder))):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(input_folder, file_name)
            img = cv2.imread(path)
            if img is None:
                continue

            cropped = crop_image(img, (x1, y1, x2, y2))
            cv2.imwrite(os.path.join(subfolders["cropped"], file_name), cropped)

            grid_data, fraction_map, intensity_map, area_map = analyze_grid_fraction(cropped, (grid_rows, grid_cols))
            labeled, _ = detect_fingers(fraction_map)
            summary = summarize_fingers(labeled)

            time_min = idx * time_interval_minutes
            for row in grid_data:
                all_grid_data.append([file_name, time_min] + row)
            for row in summary:
                all_finger_data.append([file_name, time_min] + list(row.values()))

            plot_heatmap(fraction_map, f"Fraction Map: {file_name}", os.path.join(subfolders["heatmap_fraction"], f"fraction_{file_name}.png"))
            plot_heatmap(intensity_map, f"Intensity Map: {file_name}", os.path.join(subfolders["heatmap_intensity"], f"intensity_{file_name}.png"))
            plot_heatmap(area_map, f"Blue Area Map (mm²): {file_name}", os.path.join(subfolders["heatmap_area"], f"area_{file_name}.png"))

    pd.DataFrame(
        all_grid_data,
        columns=["Image Filename", "Time (minutes)", "Row Index", "Column Index", "Blue Pixels (count)", "Blue Fraction (0-1)", "Area (mm²)", "Intensity (value channel)"]
    ).to_csv(os.path.join(subfolders["csv"], "grid_fraction_data.csv"), index=False)

    pd.DataFrame(
        all_finger_data,
        columns=["Image Filename", "Time (minutes)", "Finger ID", "Width (grid cols)", "Depth (grid rows)", "Width (mm)", "Depth (mm)"]
    ).to_csv(os.path.join(subfolders["csv"], "finger_metrics.csv"), index=False)

    # --- 6. Visualization and Finger Metrics Summary Plots ---
    df = pd.read_csv(os.path.join(subfolders["csv"], "finger_metrics.csv"))
    finger_count = df.groupby("Time (minutes)")["Finger ID"].nunique().reset_index(name="Finger Count")
    max_depth = df.groupby("Time (minutes)")["Depth (mm)"].max().reset_index(name="Max Depth (mm)")
    final_time = df["Time (minutes)"].max()
    final_fingers = df[df["Time (minutes)"] == final_time][["Width (mm)", "Depth (mm)"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.lineplot(data=finger_count, x="Time (minutes)", y="Finger Count", marker="o", ax=axes[0])
    axes[0].set_title("Finger Count Over Time")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Number of Fingers")
    axes[0].grid(True)

    sns.lineplot(data=max_depth, x="Time (minutes)", y="Max Depth (mm)", marker="o", ax=axes[1])
    axes[1].set_title("Max Finger Depth Over Time")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Depth (mm)")
    axes[1].grid(True)

    sns.boxplot(data=final_fingers, ax=axes[2])
    axes[2].set_title("Finger Width and Depth at Final Time")
    axes[2].set_ylabel("Dimension (mm)")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(subfolders["figures"], "summary_finger_plots.png"))
    plt.close(fig)

# --- 5. Main Execution ---
if __name__ == "__main__":
    example_image_path = next(
        (os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png"))),
        None
    )
    if not example_image_path:
        raise FileNotFoundError("No image found in input folder")

    image = cv2.imread(example_image_path)
    max_dim = 800
    h, w = image.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    display_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)
    cv2.imshow("Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    process_images()
