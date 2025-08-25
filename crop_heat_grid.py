import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration ---

# Image and folder paths
input_folder = r"C:\Users\defalco\OneDrive - BGU\Brilliant Blue Exp Pic\21"
output_folder = os.path.join(input_folder, "cropped_output")
os.makedirs(output_folder, exist_ok=True)

# Real-world core dimensions (adjust based on physical measurement)
core_width_mm = 50    # mm to be change!!!
core_height_mm = 75   # mm to be changed !!!!!

# Grid resolution
grid_rows = 50
grid_cols = 50

# Time between each picture (min)
time_interval_minutes = 5 # to be change!!!!!

# --- Global variables ---
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1
scale = 1.0

# --- Mouse drawing function ---
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
        cropped_width_px = abs(x_end - x_start)
        cropped_height_px = abs(y_end - y_start)
        print(f"Cropped dimensions (pixels): width = {cropped_width_px}, height = {cropped_height_px}")

# --- Blue detection grid analysis ---
def analyze_grid_binary(image, grid_size=(grid_rows, grid_cols)):
    height, width = image.shape[:2]
    cell_h, cell_w = height // grid_size[0], width // grid_size[1]

    lower_blue = np.array([85, 50, 40])
    upper_blue = np.array([145, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    grid_data = []
    binary_map = np.zeros(grid_size, dtype=int)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x1, y1 = j * cell_w, i * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cell = mask[y1:y2, x1:x2]
            blue_pixels = cv2.countNonZero(cell)
            threshold = 0.05

            binary = 1 if blue_pixels > (cell_h * cell_w * threshold) else 0
            grid_data.append([i, j, blue_pixels, binary])
            binary_map[i, j] = binary

    return grid_data, binary_map

# --- Generate heatmap plot ---
def generate_binary_map(binary_map, image_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(binary_map, cmap="Blues", cbar=False, annot=True, fmt="d")
    plt.title(f"Blue Detection Map: {image_name}")
    plt.xlabel("Grid Columns")
    plt.ylabel("Grid Rows")
    save_path = os.path.join(output_folder, f"binary_map_{image_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Binary map saved: {save_path}")

# --- Batch process all images ---
def batch_crop_images():
    global x_start, y_start, x_end, y_end, scale

    original_x_start = int(x_start / scale)
    original_y_start = int(y_start / scale)
    original_x_end = int(x_end / scale)
    original_y_end = int(y_end / scale)

    all_data = []

    for idx, file_name in enumerate(sorted(os.listdir(input_folder))):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load {file_name}. Skipping.")
                continue

            cropped_image = image[original_y_start:original_y_end, original_x_start:original_x_end]
            cv2.imwrite(output_path, cropped_image)

            cropped_height, cropped_width = cropped_image.shape[:2]
            mm_per_pixel_x = core_width_mm / cropped_width
            mm_per_pixel_y = core_height_mm / cropped_height
            print(f"{file_name}: Scale = {mm_per_pixel_x:.3f} mm/px (X), {mm_per_pixel_y:.3f} mm/px (Y)")

            grid_data, binary_map = analyze_grid_binary(cropped_image)

            time_min = idx * time_interval_minutes
            for row in grid_data:
                row.insert(0, time_min)
                row.insert(0, file_name)
            all_data.extend(grid_data)

            generate_binary_map(binary_map, file_name)

    df = pd.DataFrame(all_data, columns=["Image", "Time (min)", "Row", "Column", "Blue Pixels", "Blue Detected (0/1)"])
    save_path = os.path.join(output_folder, "grid_blue_detection.csv")
    df.to_csv(save_path, index=False)
    print(f"Grid analysis saved to {save_path}")

# --- Main execution ---
if __name__ == "__main__":
    print("Starting script...")
    if not os.path.exists(input_folder) or not os.listdir(input_folder):
        print(f"Error: Input folder missing or empty: {input_folder}")
        exit()

    example_image_path = next(
        (os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png"))),
        None
    )
    if example_image_path is None:
        print("Error: No valid images found.")
        exit()

    image = cv2.imread(example_image_path)
    if image is None:
        print("Error: Failed to load example image.")
        exit()

    max_dim = 800
    height, width = image.shape[:2]
    scale = max_dim / max(height, width) if max(height, width) > max_dim else 1.0
    display_image = cv2.resize(image, (int(width * scale), int(height * scale)))

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)
    cv2.imshow("Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    batch_crop_images()
