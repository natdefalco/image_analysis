# Full image analysis pipeline with calibration and physical units

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Global variables for drawing rectangles (##defining the cropping image)
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1
scale = 1.0

# Function to draw a rectangle and capture coordinates (##instruction how to crop)
def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, display_image, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_image = display_image.copy()
            cv2.rectangle(temp_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        print(f"Selected Coordinates (scaled): Start ({x_start}, {y_start}), End ({x_end}, {y_end})")
        print("Rectangle selection complete. Please return to the terminal and press enter to proceed.")
        input()

# Function to process and crop images in a folder (##image cropping in folder)
def crop_images_in_folder(folder_path):
    global x_start, y_start, x_end, y_end, scale

    cropped_folder = os.path.join(folder_path, "cropped_output")
    os.makedirs(cropped_folder, exist_ok=True)

    original_x_start = int(x_start / scale)
    original_y_start = int(y_start / scale)
    original_x_end = int(x_end / scale)
    original_y_end = int(y_end / scale)

    # Estimate physical dimensions of cropped area based on 300 dpi
    dpi = 300 ##assuming 300 dpi according with previous pictures. to be changed accordingly
    mm_per_pixel = 25.4 / dpi

    width_px = original_x_end - original_x_start
    height_px = original_y_end - original_y_start

    width_mm = width_px * mm_per_pixel
    height_mm = height_px * mm_per_pixel
    area_mm2 = width_mm * height_mm

    print(f"\nCropped region dimensions: {width_px} x {height_px} pixels")
    print(f"Approximate physical size: {width_mm:.2f} mm x {height_mm:.2f} mm")
    print(f"Total cropped area: {area_mm2:.2f} mm^2\n")

    for file_name in os.listdir(folder_path):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(cropped_folder, file_name)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load {file_name}. Skipping.")
                continue

            height, width = image.shape[:2]
            if original_x_end > width or original_y_end > height:
                print(f"Error: Coordinates exceed dimensions for {file_name}. Skipping.")
                continue

            cropped_image = image[original_y_start:original_y_end, original_x_start:original_x_end]
            cv2.imwrite(output_path, cropped_image)
            print(f"Cropped and saved: {file_name}")

    return cropped_folder

# Function to analyze cropped images for blue area (calibrated to mm^2)
def analyze_cropped_images(cropped_folder):
    lower_color = np.array([90, 100, 50]) ##this can be changed in different tone of blue, pls refer to
    upper_color = np.array([140, 255, 255])

    # Calibration based on DPI
    dpi = 300
    mm_per_pixel = 25.4 / dpi
    area_per_pixel_mm2 = mm_per_pixel ** 2

    areas_mm2 = []

    for file_name in sorted(os.listdir(cropped_folder)):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(cropped_folder, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {file_name}. Skipping.")
                areas_mm2.append(np.nan)
                continue

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, lower_color, upper_color)

            area_pixels = cv2.countNonZero(mask)
            area_mm2 = area_pixels * area_per_pixel_mm2
            areas_mm2.append(area_mm2)

    return areas_mm2

# Function to save results to the cropped_output folder
def save_results_to_cropped_folder(experiment_data, cropped_folder):
    try:
        os.makedirs(cropped_folder, exist_ok=True)

        results_path = os.path.join(cropped_folder, "results.xlsx")
        df = pd.DataFrame({'Image Index': range(1, len(experiment_data) + 1), 'Area (mm²)': experiment_data})

        df.to_excel(results_path, index=False)
        print(f"\nResults successfully saved at: {results_path}")
    except PermissionError:
        print("\nPermission Error: Cannot save the .xlsx file. Make sure the file isn't open in another program.")
    except Exception as e:
        print(f"\nError while saving .xlsx file: {e}")

# Function to display and select multiple folders
def select_folders(parent_directory):
    folders = [f for f in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, f))]
    if not folders:
        print("No folders found in the directory!")
        return []

    print("\nAvailable Folders:")
    for idx, folder in enumerate(folders, start=1):
        print(f"{idx}. {folder}")

    while True:
        try:
            choices = input("\nEnter the numbers of folders you want to process (e.g., 1,3,5): ")
            selected_indices = [int(c.strip()) - 1 for c in choices.split(",") if c.strip().isdigit()]
            selected_folders = [os.path.join(parent_directory, folders[i]) for i in selected_indices if 0 <= i < len(folders)]

            if selected_folders:
                print("\nYou selected:")
                for folder in selected_folders:
                    print(f"- {os.path.basename(folder)}")
                return selected_folders
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter valid folder numbers separated by commas.")

# Main function
def process_selected_folders(parent_directory):
    global display_image, scale

    selected_folders = select_folders(parent_directory)
    if not selected_folders:
        print("No folders selected. Exiting.")
        return

    experiment_data = {}

    for folder_path in selected_folders:
        folder_name = os.path.basename(folder_path)
        print(f"\nProcessing Folder: {folder_name}")
        print("Loading an image for rectangle selection...")

        example_image_path = next(
            (os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))),
            None
        )

        image = cv2.imread(example_image_path)
        max_dim = 800
        height, width = image.shape[:2]
        scale = max_dim / max(height, width) if max(height, width) > max_dim else 1.0
        display_image = cv2.resize(image, (int(width * scale), int(height * scale))) if scale < 1.0 else image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_rectangle)
        cv2.imshow("Image", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cropped_folder = crop_images_in_folder(folder_path)
        areas_mm2 = analyze_cropped_images(cropped_folder)
        experiment_data[folder_name] = areas_mm2
        save_results_to_cropped_folder(areas_mm2, cropped_folder)

    # Plot combined results
    for exp_name, areas_mm2 in experiment_data.items():
        plt.plot(range(len(areas_mm2)), areas_mm2, marker='o', label=exp_name)

    plt.title("Blue Area Over Time Across Selected Folders")
    plt.xlabel("Image Index")
    plt.ylabel("Area (mm²)")
    plt.legend()
    plt.show()

# Run the script
if __name__ == "__main__":
    parent_directory = r"C:\\Users\\defalco\\OneDrive - BGU\\Brilliant Blue Exp Pic"
    process_selected_folders(parent_directory)
