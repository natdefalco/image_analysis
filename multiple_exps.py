import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Global variables for drawing rectangles5,6
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1
scale = 1.0


# Function to draw a rectangle and capture coordinates
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
        print(" Rectangle selection complete. Please return to the terminal and press enter to proceed.")
        input()  # Pause until the user presses Enter


# Function to process and crop images in a folder
def crop_images_in_folder(folder_path):
    global x_start, y_start, x_end, y_end, scale

    cropped_folder = os.path.join(folder_path, "cropped_output")
    os.makedirs(cropped_folder, exist_ok=True)

    original_x_start = int(x_start / scale)
    original_y_start = int(y_start / scale)
    original_x_end = int(x_end / scale)
    original_y_end = int(y_end / scale)

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


# Function to analyze cropped images for blue area
def analyze_cropped_images(cropped_folder):
    lower_color = np.array([90, 100, 50])  # Adjust as needed
    upper_color = np.array([140, 255, 255])  # Adjust as needed

    areas_pixels = []

    for file_name in sorted(os.listdir(cropped_folder)):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(cropped_folder, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {file_name}. Skipping.")
                areas_pixels.append(np.nan)
                continue

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_color = np.array([90, 100, 50])
            upper_color = np.array([140, 255, 255])
            mask = cv2.inRange(hsv_img, lower_color, upper_color)

            area_pixels = cv2.countNonZero(mask)
            areas_pixels.append(area_pixels)

    return areas_pixels


# Function to save results to the cropped_output folder
def save_results_to_cropped_folder(experiment_data, cropped_folder):
    try:
        os.makedirs(cropped_folder, exist_ok=True)

        results_path = os.path.join(cropped_folder, "results.xlsx")
        df = pd.DataFrame({'Image Index': range(1, len(experiment_data) + 1), 'Area (Pixels)': experiment_data})

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
            selected_folders = [os.path.join(parent_directory, folders[i]) for i in selected_indices if
                                0 <= i < len(folders)]

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
        areas_pixels = analyze_cropped_images(cropped_folder)
        experiment_data[folder_name] = areas_pixels
        save_results_to_cropped_folder(areas_pixels, cropped_folder)

    # Plot combined results
    for exp_name, areas_pixels in experiment_data.items():
        plt.plot(range(len(areas_pixels)), areas_pixels, marker='o', label=exp_name)

    plt.title("Blue Area Over Time Across Selected Folders")
    plt.xlabel("Image Index")
    plt.ylabel("Area (Pixels)")
    plt.legend()
    plt.show()


# Run the script
if __name__ == "__main__":
    parent_directory = r"C:\Users\defalco\OneDrive - BGU\Brilliant Blue Exp Pic"
    process_selected_folders(parent_directory)
