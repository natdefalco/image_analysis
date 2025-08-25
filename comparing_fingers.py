# compare_fingers.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
parent_directory = r"C:\Users\natde\OneDrive - BGU\Brilliant Blue Exp Pic"
time_interval_minutes = 1

# --- Select experiment folders ---
folders = [
    f for f in os.listdir(parent_directory)
    if os.path.isdir(os.path.join(parent_directory, f)) and os.path.exists(
        os.path.join(parent_directory, f, "cropped_output", "csv", "finger_metrics.csv")
    )
]

print("Found folders:")
for i, folder in enumerate(folders):
    print(f"{i + 1}: {folder}")

selected = input("Enter folder numbers to compare (e.g., 1,3,5): ")
indices = [int(x.strip()) - 1 for x in selected.split(",") if x.strip().isdigit()]
selected_folders = [folders[i] for i in indices if 0 <= i < len(folders)]

if not selected_folders:
    print("No valid folders selected. Exiting.")
    exit()

# --- Load and merge data ---
all_data = []

for folder in selected_folders:
    file_path = os.path.join(parent_directory, folder, "cropped_output", "csv", "finger_metrics.csv")
    df = pd.read_csv(file_path)
    df["Experiment"] = folder
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# --- Average Finger Count Over Time ---
finger_count = df_all.groupby(["Experiment", "Time (minutes)"])["Finger ID"].nunique().reset_index(name="Finger Count")
summary_count = finger_count.groupby("Time (minutes)")["Finger Count"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(10, 6))
plt.errorbar(summary_count["Time (minutes)"], summary_count["mean"], yerr=summary_count["std"],
             capsize=4, marker='o', linestyle='-', color='tab:blue')
plt.title("Average Finger Count Over Time (All Experiments)")
plt.xlabel("Time (minutes)")
plt.ylabel("Mean Finger Count ± STD")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Average Max Depth Over Time ---
max_depths = df_all.groupby(["Experiment", "Time (minutes)"])["Depth (mm)"].max().reset_index(name="Max Depth")
depth_summary = max_depths.groupby("Time (minutes)")["Max Depth"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(10, 6))
plt.errorbar(depth_summary["Time (minutes)"], depth_summary["mean"], yerr=depth_summary["std"],
             capsize=4, marker='o', linestyle='-', color='tab:green')
plt.title("Average Max Finger Depth Over Time (All Experiments)")
plt.xlabel("Time (minutes)")
plt.ylabel("Mean Max Depth (mm) ± STD")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Histogram of Finger Widths and Depths ---
plt.figure(figsize=(10, 5))
sns.histplot(df_all["Width (mm)"], kde=True, bins=30)
plt.title("Histogram of Finger Widths (mm)")
plt.xlabel("Width (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_all["Depth (mm)"], kde=True, bins=30)
plt.title("Histogram of Finger Depths (mm)")
plt.xlabel("Depth (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Violin plots for width and depth ---
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_all, x="Experiment", y="Width (mm)", inner="quartile")
plt.title("Violin Plot of Finger Width (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df_all, x="Experiment", y="Depth (mm)", inner="quartile")
plt.title("Violin Plot of Finger Depth (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()
