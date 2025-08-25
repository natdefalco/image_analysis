# compare_fingers.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Set path to parent directory containing multiple experiment folders
parent_directory = r"C:\Users\defalco\OneDrive - BGU\Brilliant Blue Exp Pic"
time_interval_minutes = 1  # used to normalize time if needed

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

# --- Plot Finger Count Over Time ---
finger_counts = df_all.groupby(["Experiment", "Time (minutes)"])["Finger ID"].nunique().reset_index(name="Finger Count")
plt.figure(figsize=(10, 6))
sns.lineplot(data=finger_counts, x="Time (minutes)", y="Finger Count", hue="Experiment", marker="o")
plt.title("Finger Count Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Max Depth Over Time ---
max_depths = df_all.groupby(["Experiment", "Time (minutes)"])["Depth (mm)"].max().reset_index(name="Max Depth (mm)")
plt.figure(figsize=(10, 6))
sns.lineplot(data=max_depths, x="Time (minutes)", y="Max Depth (mm)", hue="Experiment", marker="o")
plt.title("Max Finger Depth Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Average Finger Count with Error Bars ---
finger_stats = df_all.groupby(["Experiment", "Time (minutes)"])["Finger ID"] \
    .agg(["nunique"]).reset_index().rename(columns={"nunique": "Finger Count"})

finger_mean_std = finger_stats.groupby(["Experiment", "Time (minutes)"])["Finger Count"] \
    .agg(["mean", "std"]).reset_index()

plt.figure(figsize=(10, 6))
for experiment in finger_mean_std["Experiment"].unique():
    data = finger_mean_std[finger_mean_std["Experiment"] == experiment]
    plt.errorbar(data["Time (minutes)"], data["mean"], yerr=data["std"],
                 label=experiment, capsize=4, marker='o')
plt.title("Avg. Finger Count Over Time with Std Dev")
plt.xlabel("Time (minutes)")
plt.ylabel("Number of Fingers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Average Max Depth with Error Bars ---
depth_stats = df_all.groupby(["Experiment", "Time (minutes)"])["Depth (mm)"] \
    .agg(["max"]).reset_index().rename(columns={"max": "Max Depth"})

depth_mean_std = depth_stats.groupby(["Experiment", "Time (minutes)"])["Max Depth"] \
    .agg(["mean", "std"]).reset_index()

plt.figure(figsize=(10, 6))
for experiment in depth_mean_std["Experiment"].unique():
    data = depth_mean_std[depth_mean_std["Experiment"] == experiment]
    plt.errorbar(data["Time (minutes)"], data["mean"], yerr=data["std"],
                 label=experiment, capsize=4, marker='o')
plt.title("Avg. Max Finger Depth Over Time with Std Dev")
plt.xlabel("Time (minutes)")
plt.ylabel("Max Finger Depth (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Boxplot of Final Finger Width and Depth ---
final_time = df_all.groupby(["Experiment"])["Time (minutes)"].max().reset_index()
df_final = pd.merge(df_all, final_time, on=["Experiment", "Time (minutes)"])

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final, x="Experiment", y="Width (mm)")
plt.title("Final Finger Width Distribution (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final, x="Experiment", y="Depth (mm)")
plt.title("Final Finger Depth Distribution (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()
