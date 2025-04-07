import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load CSV data
data = pd.read_csv('ollama_process_usage.csv')  # Replace with your actual CSV filename

# Convert the Timestamp column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Step 2: Create the results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Step 3: Calculate the relative time for each 'LLM Model' based on its own start time
data['Relative Time (s)'] = 0  # Initialize the new column for relative time

# Group by 'LLM Model' and compute the relative time for each group
for model, group in data.groupby('LLM Model'):
    # Get the first (minimum) timestamp for the current model
    start_time = group['Timestamp'].min()  # Use min() to get the first timestamp for the model
    data.loc[data['LLM Model'] == model, 'Relative Time (s)'] = (data['Timestamp'] - start_time).dt.total_seconds()

# Step 4: Create subplots for CPU usage and Memory usage for each model
models = data['LLM Model'].unique()  # Get the unique models

# Create a figure with subplots for CPU and Memory usage
fig, axes = plt.subplots(len(models), 2, figsize=(12, len(models) * 6))  # 2 columns for CPU and Memory

# Loop through each model to create its subplot for CPU and Memory
for i, model in enumerate(models):
    # CPU plot
    ax1 = axes[i, 0] if len(models) > 1 else axes[0]
    group = data[data['LLM Model'] == model]
    ax1.plot(group['Relative Time (s)'], group['CPU (%)'], marker='o', label=model)
    ax1.set_title(f'CPU Usage for {model}')
    ax1.set_xlabel('Relative Time (s)')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.grid(True)
    ax1.legend(title='LLM Model')
    
    # Memory plot
    ax2 = axes[i, 1] if len(models) > 1 else axes[1]
    ax2.plot(group['Relative Time (s)'], group['Memory (GB)'], marker='o', label=model)
    ax2.set_title(f'Memory Usage for {model}')
    ax2.set_xlabel('Relative Time (s)')
    ax2.set_ylabel('Memory Usage (GB)')
    ax2.grid(True)
    ax2.legend(title='LLM Model')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('results/all_models_performance_subplots.png')
plt.close()

print("Subplots for CPU and Memory usage have been saved in the 'results/' folder.")
