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
    # Sort the group by Timestamp to ensure correct order
    group = group.sort_values(by='Timestamp')
    
    # Get the first (minimum) timestamp for the current model
    start_time = group['Timestamp'].min()  # Use min() to get the first timestamp for the model
    
    # Update the relative time for the current model
    data.loc[data['LLM Model'] == model, 'Relative Time (s)'] = (data['Timestamp'] - start_time).dt.total_seconds()

# Step 4: Combine CPU usage for all models in one plot with relative time on the x-axis
plt.figure(figsize=(10, 6))
for model, group in data.groupby('LLM Model'):
    plt.plot(group['Relative Time (s)'], group['CPU (%)'], marker='o', label=model)

# Add stats to CPU plot
cpu_mean = data['CPU (%)'].mean()
cpu_min = data['CPU (%)'].min()
cpu_max = data['CPU (%)'].max()
plt.title('CPU Usage for All LLM Models')
plt.xlabel('Relative Time (s)')
plt.ylabel('CPU Usage (%)')
plt.grid(True)
plt.legend(title='LLM Models')

# Display statistics on the plot
plt.text(0.5, 0.9, f'Mean: {cpu_mean:.2f}%\nMin: {cpu_min:.2f}%\nMax: {cpu_max:.2f}%', 
         transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')

plt.tight_layout()
plt.savefig('results/all_models_cpu_usage_relative_time.png')
plt.close()

# Step 5: Combine Memory usage for all models in one plot with relative time on the x-axis
plt.figure(figsize=(10, 6))
for model, group in data.groupby('LLM Model'):
    plt.plot(group['Relative Time (s)'], group['Memory (GB)'], marker='o', label=model)

# Add stats to Memory plot
mem_mean = data['Memory (GB)'].mean()
mem_min = data['Memory (GB)'].min()
mem_max = data['Memory (GB)'].max()
plt.title('Memory Usage for All LLM Models')
plt.xlabel('Relative Time (s)')
plt.ylabel('Memory Usage (GB)')
plt.grid(True)
plt.legend(title='LLM Models')

# Display statistics on the plot
plt.text(0.5, 0.9, f'Mean: {mem_mean:.2f} GB\nMin: {mem_min:.2f} GB\nMax: {mem_max:.2f} GB', 
         transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')

plt.tight_layout()
plt.savefig('results/all_models_memory_usage_relative_time.png')
plt.close()

print("Combined plots with relative time axes have been saved in the 'results/' folder.")
