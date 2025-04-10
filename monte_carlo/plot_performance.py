import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
data = pd.read_csv('ollama_process_usage.csv')

# Convert the 'Timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Create a directory for saving the plots if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Add a column for relative time (in seconds)
data['Relative Time (s)'] = 0

# Iterate over each model group and calculate the relative time
for model, group in data.groupby('LLM Model'):
    group = group.sort_values(by='Timestamp')
    start_time = group['Timestamp'].min()
    
    # Calculate relative time (in seconds)
    data.loc[data['LLM Model'] == model, 'Relative Time (s)'] = (data['Timestamp'] - start_time).dt.total_seconds()

# Group by LLM Model and Prompt Index and calculate the mean for each group
grouped_data = data.groupby(['LLM Model', 'Prompt Index']).agg(
    avg_cpu=('CPU (%)', 'mean'),
    avg_memory=('Memory (GB)', 'mean'),
    avg_relative_time=('Relative Time (s)', 'mean')
).reset_index()

# Plot CPU usage averaged over the Prompt Index
plt.figure(figsize=(10, 6))
for model, group in grouped_data.groupby('LLM Model'):
    plt.plot(group['avg_relative_time'], group['avg_cpu'], marker='o', label=model)
plt.xlabel('Relative Time (s)')
plt.ylabel('Average CPU (%)')
plt.title('Average CPU Usage Over Time (Averaged by Prompt Index)')
plt.legend()
plt.tight_layout()
plt.savefig('results/all_models_avg_cpu_usage_relative_time.png')
plt.close()

# Plot Memory usage averaged over the Prompt Index
plt.figure(figsize=(10, 6))
for model, group in grouped_data.groupby('LLM Model'):
    plt.plot(group['avg_relative_time'], group['avg_memory'], marker='o', label=model)
plt.xlabel('Relative Time (s)')
plt.ylabel('Average Memory (GB)')
plt.title('Average Memory Usage Over Time (Averaged by Prompt Index)')
plt.legend()
plt.tight_layout()
plt.savefig('results/all_models_avg_memory_usage_relative_time.png')
plt.close()

print("Combined plots with averaged values over Prompt Index have been saved in the 'results/' folder.")
