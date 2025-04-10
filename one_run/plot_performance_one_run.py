import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('ollama_process_usage.csv')

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

if not os.path.exists('results'):
    os.makedirs('results')

data['Relative Time (s)'] = 0

# Iterate over each model group and calculate the start and end times
for model, group in data.groupby('LLM Model'):
    group = group.sort_values(by='Timestamp')
    start_time = group['Timestamp'].min()
    end_time = group['Timestamp'].max()
    
    # Calculate relative start and end times (in seconds)
    relative_start_time = (start_time - start_time).total_seconds()
    relative_end_time = (end_time - start_time).total_seconds()
    
    print(f"Model: {model} - Start Time: {start_time} - End Time: {end_time}")
    print(f"Model: {model} - Relative Start Time: {relative_start_time} seconds - Relative End Time: {relative_end_time} seconds")
    
    data.loc[data['LLM Model'] == model, 'Relative Time (s)'] = (data['Timestamp'] - start_time).dt.total_seconds()

# Plot CPU usage
plt.figure(figsize=(10, 6))
for model, group in data.groupby('LLM Model'):
    plt.plot(group['Relative Time (s)'], group['CPU (%)'], marker='o', label=model)
plt.tight_layout()
plt.savefig('results/all_models_cpu_usage_relative_time.png')
plt.close()

# Plot Memory usage
plt.figure(figsize=(10, 6))
for model, group in data.groupby('LLM Model'):
    plt.plot(group['Relative Time (s)'], group['Memory (GB)'], marker='o', label=model)
plt.tight_layout()
plt.savefig('results/all_models_memory_usage_relative_time.png')
plt.close()

print("Combined plots with relative time axes have been saved in the 'results/' folder.")
