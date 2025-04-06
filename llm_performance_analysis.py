import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load CSV data
data = pd.read_csv('ollama_process_usage.csv')  # Replace with your actual CSV filename

# Step 2: Create the results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Step 3: Calculate statistics and plot CPU and Memory usage for all models combined
# Step 4: Combine CPU usage for all models in one plot
plt.figure(figsize=(10, 6))
for model, group in data.groupby('LLM Model'):
    plt.plot(group['CPU (%)'], marker='o', label=model)

# Add stats to CPU plot
cpu_mean = data['CPU (%)'].mean()
cpu_min = data['CPU (%)'].min()
cpu_max = data['CPU (%)'].max()
plt.title('CPU Usage for All LLM Models')
plt.xlabel('Index')
plt.ylabel('CPU Usage (%)')
plt.grid(True)
plt.legend(title='LLM Models')
plt.text(0.5, 0.9, f'Mean: {cpu_mean:.2f}%\nMin: {cpu_min:.2f}%\nMax: {cpu_max:.2f}%', 
         transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')
plt.tight_layout()
plt.savefig('results/all_models_cpu_usage.png')
plt.close()

# Step 5: Combine Memory usage for all models in one plot
plt.figure(figsize=(10, 6))
for model, group in data.groupby('LLM Model'):
    plt.plot(group['Memory (GB)'], marker='o', label=model)

# Add stats to Memory plot
mem_mean = data['Memory (GB)'].mean()
mem_min = data['Memory (GB)'].min()
mem_max = data['Memory (GB)'].max()
plt.title('Memory Usage for All LLM Models')
plt.xlabel('Index')
plt.ylabel('Memory Usage (GB)')
plt.grid(True)
plt.legend(title='LLM Models')
plt.text(0.5, 0.9, f'Mean: {mem_mean:.2f} GB\nMin: {mem_min:.2f} GB\nMax: {mem_max:.2f} GB', 
         transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')
plt.tight_layout()
plt.savefig('results/all_models_memory_usage.png')
plt.close()

print("Combined plots have been saved in the 'results/' folder.")
