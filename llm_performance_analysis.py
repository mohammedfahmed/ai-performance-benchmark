import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load CSV data
data = pd.read_csv('ollama_process_usage.csv')  # Replace with your actual CSV filename

# Step 2: Create the results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Step 3: Group by 'LLM Model' for analysis
llm_groups = data.groupby('LLM Model')

# Step 4: Analyze CPU and Memory usage for each LLM
for model, group in llm_groups:
    # Step 5: Plot CPU usage
    plt.figure(figsize=(10, 6))
    plt.plot(group['CPU (%)'], marker='o', linestyle='-', color='b')
    plt.title(f'CPU Usage for {model}')
    plt.xlabel('Index')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{model}_cpu_usage.png')
    plt.close()

    # Step 6: Plot Memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(group['Memory (GB)'], marker='o', linestyle='-', color='g')
    plt.title(f'Memory Usage for {model}')
    plt.xlabel('Index')
    plt.ylabel('Memory Usage (GB)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{model}_memory_usage.png')
    plt.close()

print("Plots have been saved in the 'results/' folder.")
