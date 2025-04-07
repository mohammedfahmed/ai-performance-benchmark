import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('ollama_process_usage.csv') 

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

if not os.path.exists('results'):
    os.makedirs('results')

data['Relative Time (s)'] = 0  

for model, group in data.groupby('LLM Model'):
    group = group.sort_values(by='Timestamp')
    start_time = group['Timestamp'].min()  
    print(start_time)
    data.loc[data['LLM Model'] == model, 'Relative Time (s)'] = (data['Timestamp'] - start_time).dt.total_seconds()

print(data)


plt.figure(figsize=(10, 6))

for model, group in data.groupby('LLM Model'):
    plt.plot(group['Relative Time (s)'], group['CPU (%)'], marker='o', label=model)

# # Add stats to CPU plot
# cpu_mean = data['CPU (%)'].mean()
# cpu_min = data['CPU (%)'].min()
# cpu_max = data['CPU (%)'].max()
# plt.title('CPU Usage for All LLM Models')
# plt.xlabel('Relative Time (s)')
# plt.ylabel('CPU Usage (%)')
# plt.grid(True)
# plt.legend(title='LLM Models')

# # Display statistics on the plot
# plt.text(0.5, 0.9, f'Mean: {cpu_mean:.2f}%\nMin: {cpu_min:.2f}%\nMax: {cpu_max:.2f}%', 
         transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')

plt.tight_layout()
plt.savefig('results/all_models_cpu_usage_relative_time.png')
plt.close()

plt.figure(figsize=(10, 6))

for model, group in data.groupby('LLM Model'):
    plt.plot(group['Relative Time (s)'], group['Memory (GB)'], marker='o', label=model)

# # Add stats to Memory plot
# mem_mean = data['Memory (GB)'].mean()
# mem_min = data['Memory (GB)'].min()
# mem_max = data['Memory (GB)'].max()
# plt.title('Memory Usage for All LLM Models')
# plt.xlabel('Relative Time (s)')
# plt.ylabel('Memory Usage (GB)')
# plt.grid(True)
# plt.legend(title='LLM Models')

# # Display statistics on the plot
# plt.text(0.5, 0.9, f'Mean: {mem_mean:.2f} GB\nMin: {mem_min:.2f} GB\nMax: {mem_max:.2f} GB', 
#          transform=plt.gca().transAxes, fontsize=12, va='top', ha='center')

plt.tight_layout()
plt.savefig('results/all_models_memory_usage_relative_time.png')
plt.close()

print("Combined plots with relative time axes have been saved in the 'results/' folder.")
