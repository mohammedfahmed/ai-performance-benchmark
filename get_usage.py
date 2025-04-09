import psutil
import time
import os
import csv
from datetime import datetime

def get_ollama_processes_usage(csv_writer):
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            # Check if the process name contains "ollama"
            if 'ollama' in proc.info['name'].lower():
                config_file_path = "config.txt"  # model_name=llama2, prompt_index=1

                # Initialize the variables
                llm_model = None
                prompt_index = None

                # Read the config file to extract model_name and prompt_index
                with open(config_file_path, 'r') as file:
                    for line in file:
                        line = line.strip()  # Remove leading/trailing whitespace
                        llm_model = line.split('model_name=')[1].split(',')[0].strip()
                        prompt_index = int(line.split('prompt_index=')[1].strip())  # Assuming prompt_index is an integer
                            
                        # Break the loop if both values are found
                        if llm_model and prompt_index is not None:
                            break
                            
                pid = proc.info['pid']
                name = proc.info['name']
                cpu_percent = proc.info['cpu_percent']
                memory_info = proc.info['memory_info']
                memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB

                # Capture the current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Print the data to the console
                print(f"(PID: {pid}) | CPU: {cpu_percent}% | Memory: {memory_usage:.2f} GB | LLM Model: {llm_model} | Prompt Index: {prompt_index} | Timestamp: {timestamp}")

                # Save the results to the CSV file, including the timestamp and prompt_index
                csv_writer.writerow([timestamp, pid, name, cpu_percent, memory_usage, llm_model, prompt_index])


        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle processes that might terminate or have restricted access
            continue

# Function to monitor the processes and save them to CSV
def monitor_ollama_usage():
    # Open the CSV file in append mode to save data over time
    with open('ollama_process_usage.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        
        # Write headers if the file is empty
        if file.tell() == 0:
            csv_writer.writerow(['Timestamp', 'PID', 'Name', 'CPU (%)', 'Memory (GB)', 'LLM Model'])
        
        while True:
            get_ollama_processes_usage(csv_writer)
            time.sleep(1)  # Wait for 1 second before checking again

if __name__ == '__main__':
    monitor_ollama_usage()
