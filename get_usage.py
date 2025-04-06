import psutil
import time

def get_ollama_processes_usage():
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            # Check if the process name contains "ollama"
            if 'ollama' in proc.info['name'].lower():
                pid = proc.info['pid']
                name = proc.info['name']
                cpu_percent = proc.info['cpu_percent']
                memory_info = proc.info['memory_info']
                memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
                print(f"(PID: {pid}) | CPU: {cpu_percent}% | Memory: {memory_usage:.2f} GB")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle processes that might terminate or have restricted access
            continue

def monitor_ollama_usage():
    while True:
        get_ollama_processes_usage()
        time.sleep(1)  # Wait for 1 second before checking again

if __name__ == '__main__':
    monitor_ollama_usage()
