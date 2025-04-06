import ollama
import time
import psutil
import threading

def track_resources(pid, interval=0.1, stop_event=None):
    """Tracks CPU and memory usage of a given process ID."""
    process = psutil.Process(pid)
    cpu_history = []
    memory_history = []
    timestamps = []

    while not stop_event or not stop_event.is_set():
        try:
            cpu_percent = process.cpu_percent()
            mem_info = process.memory_info()
            mem_rss_mb = mem_info.rss / (1024 * 1024)  # Resident Set Size in MB

            cpu_history.append(cpu_percent)
            memory_history.append(mem_rss_mb)
            timestamps.append(time.time())

            time.sleep(interval)
        except psutil.NoSuchProcess:
            print(f"Process with PID {pid} no longer exists.")
            break
        except Exception as e:
            print(f"Error tracking resources: {e}")
            break

    return timestamps, cpu_history, memory_history

def execute_ollama_chat(model_name, prompt):
    """Executes the Ollama chat and tracks its resource usage."""
    try:
        # Get the PID of the current Python process
        parent_pid = psutil.Process().pid

        # We need to find the child process spawned by the ollama.chat call.
        # This might require a short delay to allow the child process to start.
        time.sleep(0.5)

        ollama_child_process = None
        for child in psutil.Process(parent_pid).children():
            # You might need to refine this based on how Ollama spawns processes.
            # Checking the command line or process name might be necessary for more accuracy.
            # For a simple case, we assume the first child created after the delay is the Ollama process.
            ollama_child_process = child
            break

        if not ollama_child_process:
            print("Could not identify the Ollama child process.")
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            return response, None, None, None

        ollama_pid = ollama_child_process.pid
        stop_tracking = threading.Event()
        tracker_thread = threading.Thread(target=track_resources, args=(ollama_pid, 0.1, stop_tracking))
        tracker_thread.start()

        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

        stop_tracking.set()
        tracker_thread.join()

        timestamps, cpu_history, memory_history = track_resources(ollama_pid) # Get the final data

        return response, timestamps, cpu_history, memory_history

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model_name = "llama2"  # Replace with your desired model
    prompt = "Tell me a short story."

    response, timestamps, cpu_usage, memory_usage = execute_ollama_chat(model_name, prompt)

    if response:
        print("Ollama Response:")
        print(response['message']['content'])

    if timestamps and cpu_usage and memory_usage:
        # Plotting the results
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(timestamps, cpu_usage)
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"CPU Usage during Ollama Chat ({model_name})")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(timestamps, memory_usage)
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Usage (MB)")
        plt.title(f"Memory Usage during Ollama Chat ({model_name})")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
