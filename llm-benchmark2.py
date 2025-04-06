import ollama
import time
import psutil
import threading

def track_resources(pid, interval=0.1, stop_event=None):
    # ... (rest of the track_resources function remains the same) ...

def execute_ollama_chat(model_name, prompt):
    try:
        parent_process = psutil.Process()

        def find_ollama_child():
            time.sleep(0.5)  # Give time for the child process to start
            for child in parent_process.children():
                try:
                    cmdline = child.cmdline()
                    if any("ollama" in arg for arg in cmdline):
                        return child
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied:
                    pass # Might not have permissions to access cmdline
            return None

        tracker_thread = None
        stop_tracking = threading.Event()
        ollama_process = None

        # Start the Ollama chat
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

        # Now try to find the child process
        ollama_process = find_ollama_child()

        if ollama_process:
            ollama_pid = ollama_process.pid
            tracker_thread = threading.Thread(target=track_resources, args=(ollama_pid, 0.1, stop_tracking))
            tracker_thread.start()
        else:
            print("Could not identify the Ollama child process even after checking command line.")
            return response, None, None, None

        stop_tracking.set()
        if tracker_thread and tracker_thread.is_alive():
            tracker_thread.join()

        timestamps, cpu_history, memory_history = [], [], []
        if ollama_process and ollama_process.is_running():
            timestamps, cpu_history, memory_history = track_resources(ollama_process.pid)

        return response, timestamps, cpu_history, memory_history

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None

if __name__ == "__main__":
    # ... (rest of the if __name__ block remains the same) ...
