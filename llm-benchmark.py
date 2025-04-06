import ollama
import time
import psutil
import threading
import matplotlib.pyplot as plt

# Constants
GB = 1024 ** 3  # 1 GB = 1024^3 bytes
EVALUATION_PROMPT = """
Please respond with a brief summary of the following text:
'Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.'
"""
MODELS_TO_EVALUATE = ["llama2", "mistral", "mixtral", "llava"]  # Updated list with more Ollama-friendly models
RESOURCE_MONITOR_INTERVAL = 2  # seconds

# Function to get and print detailed system information
def print_system_info():
    print("-" * 40)
    print("System Information:")
    cpu_count = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / GB
    print(f"  CPU Cores (Logical): {cpu_count}")
    print(f"  Total Memory: {total_memory_gb:.2f} GB")
    print("-" * 40)

# Function to pull a model if not available
def pull_model(model_name):
    try:
        print(f"Attempting to pull model: {model_name}...")
        ollama.pull(model_name)
        print(f"Model '{model_name}' pulled successfully.")
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")

# Function to collect resource usage (CPU, memory) at intervals
def collect_resource_usage(cpu_usage_list, mem_available_list, stop_event):
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=RESOURCE_MONITOR_INTERVAL)
        mem_info = psutil.virtual_memory()
        mem_available_gb = mem_info.available / GB  # Convert memory to GB

        cpu_usage_list.append(cpu_usage)
        mem_available_list.append(mem_available_gb)

# Function to evaluate a model's performance and track resources
def evaluate_model(model_name, prompt):
    print(f"\n--- Evaluating Model: {model_name} ---")
    try:
        # Check if the model is available, if not, try pulling it
        try:
            ollama.show(model_name)  # Check if model exists locally
        except Exception:
            print(f"Model '{model_name}' not found locally. Attempting to pull it...")
            pull_model(model_name)

        # Lists to track CPU and memory usage over time
        cpu_usage_list = []
        mem_available_list = []

        # Event to signal when resource collection should stop
        stop_event = threading.Event()

        # Start tracking the time for CPU and memory usage
        resource_thread = threading.Thread(
            target=collect_resource_usage,
            args=(cpu_usage_list, mem_available_list, stop_event),
            daemon=True  # Allow the main thread to exit even if this is running
        )
        resource_thread.start()

        # Generate the response while resources are being tracked
        response_start_time = time.time()
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        model_response = response['message']['content']
        response_time = time.time() - response_start_time

        # Signal the resource collection thread to stop
        stop_event.set()

        # Calculate the word count in the response
        word_count = len(model_response.split())
        print(f"Response: {model_response}")
        print(f"Word Count: {word_count} words")

        # Calculate average CPU and memory usage during generation
        if cpu_usage_list:
            avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
        else:
            avg_cpu_usage = 0

        if mem_available_list:
            # Calculate average memory *available*, so higher is better
            avg_mem_available = sum(mem_available_list) / len(mem_available_list)
        else:
            avg_mem_available = 0

        # Print resource usage stats
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"Average CPU Usage (during generation): {avg_cpu_usage:.2f}%")
        print(f"Average Memory Available (during generation): {avg_mem_available:.2f} GB")

        # Plot CPU and memory usage
        time_points = [i * RESOURCE_MONITOR_INTERVAL for i in range(len(cpu_usage_list))]
        plt.figure(figsize=(10, 6))

        # Plot CPU usage
        plt.subplot(2, 1, 1)
        plt.plot(time_points, cpu_usage_list, label="CPU Usage (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.title(f"{model_name} - CPU Usage")
        plt.grid(True)
        plt.legend()

        # Plot memory availability
        plt.subplot(2, 1, 2)
        plt.plot(time_points, mem_available_list, label="Memory Available (GB)", color='green')
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Available (GB)")
        plt.title(f"{model_name} - Memory Availability")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
    finally:
        print("-" * 40)

# Main function to run the evaluations
def main():
    print_system_info()
    for model in MODELS_TO_EVALUATE:
        evaluate_model(model, EVALUATION_PROMPT)

if __name__ == "__main__":
    main()
