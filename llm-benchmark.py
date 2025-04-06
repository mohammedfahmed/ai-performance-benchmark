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

# Function to get more detailed system information
def get_system_info():
    print("-" * 40)

# Function to pull models if not available
def pull_model(model_name):
    print(f"Error pulling model {model_name}: {e}")

# Function to collect resource usage (CPU, memory) every 2 seconds
def collect_resource_usage(cpu_usage_list, mem_usage_list, stop_event):
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=2)
        mem_info = psutil.virtual_memory()
        mem_available = mem_info.available / GB  # Convert memory to GB

        cpu_usage_list.append(cpu_usage)
        mem_usage_list.append(mem_available)

# Function to evaluate a model's performance and track resources
def evaluate_model(model_name, prompt):
    print(f"\n--- Evaluating Model: {model_name} ---")
    try:
        # Check if the model is available, if not, try pulling it
        try:
            ollama.show(model_name)  # Check if model exists locally
        except Exception as e:
            print(f"Model {model_name} not found locally. Attempting to pull it...")
            pull_model(model_name)

        # Lists to track CPU and memory usage over time
        cpu_usage_list = []
        mem_usage_list = []

        # Event to signal when resource collection should stop
        stop_event = threading.Event()

        # Start tracking the time for CPU and memory usage
        resource_thread = threading.Thread(target=collect_resource_usage, args=(cpu_usage_list, mem_usage_list, stop_event))
        resource_thread.start()

        # Generate the response while resources are being tracked
        response_start_time = time.time()
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        model_response = response['message']['content']
        response_time = time.time() - response_start_time

        # Signal the resource collection thread to stop
        stop_event.set()

        # Wait for the resource collection thread to finish
        resource_thread.join()

        # Calculate the word count in the response
        word_count = len(model_response.split())
        print(f"Response: {model_response}")
        print(f"Word Count: {word_count} words")

        # Calculate average CPU and memory usage during generation
        avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
        avg_mem_usage = sum(mem_usage_list) / len(mem_usage_list)

        # Print resource usage stats
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"Average CPU Usage (during generation): {avg_cpu_usage:.2f}%")
        print(f"Average Memory Consumed (during generation): {avg_mem_usage:.2f} GB")

        # Plot CPU and memory usage
        plt.figure(figsize=(10, 5))

        # Plot CPU usage
        plt.subplot(2, 1, 1)
        plt.plot(cpu_usage_list, label="CPU Usage (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.legend()

        # Plot memory usage
        plt.subplot(2, 1, 2)
        plt.plot(mem_usage_list, label="Memory Available (GB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Available (GB)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
    finally:
        print("-" * 40)

# Main function to run the evaluations
def main():
    get_system_info()
    for model in MODELS_TO_EVALUATE:
        evaluate_model(model, EVALUATION_PROMPT)

if __name__ == "__main__":
    main()
