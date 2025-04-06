import ollama
import time
import psutil
import platform
import matplotlib.pyplot as plt

# Constants
MB = 1024 ** 2
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

        # List to track CPU and memory usage over time
        cpu_usage_list = []
        mem_usage_list = []

        # Start tracking the time for CPU and memory usage
        start_time = time.time()
        cpu_usage_start = psutil.cpu_percent()
        mem_info_start = psutil.virtual_memory()
        mem_available_start = mem_info_start.available / MB

        # Start measuring CPU and memory usage at 2-second intervals
        while True:
            cpu_usage = psutil.cpu_percent(interval=2)
            mem_info = psutil.virtual_memory()
            mem_available = mem_info.available / MB

            cpu_usage_list.append(cpu_usage)
            mem_usage_list.append(mem_available)

            elapsed_time = time.time() - start_time
            if elapsed_time > 10:  # Stop after 10 seconds (just for example)
                break

        # Get final resource usage after the response is generated
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        model_response = response['message']['content']

        # Calculate the word count in the response
        word_count = len(model_response.split())
        print(f"Response: {model_response}")
        print(f"Word Count: {word_count} words")

        # Calculate CPU and memory usage deltas
        cpu_usage_end = psutil.cpu_percent()
        mem_info_end = psutil.virtual_memory()
        mem_available_end = mem_info_end.available / MB

        cpu_delta = cpu_usage_end - cpu_usage_start
        mem_delta = mem_available_start - mem_available_end  # Memory consumed

        # Calculate average CPU and memory usage during generation
        avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
        avg_mem_usage = sum(mem_usage_list) / len(mem_usage_list)

        # Print resource usage stats
        print(f"Response Time: {elapsed_time:.2f} seconds")
        print(f"Average CPU Usage (during generation): {avg_cpu_usage:.2f}%")
        print(f"Average Memory Consumed (during generation): {avg_mem_usage:.2f} MB")
        print(f"Total CPU Usage (during generation): {cpu_delta:.2f}%")
        print(f"Total Memory Consumed (during generation): {mem_delta:.2f} MB")

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
        plt.plot(mem_usage_list, label="Memory Available (MB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Available (MB)")
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
