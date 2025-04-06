import ollama
import time
import psutil
import platform
import os

GB = 1024 ** 3

EVALUATION_PROMPT = """
Please respond with a brief summary of the following text:
'Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.'
"""

MODELS_TO_EVALUATE = ["llama2", "mistral", "mixtral", "llava"]

def print_system_info():
    print("--- System Information ---")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    cpu_info = {}
    if platform.system() == "Linux":
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    cpu_info[key.strip()] = value.strip()
        print(f"  Model Name: {cpu_info.get('model name', 'N/A')}")
        print(f"  Cores: {cpu_info.get('cpu cores', 'N/A')}")
        print(f"  Threads: {cpu_info.get('siblings', 'N/A')}")
        print(f"  CPU Frequency: {cpu_info.get('cpu MHz', 'N/A')} MHz (current)")
    elif platform.system() == "Windows":
        import wmi
        c = wmi.WMI()
        for processor in c.Win32_Processor():
            print(f"  Model: {processor.Name}")
            print(f"  Cores: {processor.NumberOfCores}")
            print(f"  Threads: {processor.NumberOfLogicalProcessors}")
            print(f"  Current Clock Speed: {processor.CurrentClockSpeed} MHz")
    elif platform.system() == "Darwin":  # macOS
        import subprocess
        print("  Model:", subprocess.getoutput("sysctl -n machdep.cpu.brand_string"))
        print("  Cores:", subprocess.getoutput("sysctl -n machdep.cpu.core_count"))
        print("  Threads:", subprocess.getoutput("sysctl -n machdep.cpu.thread_count"))
        print("  CPU Frequency:", subprocess.getoutput("sysctl -n hw.cpufrequency_max") + " Hz (max)")

    cpu_usage_at_start_percent = psutil.cpu_percent()
    mem_usage_at_start_percent = psutil.virtual_memory().percent

    print(f"CPU Usage (at start): {cpu_usage_at_start_percent}%")
    print(f"Memory Usage (at start): {mem_usage_at_start_percent}%")
    mem = psutil.virtual_memory()

    mem_total = mem.total / GB
    mem_available_at_start_gb = mem.available / GB

    print(f"  Total Memory: {mem_total:.2f} GB")
    print(f"  Available Memory (at start): {mem_available_at_start_gb:.2f} GB")
    print("-" * 40)
    return cpu_usage_at_start_percent, mem_available_at_start_gb

def pull_model(model_name):
    try:
        print(f"Attempting to pull model: {model_name}")
        ollama.pull(model_name)
        print(f"Model {model_name} pulled successfully.")
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")

def evaluate_model(model_name, prompt):
    print(f"\n--- Evaluating Model: {model_name} ---")
    os.environ['llm_model'] = model_name
    print(os.getenv('llm_model', 'Not Set'))

    try:
        try:
            ollama.show(model_name)
        except Exception as e:
            print(f"Model {model_name} not found locally. Attempting to pull it...")
            pull_model(model_name)

        cpu_usage_start = psutil.cpu_percent()
        mem_available_start = psutil.virtual_memory().available / GB

        start_time = time.time()
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        response_time = time.time() - start_time

        cpu_usage_end = psutil.cpu_percent()
        mem_info_end = psutil.virtual_memory()
        mem_available_end = mem_info_end.available / GB

        cpu_delta = cpu_usage_end - cpu_usage_start
        mem_delta = mem_available_end - mem_available_start

        # Extract the model's answer from the response
        model_response = response['message']['content']
        print(f"Response length: {len(model_response)}")
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"CPU Usage (during generation): {cpu_delta:.2f}%")
        print(f"Memory Consumed (during generation): {mem_delta:.2f} GB")
        os.environ['llm_model'] = ''

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
