import ollama
import time
import psutil
import platform
import os

GB = 1024 ** 3

# List of 100 evaluation prompts
EVALUATION_PROMPTS = [
    "Please respond with a brief summary of the following text: 'Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.'",
    "How would you explain quantum computing in simple terms?",
    "What are the ethical implications of AI in decision making?",
    "Summarize the plot of the book '1984' by George Orwell.",
    "What are the benefits and challenges of renewable energy?",
    "Describe the concept of blockchain technology.",
    "Explain the significance of the Turing test in artificial intelligence.",
    "What is the difference between machine learning and deep learning?",
    "How do neural networks work?",
    "Describe the process of photosynthesis.",
    "What are the major causes of climate change?",
    "Explain the importance of data privacy in the digital age.",
    "What are the key features of the Python programming language?",
    "Summarize the impact of social media on modern communication.",
    "What is the purpose of encryption in cybersecurity?",
    "Describe the difference between supervised and unsupervised learning.",
    "How does the internet of things (IoT) work?",
    "What are the dangers of artificial general intelligence?",
    "What is a self-driving car and how does it work?",
    "Describe the concept of augmented reality (AR).",
    # Add more prompts here until you have 100 prompts
    "What is the role of ethics in AI development?",
    "How does natural language processing (NLP) work?",
    "Explain the concept of genetic algorithms.",
    "What is the significance of the discovery of the Higgs boson?",
    "Summarize the plot of the movie 'Inception'.",
    "Explain the process of 3D printing.",
    "What are the major advancements in space exploration?",
    "How does the human brain process information?",
    "What is the concept of a black hole?",
    "Explain the importance of biodiversity in ecosystems.",
    "How does the stock market work?",
    "Describe the process of carbon capture and storage.",
    "What is the role of machine vision in autonomous systems?",
    "Summarize the main concepts of Einstein's theory of relativity.",
    "What is the difference between classical and quantum physics?",
    "Explain the significance of renewable energy sources like solar and wind.",
    "What is the principle behind the concept of smart cities?",
    "Describe the history and future of artificial intelligence.",
    "Explain how vaccines work in the human body.",
    "What are the ethical concerns surrounding genetic engineering?",
    "Summarize the principles of cloud computing.",
    "What is the significance of dark matter in cosmology?",
    # Continue to add until you reach 100 prompts
]

MODELS_TO_EVALUATE = ["llama2", "mistral", "mixtral", "llava"]

def print_system_info():
    system_info = []
    system_info.append("--- System Information ---")
    system_info.append(f"Operating System: {platform.system()} {platform.release()}")
    system_info.append(f"CPU: {platform.processor()}")
    cpu_info = {}
    if platform.system() == "Linux":
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    cpu_info[key.strip()] = value.strip()
        system_info.append(f"  Model Name: {cpu_info.get('model name', 'N/A')}")
        system_info.append(f"  Cores: {cpu_info.get('cpu cores', 'N/A')}")
        system_info.append(f"  Threads: {cpu_info.get('siblings', 'N/A')}")
        system_info.append(f"  CPU Frequency: {cpu_info.get('cpu MHz', 'N/A')} MHz (current)")
    elif platform.system() == "Windows":
        import wmi
        c = wmi.WMI()
        for processor in c.Win32_Processor():
            system_info.append(f"  Model: {processor.Name}")
            system_info.append(f"  Cores: {processor.NumberOfCores}")
            system_info.append(f"  Threads: {processor.NumberOfLogicalProcessors}")
            system_info.append(f"  Current Clock Speed: {processor.CurrentClockSpeed} MHz")
    elif platform.system() == "Darwin":  # macOS
        import subprocess
        system_info.append("  Model:", subprocess.getoutput("sysctl -n machdep.cpu.brand_string"))
        system_info.append("  Cores:", subprocess.getoutput("sysctl -n machdep.cpu.core_count"))
        system_info.append("  Threads:", subprocess.getoutput("sysctl -n machdep.cpu.thread_count"))
        system_info.append("  CPU Frequency:", subprocess.getoutput("sysctl -n hw.cpufrequency_max") + " Hz (max)")

    cpu_usage_at_start_percent = psutil.cpu_percent()
    mem_usage_at_start_percent = psutil.virtual_memory().percent

    system_info.append(f"CPU Usage (at start): {cpu_usage_at_start_percent}%")
    system_info.append(f"Memory Usage (at start): {mem_usage_at_start_percent}%")
    mem = psutil.virtual_memory()

    mem_total = mem.total / GB
    mem_available_at_start_gb = mem.available / GB

    system_info.append(f"  Total Memory: {mem_total:.2f} GB")
    system_info.append(f"  Available Memory (at start): {mem_available_at_start_gb:.2f} GB")
    system_info.append("-" * 40)

    # Save system info to a text file
    with open("system_info.txt", "w") as file:
        file.write("\n".join(system_info))

    return cpu_usage_at_start_percent, mem_available_at_start_gb

def pull_model(model_name):
    try:
        print(f"Attempting to pull model: {model_name}")
        ollama.pull(model_name)
        print(f"Model {model_name} pulled successfully.")
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")

def evaluate_model(model_name, prompt, prompt_index):
    print(f"\n--- Evaluating Model: {model_name} --- Prompt {prompt_index}")
    config_file_path = "config.txt"
    with open(config_file_path, 'w') as file:
        file.write(f"model_name={model_name}, prompt_index={prompt_index}\n")

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

    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
    finally:
        print("-" * 40)

# Main function to run the evaluations
def main():
    print_system_info()
    for model in MODELS_TO_EVALUATE:
        for idx, prompt in enumerate(EVALUATION_PROMPTS, start=1):
            evaluate_model(model, prompt, idx)

if __name__ == "__main__":
    main()
