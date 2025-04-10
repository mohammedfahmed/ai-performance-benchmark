import torch
import psutil
import time
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import numpy as np
import os
import platform  # For more detailed system info

# Constants for better readability and easier modification
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MB = 1024 ** 2
NUM_SAMPLES = 10000  # Increased number of samples for a more stable benchmark
INPUT_SIZE = 784
OUTPUT_SIZE = 10
BATCH_SIZE = 128  # Increased batch size
NUM_EPOCHS = 5     # Reduced epochs for multiple runs
LEARNING_RATE = 0.005 # Adjusted learning rate
WEIGHT_DECAY = 1e-4  # Added weight decay for regularization
LOG_INTERVAL = 100  # Print loss every LOG_INTERVAL batches
NETWORK_SIZES = [
    {"hidden_size": 256, "num_layers": 6, "dropout": 0.2, "name": "SmallNet"},   # Smaller model (for comparison)
    {"hidden_size": 1024, "num_layers": 12, "dropout": 0.3, "name": "MediumNet"},  # Medium model
    {"hidden_size": 8192, "num_layers": 24, "dropout": 0.4, "name": "LargeNet"}, # Larger model, closer to LLMs
]

# Function to get more detailed system information
def get_system_info():
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

    print(f"CPU Usage (at start): {psutil.cpu_percent()}%")
    print(f"Memory Usage (at start): {psutil.virtual_memory().percent}%")
    mem = psutil.virtual_memory()
    print(f"  Total Memory: {mem.total / MB:.2f} MB")
    print(f"  Available Memory (at start): {mem.available / MB:.2f} MB")

    if torch.cuda.is_available():
        print(f"\n--- GPU Information ---")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / MB:.2f} MB")
            print(f"  Allocated Memory (at start): {torch.cuda.memory_allocated(i) / MB:.2f} MB")
            print(f"  Cached Memory (at start): {torch.cuda.memory_cached(i) / MB:.2f} MB")
    else:
        print("\nNo CUDA-enabled GPU available.")
    print("-" * 40)

# Neural Network Training and Inference Benchmark with configurable size
def neural_net_benchmark(network_config):
    print(f"\n--- Neural Network Benchmark: {network_config['name']} ---")
    hidden_size = network_config['hidden_size']
    num_layers = network_config['num_layers']
    dropout_prob = network_config['dropout']
    model_name = network_config['name']

    # Create a configurable neural network
    class FlexibleNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_prob))
            self.output_layer = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = self.output_layer(x)
            return x

    # Create a random dataset for training
    x = torch.randn(NUM_SAMPLES, INPUT_SIZE)
    y = torch.randint(0, OUTPUT_SIZE, (NUM_SAMPLES,))

    # Create DataLoader for batching
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the neural network, loss function, and optimizer
    model = FlexibleNN(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layers, dropout_prob).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Using Adam optimizer

    # Report network size (number of parameters)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Network Architecture: {num_layers} layers, Hidden Size: {hidden_size}, Dropout: {dropout_prob}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 40)

    # Start timing the training
    start_time = time.time()
    total_loss = 0.0
    peak_memory_allocated = 0
    peak_memory_cached = 0

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        epoch_start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if torch.cuda.is_available():
                peak_memory_allocated = max(peak_memory_allocated, torch.cuda.memory_allocated(DEVICE))
                peak_memory_cached = max(peak_memory_cached, torch.cuda.memory_cached(DEVICE))

            if (i + 1) % LOG_INTERVAL == 0:
                gpu_mem_str = f", GPU Mem Allocated: {torch.cuda.memory_allocated(DEVICE) / MB:.2f} MB" if torch.cuda.is_available() else ""
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}{gpu_mem_str}", end='\r')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = running_loss / len(train_loader)
        total_loss += avg_loss
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} completed, Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f} seconds")

    end_time = time.time()
    training_time = end_time - start_time
    avg_total_loss = total_loss / NUM_EPOCHS

    print(f"\n--- Benchmark Results: {model_name} ---")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Average Loss over {NUM_EPOCHS} Epochs: {avg_total_loss:.4f}")
    if torch.cuda.is_available():
        print(f"Peak GPU Memory Allocated: {peak_memory_allocated / MB:.2f} MB")
        print(f"Peak GPU Memory Cached: {peak_memory_cached / MB:.2f} MB")

    # Inference Benchmark
    model.eval()
    inference_time = 0.0
    with torch.no_grad():
        x_inference = torch.randn(1000, INPUT_SIZE).to(DEVICE)
        start_inference_time = time.time()
        for _ in range(100):  # 100 iterations of inference
            outputs = model(x_inference)
        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
    print(f"Inference Time for 100 iterations: {inference_time:.2f} seconds")

    # Report final CPU and Memory usage
    print(f"CPU Usage (at end): {psutil.cpu_percent()}%")
    mem_end = psutil.virtual_memory()
    print(f"Memory Usage (at end): {mem_end.percent}%")
    print(f"  Available Memory (at end): {mem_end.available / MB:.2f} MB")
    print("-" * 40)

# Main function to run the benchmarks
def main():
    get_system_info()
    for network_config in NETWORK_SIZES:
        neural_net_benchmark(network_config)

if __name__ == "__main__":
    main()
