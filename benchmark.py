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
HIDDEN_SIZE = 256  # Increased hidden layer size
OUTPUT_SIZE = 10
BATCH_SIZE = 128  # Increased batch size
NUM_EPOCHS = 10     # Increased number of epochs
LEARNING_RATE = 0.005 # Adjusted learning rate
WEIGHT_DECAY = 1e-4  # Added weight decay for regularization
LOG_INTERVAL = 100  # Print loss every LOG_INTERVAL batches

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

    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    mem = psutil.virtual_memory()
    print(f"  Total Memory: {mem.total / MB:.2f} MB")
    print(f"  Available Memory: {mem.available / MB:.2f} MB")

    if torch.cuda.is_available():
        print(f"\n--- GPU Information ---")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / MB:.2f} MB")
            print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / MB:.2f} MB")
            print(f"  Cached Memory: {torch.cuda.memory_cached(i) / MB:.2f} MB")
    else:
        print("\nNo CUDA-enabled GPU available.")
    print("-" * 30)

# Improved Neural Network Training Benchmark with more metrics
def neural_net_benchmark():
    print("\n--- Neural Network Benchmark ---")

    # Create a slightly more complex neural network
    class ImprovedNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(HIDDEN_SIZE // 2, OUTPUT_SIZE)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu2(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x

    # Create a random dataset for training
    x = torch.randn(NUM_SAMPLES, INPUT_SIZE)
    y = torch.randint(0, OUTPUT_SIZE, (NUM_SAMPLES,))

    # Create DataLoader for batching
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the neural network, loss function, and optimizer
    model = ImprovedNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Using Adam optimizer

    # Start timing the training
    start_time = time.time()
    total_loss = 0.0

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

            if (i + 1) % LOG_INTERVAL == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                      f"GPU Mem Allocated: {torch.cuda.memory_allocated(DEVICE) / MB:.2f} MB"
                      if torch.cuda.is_available() else "", end='\r')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = running_loss / len(train_loader)
        total_loss += avg_loss
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} completed, Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f} seconds")

    end_time = time.time()
    training_time = end_time - start_time
    avg_total_loss = total_loss / NUM_EPOCHS
    print(f"\nNeural Network Training Time: {training_time:.2f} seconds")
    print(f"Average Loss over {NUM_EPOCHS} Epochs: {avg_total_loss:.4f}")
    print("-" * 30)

# Main function to run the benchmarks
def main():
    get_system_info()
    neural_net_benchmark()

if __name__ == "__main__":
    main()
