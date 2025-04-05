import torch
import psutil
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import numpy as np
import os

# Constants for better readability and easier modification
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MB = 1024 ** 2
NUM_SAMPLES = 1000
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
GPT2_MODEL_NAME = "gpt2"
LLM_MAX_LENGTH = 50
SAMPLE_PROMPT = "The future of AI is"

# Function to get system information
def get_system_info():
    print("--- System Information ---")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / MB:.2f} MB")
        print(f"GPU Allocated Memory: {torch.cuda.memory_allocated(DEVICE) / MB:.2f} MB")
        print(f"GPU Cached Memory: {torch.cuda.memory_cached(DEVICE) / MB:.2f} MB")
    else:
        print("No CUDA-enabled GPU available.")
    print("-" * 30)

# Simple Neural Network Training Benchmark
def neural_net_benchmark():
    print("\n--- Neural Network Benchmark ---")

    # Create a simple neural network for benchmarking
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
            self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create a random dataset for training
    x = torch.randn(NUM_SAMPLES, INPUT_SIZE)
    y = torch.randint(0, OUTPUT_SIZE, (NUM_SAMPLES,))

    # Create DataLoader for batching
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the neural network, loss function, and optimizer
    model = SimpleNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Start timing the training
    start_time = time.time()

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Neural Network Training Time: {training_time:.2f} seconds")
    print("-" * 30)

# LLM Inference Benchmark (using GPT-2)
def llm_inference_benchmark():
    print("\n--- LLM Inference Benchmark (GPT-2) ---")

    # Load the pre-trained GPT-2 model and tokenizer
    try:
        model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME).to(DEVICE)
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    except Exception as e:
        print(f"Error loading GPT-2 model/tokenizer: {e}")
        return

    # Encode a sample prompt for inference
    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt").to(DEVICE)

    # Start timing the inference
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=LLM_MAX_LENGTH)

    end_time = time.time()
    inference_time = end_time - start_time

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {SAMPLE_PROMPT}")
    print(f"Generated Text: {generated_text}")
    print(f"LLM Inference Time: {inference_time:.2f} seconds")
    print("-" * 30)

# Main function to run the benchmarks
def main():
    get_system_info()
    neural_net_benchmark()
    llm_inference_benchmark()

if __name__ == "__main__":
    main()
