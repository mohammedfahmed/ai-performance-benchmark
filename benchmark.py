import torch
import psutil
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import numpy as np
from torch.nn.utils import init_empty_weights

# Function to get system information
def get_system_info():
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
        print(f"GPU Usage: {torch.cuda.memory_cached() / (1024 ** 2)} MB")

# Simple Neural Network Training Benchmark
def neural_net_benchmark():
    print("\nStarting Neural Network Benchmark...")
    
    # Create a simple neural network for benchmarking
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(784, 128)  # Example: Input size 784 (28x28 image)
            self.fc2 = nn.Linear(128, 10)   # Output size 10 (for classification)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create a random dataset for training
    num_samples = 1000
    x = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))

    # Create DataLoader for batching
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the neural network, loss function, and optimizer
    model = SimpleNN().cuda() if torch.cuda.is_available() else SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Start timing the training
    start_time = time.time()

    model.train()
    for epoch in range(5):  # Train for 5 epochs
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda() if torch.cuda.is_available() else inputs, labels.cuda() if torch.cuda.is_available() else labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    end_time = time.time()
    print(f"Neural Network Training Time: {end_time - start_time:.2f} seconds")

# LLM Inference Benchmark (using GPT-2)
def llm_inference_benchmark():
    print("\nStarting LLM Inference Benchmark...")

    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda() if torch.cuda.is_available() else GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode a sample prompt for inference
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Start timing the inference
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=50)

    end_time = time.time()

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")
    print(f"LLM Inference Time: {end_time - start_time:.2f} seconds")

# Main function to run the benchmarks
def main():
    get_system_info()
    neural_net_benchmark()
    llm_inference_benchmark()

if __name__ == "__main__":
    main()
