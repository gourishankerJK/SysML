import torch
import json

import torch.nn as nn
import torch.optim as optim
# Read config.json file
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract necessary values from config
layers = config["layers"]
embedding_size = config["embedding_size"]



def parse_layers_from_arch(arch):
    arch = {
    'LeNet5': [('C', 6, 5, 'not_same', 3),
                ('M',2,2),
                ('C', 16, 5, 'not_same'),
                ('M',2,2),
                ('fc' , 400 , 120 , ),
                 ('fc' , 120 , 84),
                ('fc' , 84 , 10)] ,
    }
    
    layers = arch['LeNet5']
    layer2vec = [0] * embedding_size
    index = 0
    for layer in layers : 
        if layer[0] == 'C' : 
             layer2vec[index] = layer[1]
             layer2vec[index+1] = layer[2]
             layer2vec[index+2] = config[layer[3]] * (layer[2]) // 2
             layer2vec[index+3] = config["end"]
             index += 4
        elif layer[0] == 'M' : 
             layer2vec[index] = layer[1]
             layer2vec[index+1] = layer[2]
             layer2vec[index+2] = config["end"]
             index += 3
        elif layer[0] == 'fc' : 
            layer2vec[index] = layer[1]
            layer2vec[index+1] = layer[2]
            layer2vec[index+2] = config["end"]
            index += 3
        else : 
            raise ValueError("Invalid layer type")
    print(layer2vec)
    


parse_layers_from_arch(None)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define your layers here
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define hyperparameters
input_size = 784  # Input size of the network
hidden_size = 128  # Number of neurons in the hidden layer
output_size = 10  # Number of classes in the output layer

# Create an instance of the neural network
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')