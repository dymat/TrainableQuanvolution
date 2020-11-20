
# coding: utf-8

# In[2]:


import torch
from torch import nn

import torchvision

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

from sklearn.metrics import accuracy_score


# In[3]:


class QonvLayer(nn.Module):
    def __init__(self, stride=2, device="default.qubit", wires=4, circuit_layers=4, n_rotations=8, out_channels=4, seed=None):
        super(QonvLayer, self).__init__()
        
        # init device
        self.wires = wires
        self.dev = qml.device(device, wires=self.wires)
        
        self.stride = stride
        self.out_channels = min(out_channels, wires)
        
        if seed is None:
            seed = np.random.randint(low=0, high=10e6)
            
        print("Initializing Circuit with random seed", seed)
        
        # random circuits
        @qml.qnode(device=self.dev)
        def circuit(inputs, weights):
            n_inputs=4
            # Encoding of 4 classical input values
            for j in range(n_inputs):
                qml.RY(inputs[j], wires=j)
            # Random quantum circuit
            RandomLayers(weights, wires=list(range(self.wires)), seed=seed)
            
            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)]
        
        weight_shapes = {"weights": [circuit_layers, n_rotations]}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
    
    
    def draw(self):
        # build circuit by sending dummy data through it
        _ = self.circuit(inputs=torch.from_numpy(np.zeros(4)))
        print(self.circuit.qnode.draw())
        self.circuit.zero_grad()
        
    
    def forward(self, img):
        bs, h, w, ch = img.size()
        if ch > 1:
            img = img.mean(axis=-1).reshape(bs, h, w, 1)
                        
        kernel_size = 2        
        h_out = (h-kernel_size) // self.stride + 1
        w_out = (w-kernel_size) // self.stride + 1
        
        
        out = torch.zeros((bs, h_out, w_out, self.out_channels))
        
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for b in range(bs):
            for j in range(0, h_out, self.stride):
                for k in range(0, w_out, self.stride):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = self.circuit(
                        inputs=torch.Tensor([
                            img[b, j, k, 0],
                            img[b, j, k + 1, 0],
                            img[b, j + 1, k, 0],
                            img[b, j + 1, k + 1, 0]
                        ])
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(self.out_channels):
                        out[b, j // kernel_size, k // kernel_size, c] = q_results[c]
                        
                 
        return out


# In[16]:


qonv = QonvLayer(circuit_layers=2, n_rotations=4, out_channels=4, stride=2)
qonv.draw()
x = torch.rand(size=(10,28,28,1))
qonv(x).shape


# In[17]:


def transform(x):
    x = np.array(x)
    x = x/255.0
    
    return torch.from_numpy(x).float()


# In[18]:


train_set = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4)


# # Experiment I (one Quanvolutional Layer)

# In[49]:


def training_experiment_1():
    print("Starting Experiment I")

    model = torch.nn.Sequential(
        QonvLayer(stride=2, circuit_layers=2, n_rotations=4, out_channels=4),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=14*14*4, out_features=10)
    )

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        for i, (x, y) in enumerate(train_loader):

            # prepare inputs and labels
            x = x.view(-1, 28, 28, 1)
            y = y.long()

            # reset optimizer
            optimizer.zero_grad()

            # engage
            y_pred = model(x)

            # error, gradients and optimization
            loss = criterion(y_pred, y)  
            loss.backward()
            optimizer.step()

            # output
            acc = accuracy_score(y, y_pred.argmax(-1).numpy())       

            print("Epoch:", epoch, "\tStep:", i, "\tAccuracy:", acc, "\tLoss:", loss.item())
            print("Gradients Layer 0:")
            print(model[0].circuit.weights.grad)

            if i % 5 == 0:
                model[0].draw()
            
            print("---------------------------------------")
            
            # early break
            if i > 0 and i % 10 == 0:
                break
            
    return model


# # Experiment II (two stacked Quanvolutional Layers)

# In[48]:


def training_experiment_2():
    print("Starting Experiment II")

    model = torch.nn.Sequential(
        QonvLayer(stride=2, circuit_layers=2, n_rotations=4, out_channels=4),
        QonvLayer(stride=2, circuit_layers=2, n_rotations=4, out_channels=4),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=7*7*4, out_features=10)
    )

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        for i, (x, y) in enumerate(train_loader):

            # prepare inputs and labels
            x = x.view(-1, 28, 28, 1)
            y = y.long()

            # reset optimizer
            optimizer.zero_grad()

            # engage
            y_pred = model(x)

            # error, gradients and optimization
            loss = criterion(y_pred, y)  
            loss.backward()
            optimizer.step()


            # output
            acc = accuracy_score(y, y_pred.argmax(-1).numpy())  

            print("Epoch:", epoch, "\tStep:", i, "\tAccuracy:", acc, "\tLoss:", loss.item())
            print("Gradients Layer 0:")
            print(model[0].circuit.weights.grad)
            print("Gradients Layer 1:")
            print(model[1].circuit.weights.grad)

            if i % 5 == 0:
                print("Current Circuit Layer 0:")
                model[0].draw()
                print("Current Circuit Layer 1:")
                model[1].draw()

            print("---------------------------------------")
            
            # early break
            if i > 0 and i % 10 == 0:
                break
            
    return model


# In[ ]:


if __name__ == "__main__":
    training_experiment_1()
    training_experiment_2()

