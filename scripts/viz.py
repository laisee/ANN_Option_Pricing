import torch
import torch.nn as nn
import torchviz

# Define the network (same as above)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Initialize the network
net = Net()

# Create a dummy input tensor
input_tensor = torch.randn(1, 1, 28, 28)

# Visualize the network
dot_graph = torchviz.make_dot(net(input_tensor), params=dict(net.named_parameters()))
dot_graph.render("network", format="png")
