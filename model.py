
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_chanels = 1):
        super().__init__()

        #encoder

        # input channel 1, output channel 10, kernel_size 5, stride=1
        self.conv1 = nn.Conv2d(in_channels=num_chanels, out_channels=10,
                               kernel_size=5, stride=1,
                              )
        # input channel 10, output channel 20, kernel_size 5, stride=1
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20,
                               kernel_size=5, stride=1,
                              )
        # dropout layer
        self.conv2_drop = nn.Dropout(p=0.5)
        # fully connected layer from 320 --> 50
        self.fc1 = nn.Linear(20*244*320, 50)
        # fully connected layer from 50 --> 10
        self.fc2 = nn.Linear(50, 20)
        # self.hid_2mu = nn.Linear(h_dim, z_dim)
        # self.hid_2sigma = nn.Linear(h_dim, z_dim)



        #decoder
        self.fc2_rev = nn.Linear(20, 50)
        self.fc1_rev = nn.Linear(50, 20*244*320)
        self.conv2_rev = nn.ConvTranspose2d(in_channels=20, out_channels=10,kernel_size=5, stride=1)
        self.conv1_rev = nn.ConvTranspose2d(in_channels=10, out_channels=1,kernel_size=5, stride=1)



        # some transformations

        self.relu = nn.ReLU()

    def encode(self, x):
         # convolution 1
        x = self.conv1(x)
        #here
        # x = F.max_pool2d(x, 2)
        x = F.relu(x)

        # convolution 2
        x = self.conv2(x)

        x = self.conv2_drop(x)
        # x = F.max_pool2d(x, 2)
        x = F.relu(x)
        print("model", x.shape)

        # linear fully connected layers
        x = x.view(-1, 20*244*320)

        x = self.fc1(x)

        mu, sigma = self.fc2(x), self.fc2(x)

        return mu, sigma

    def decode(self, z):
        z = self.fc2_rev(z)
        z  = self.fc1_rev(z)

        z = z.view(-1, 20, 244, 320)
        # z = F.relu(z)
        z = self.conv2_rev(z)
        # z = F.relu(z)
        z = self.conv1_rev(z)



        return z

    def forward(self, z):
        mu, sigma = self.encode(z)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma