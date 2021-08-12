#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:12:04 2020

@author: victoria
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


#class for individual sub-denoising autoencoders
class DAE(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
       super().__init__()
       #Linear encoding layer
       self.encoder_hidden_layer = nn.Linear(
            in_features=input_size, out_features=output_size
        )
       #Linear decoding layer
       self.decoder_output_layer = nn.Linear(
            in_features= output_size, out_features=input_size
        )
        
       #loss criterion and optimizer during greedy training stages
       self.criterion = nn.MSELoss()
       #choice between two optimizers, SGD and Adam
       # self.optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
       self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        

    def forward(self, features, i, t):
        #perctenage guassian distributed noise to corrupt data data
        features =  features.detach()
        percentage = 0.05
        noise = np.random.normal(0, features.std(), features.shape) * percentage
        noised_features = (features+noise).float()
    
    
    
        #inference and take steps
        encoded = self.encode(noised_features.float())
        reconstructed = self.reconstruct(encoded)
        self.optimizer.zero_grad()
        loss =self.criterion(reconstructed.float(), Variable(features.data.float(), requires_grad=True))
        
        
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        if(i%1000==0):
            print("epoch : {}/{}, loss = {:.11f}".format(i, t, loss))
        
        return encoded
    
    def encode(self, features):
        #pass through encoding layer & rely activation
        activation = self.encoder_hidden_layer(features)
        code = torch.relu(activation)
        
        return code
    
    def reconstruct(self, features):
        #pass through decoding lauer
        activation = self.decoder_output_layer(features)
        return activation
    
    
#full stacked autoencoder class
class StackedAutoEncoder(nn.Module):
   

    def __init__(self, **kwargs):
        super(StackedAutoEncoder, self).__init__()
        #comprises of three DAEs, all encoding into lower dimensions
        self.ae1 = DAE(kwargs["input_shape"], kwargs["encode_shape"], 1e-2)
        self.ae2 = DAE(kwargs["encode_shape"],kwargs["encode_shape"] -2, 1e-2)
        self.ae3 = DAE(kwargs["encode_shape"] -2, kwargs["encode_shape"] -4, 1e-2)

    def forward(self, x):
        
        #train first autoencoder for 20000 epochs
        print("Training A1")
        for i in range(0,20000):
    
            a1 = self.ae1.forward(x,i, 20000)
            
        print(a1)
        
        
        #train second autoencoder for 20000 epochs
        print("Training A2")
        
        for i in range(0,20000):
            a2 = self.ae2.forward(a1,i, 20000)
            
        print(a2)
        
        #train third autoencoder for 20000 epochs
        print("Training A3")
        for i in range(0,20000):
            a3 = self.ae3.forward(a2,i,20000)
        print(a3)
        return self.reconstruct(a3)
        

    def reconstruct(self, x):
            #fully reconstruct encoded data, x
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct

      
    def encode(self, features):
        #encode input data into lower dimensionality
        a1 = self.ae1.encode(features)
        a2 = self.ae2.encode(a1)
        a3 = self.ae3.encode(a2)
       
        return a3