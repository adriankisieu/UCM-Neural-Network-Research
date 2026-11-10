"""
This project tests whether a forward-pass deep neural network trained in PyTorch on the MNIST handwritten-digit dataset can be transferred onto real memristor hardware without 
losing significant accuracy. The main challenge is that original weights are continuous numbers, while physical memristors can only store specific conductance levels defined by 
the device’s measured Conductance Conversion Curve (CCC). After training the model with standard stochastic gradient descent, each weight is converted through this curve and 
quantized to a conductance the device can actually hold. The goal is to make the network physically implementable on memristor hardware while keeping the accuracy drop as small 
as possible.
"""
