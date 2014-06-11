import numpy as np

'''
Activation functions
'''

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z*0.1))
sigmoid_vec = np.vectorize(sigmoid)
def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))
sigmoid_prime_vec = np.vectorize(sigmoid_prime)


def tanh(z):
    return (np.tanh(z)+1.0)/2.0 # Scaled to [0,1]
tanh_vec = np.vectorize(tanh)
def tanh_prime(z):
    return (1.0 - np.tanh(z)**2) / 2.0
tanh_prime_vec = np.vectorize(tanh_prime)


def softplus(z):
    return np.log(1.0+np.exp(z))
softplus_vec = np.vectorize(softplus)
def softplus_prime(z):
    return  1.0/(1.0+np.exp(-z)) 
softplus_prime_vec = np.vectorize(softplus_prime)


def rectifier(z):
    return max(0,z)
rectifier_vec = np.vectorize(rectifier)
def rectifier_prime(z):
    return 1.0*(z>0)
rectifier_prime_vec = np.vectorize(rectifier_prime)

# Dict of functions and their derivatives
functions = {'sigmoid': (sigmoid_vec, sigmoid_prime_vec),
             'tanh': (tanh_vec, tanh_prime_vec),
            }


