# -*- coding: utf-8 -*-
# ==== pythonstartup.py ====
import os
# add something to clear the screen
class cls(object):
    def __repr__(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        return ''

cls = cls()

# ==== end pythonstartup.py ====


"""
Created on Wed May 14 18:03:06 2014

@author: grewe
"""

import mnist_loader
import Basic_Network as bn
import matplotlib.pyplot as plt
import numpy as np


# Load data (see data/README for instructions on downloading MNIST set)
training_data, validation_data, test_data =mnist_loader.load_data_wrapper()
#training_data=mnist_loader.load_training_data_with_label(0)



#training_data1=[training_data[x] for x in xrange(len(training_data)) if training_data[x][1]==training_data[0][1]]
#print (len(training_data1))
#print( len(training_data1[1][0]))
# defining network parameters
net_params = {        'layers':             [(10,'instr'), (784,'input'), (300,'hidden'), (300,'hidden'),(300,'hidden'), (300,'hidden'), (10,'output')],
                      'layer_connect':      [(0,), (0,), (0,1), (0,2), (0,3), (0,4), (0,5)], # forward connections to hidden/output layer 
                      'layer_act_reg':     [ 100, 100, 30, 20, 15, 10, 10 ],  # [ 100, 100, 30, 20, 10, 10, 10 ],
                      'bias_offset':        [-1 for x in xrange(7) ],
                      'inst_strenght':      [1, 1, 5, 10, 20, 40, 100]   # [1, 1, 5, 20, 30, 100, 100] 
                      
 }                    
neuron_params = {     'af_name':              'sigmoid',
                      'learning_rule_name':   'oja',
                      'eta':                   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ]   }



net=bn.basic_network(net_params,neuron_params)  # init network parameters


for l in xrange(15):    # going through the training data
    plt.clf() 
    abc=net.network_activation([training_data[l][1],(training_data[l][0])])
    print('final layer activation')
    print([len(abc[x]) for x in xrange(len(abc))])
    
    # plotting the network activity
    fig = plt.figure(1)
    plt.ion()
    plt.show()
    for x in xrange(len(abc)):
        #plt.subplot(1,len(abc),x+1) 
        hh=[(10,1), (28,28), (30,10), (30,10), (30,10), (30,10), (10,1)]
        digits=np.reshape(abc[x],hh[x])
        imgplot=fig.add_subplot(1,len(abc),x+1)
        imgplot=plt.imshow(digits, interpolation="nearest")
        imgplot.set_cmap('gray')            
    plt.draw()   



    fig2 = plt.figure(2)
    plt.ion()    
    plt.show()
    for x in xrange(len(abc)):
        histplot=fig2.add_subplot(1,len(abc),x+1)
        histplot=plt.hist(abc[x], bins=20, color='blue')         
    plt.draw()   

