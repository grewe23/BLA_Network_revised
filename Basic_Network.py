# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:11:50 2014

@author: grewe
"""
import numpy as np
import activation_functions as af


'''
class Network takes network initialization paramters then inputs and 
sends generated network outputs. It should comunicate only with the Layer class 
to initiate layers, send inputs and to get back the layer output activations.
'''










class neuron():
    def __init__(self, eta, activation_function, learning_rule):
        self.intralll=eta
    
    def neuron_activation(self,a):
        print('neuron class: ' + str(a))





class layer():
    def __init__(self, biases, intra_l_weights, inter_l_weights, bias_offsets, layer_down_regs):
        self.biases=biases
        self.IntraW=intra_l_weights
        self.InterW=inter_l_weights
        self.layer_down_reg=layer_down_regs
        self.bo=bias_offsets #  biases offset
    
   
   
    def layer_activation(self,layer_input):   
        z= self.biases + self.bo 
        zs=z
        reg_activity,highest_act =[], []
        num_highest_neurons=0
        
        print('biases ' + str(len(self.biases)))
        print('intra layer weights ' + str(np.shape(self.IntraW)) )         
        print('inter layer weights ' + str([np.shape(self.InterW[x]) for x in xrange(len(self.InterW))]))
        print('layer input ' + str([len(layer_input[y]) for y in xrange(len(layer_input))]))          
        
        for l in xrange(len(layer_input)):    # going through inputs and adding them to z first time                 
            z= z + np.dot(self.InterW[l],layer_input[l]) # z based on pure inputs
        layer_input.append(af.sigmoid_vec(z)) # layer activation based on pure inputs 
        ''''
        for k in xrange(len(layer_input)-1):      # intra_layer calculations                       
                zs= zs + np.dot(self.InterW[k],layer_input[k]) # z based on pure inputs
        zs= zs + np.dot(self.IntraW,layer_input[len(layer_input)-1]) #adding intra_layer activity as last step
        '''
        # intra layer regulationto 10%                   
        reg_activity=af.sigmoid_vec(z)  # final layer activity before regulation
        
        if self.layer_down_reg!=100:
            num_highest_neurons=int(len(reg_activity)/(100/self.layer_down_reg))
            highest_act=np.argpartition(-reg_activity.T, num_highest_neurons) #  indicies of the 10 highherst acitivities     
            for s in xrange(len(reg_activity)):
                if s in highest_act.T[:(num_highest_neurons)]:
                    reg_activity[s]=reg_activity[s]
                else:
                    reg_activity[s]=0
        else:
            pass
            
        layer_output=reg_activity
        
        new_INTRA_weights=self.IntraW
        new_INTER_weights=self.InterW
        new_biases=self.biases
        '''print('Biases for current Layer ' +str(len(self.biases)))
        print('INTRA layer weights:' + str(np.shape(self.IntraW)))        
        print('INTER layer weights:' + str([np.shape(self.InterW[x]) for x in xrange(len(self.InterW))]))
        print('Dimensions of layer input: ' + str(len(layer_input)))  '''   
        
        return layer_output, new_biases, new_INTRA_weights, new_INTER_weights
        


      



# defines the network class
class basic_network():

    def __init__(self, net_params, neuron_params):
    
        self.net_params=net_params
        self.neuron_params=neuron_params        
        self.layers=[x for (x,y) in net_params['layers']]
        self.layer_IO=[y for (x,y) in net_params['layers']]
        self.layer_connect=net_params['layer_connect']
        self.bias_offsets=net_params['bias_offset']
        self.layer_act_reg=net_params['layer_act_reg']
        self.instr_strength=net_params['inst_strenght']
        
        # create and initialize biases & wheigt matrices
        self.biases = []
        self.INTRA_layer_weights=[]
        self.INTER_layer_weights=[]
        
        for x in xrange(len(self.layers)):
            
            if  self.layer_IO[x]=='input':
                self.INTER_layer_weights.append([])
                self.INTRA_layer_weights.append([])
                self.biases.append([])            
            
            elif self.layer_IO[x]=='hidden':
                self.INTRA_layer_weights.append((np.random.randn(self.layers[x],self.layers[x])))
                self.biases.append((np.random.randn(self.layers[x],1)))
                self.INTER_layer_weights.append([(np.random.randn((self.layers[x]),(self.layers[k]))) for k in self.layer_connect[x]])
                
            elif self.layer_IO[x]=='instr': 
                self.INTER_layer_weights.append([])
                self.INTRA_layer_weights.append([])
                self.biases.append([])
                
            else: # this is initializing the output layer
                self.INTRA_layer_weights.append(np.random.randn(self.layers[x],self.layers[x]))
                self.biases.append((np.random.randn(self.layers[x],1)))
                self.INTER_layer_weights.append([(np.random.randn((self.layers[x]),(self.layers[k]))) for k in self.layer_connect[x]])
                
                
        #printing network properties
        print ('----------------------------------------------------')     
        print ('Layers:' + str(zip(self.layers,self.layer_IO)))                 #printing how many layers we initialize       
        print('Layer connection:'+ str(self.layer_connect))
        print('---')           #printing layer connections
        print('Biases:' + str([len(self.biases[x]) for x in xrange(len(self.biases))]))   #printing how many bises we have   
        print('INTRA layer weights:' + str([np.shape(self.INTRA_layer_weights[x]) for x in xrange(len(self.INTRA_layer_weights))]))        
        print('INTER layer weights:' + str([np.shape(self.INTER_layer_weights[x]) for x in xrange(len(self.INTER_layer_weights))]))
        
   
    # define net function that takes input and returns network output       
    def network_activation(self,input_vector):
        net_activation=[]
        
            # go through layers, and handle respectively to input/hidden/output
        for x in xrange(len(self.layers)):
           print ('----------------------------------------------------')
           print('Compute ' +self.layer_IO[x]  +' layer - Layer Nr: ' +str(x) +' - Layer has ' +str(self.layers[x]) +' neurons')
           layer_input=[]
           
           if self.layer_IO[x]=='instr':             
               net_activation.append(input_vector[0])
           elif self.layer_IO[x]=='input':
               net_activation.append(input_vector[1])
               
           elif self.layer_IO[x]=='hidden': 
               activate=layer(self.biases[x], self.INTRA_layer_weights[x],self.INTER_layer_weights[x], self.bias_offsets[x], self.layer_act_reg[x])   #init layer class
               for h in self.layer_connect[x]:
                   if h==0:
                       layer_input.append(net_activation[h]*self.instr_strength[x])
                   else:
                       layer_input.append(net_activation[h])
                       
               layer_activations=activate.layer_activation(layer_input)  # send input to layer and get layer activity
               net_activation.append(layer_activations[0]) # append layer activity to net activity
               
               #getting&updating  biases&weights from layer class
               self.biases[x]=layer_activations[1]
               self.INTRA_layer_weights[x]=layer_activations[2]
               self.INTER_layer_weights[x]=layer_activations[3]
               
                       
           else:  # this is for the output layer which is set to the activit of the instructive layer
               net_activation.append(input_vector[0])

           print('Length of layer output:' + str(len(net_activation[x])))
           print('length of net activation vector ' + str(len(net_activation)))
        return net_activation            
            
         





             
         
     
     
        
    
    
        
        
        
        
        
        
        
     
     
        
        
    