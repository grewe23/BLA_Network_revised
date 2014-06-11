# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:11:50 2014

@author: grewe
"""
import numpy as np
from activation_functions import sigmoid_vec as af


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
    def __init__(self, biases, intra_l_weights, inter_l_weights, bias_offsets, layer_down_regs, learning_rate, layer_ID):
        self.biases=biases
        self.IntraW=intra_l_weights
        self.IntraW_test=intra_l_weights
        self.InterW=inter_l_weights
        self.InterW_test=inter_l_weights
        self.layer_down_reg=layer_down_regs
        self.bo=bias_offsets #  biases offset
        self.learning_rate=learning_rate
        self.layer_ID=layer_ID

   
    def layer_activation(self,layer_input):   
        z= np.add(self.biases, self.bo)
        layer_output,  reg_activity,  highest_act =[], [], []
        num_highest_neurons=0
        
        
        if self.layer_ID == 'hidden':
            for l in xrange(len(layer_input)):              # going through inputs and adding them to z - first run                 
                z += np.dot(self.InterW[l],layer_input[l])  # z based on pure inputs - first run
                
            layer_input.append(af(z))                                                   # layer activation based on pure inputs - first run             
            reg_activity= af(z + np.dot(self.IntraW,layer_input[len(layer_input)-1]))   #adding intra_layer activity as last step
        
            # LAYER DONW-REGULATION STARTS HERE
            num_highest_neurons=int(len(reg_activity)/(100/self.layer_down_reg))    # layer down regulation to xx % (most active neurons)
            highest_act=np.argpartition(-reg_activity.T, num_highest_neurons)       # indicies of the  neurons with the highest acitivities     
            for s in xrange(len(reg_activity)):
                if s in highest_act.T[:(num_highest_neurons)]:
                    reg_activity[s]=reg_activity[s]
                else:
                    reg_activity[s]=0           # setting low active neurons to 0
            
            layer_output=reg_activity
        
        else:          
            layer_output=af(layer_input[0]) # if layer ID is 'output' we clamp the layer activity to the label/ instructive layer
            layer_input.append(layer_output)
      
            
        print('layer input ' + str([len(layer_input[y]) for y in xrange(len(layer_input)-1)]))  
        print('biases-' + str(len(self.biases))+ '   intra layer weights-' + str(np.shape(self.IntraW))+ '   inter layer weights-' + str([np.shape(self.InterW[x]) for x in xrange(len(self.InterW))]))
        print('layer output-' + str(len(layer_output)))
        
        
        
        # LEARNING STARTS HERE
        
        new_biases=self.biases
        updated_InterW=self.InterW 
        updated_IntraW=self.IntraW
       

        print('INTER MAX/MIN  UPDATED   ' + str(np.amax(updated_InterW[1])) +'   /   ' +str(np.amin(updated_InterW[1])))
        print('INTRA MAX/MIN  BEFORE   ' + str(np.amax(updated_IntraW)) +'   /   ' +str(np.amin(updated_IntraW))) 
        
        # update inter layer weights
        # applied learning rule is : dW= sign(w) * eta * (x * (y**3) - abs(w))      - classical OJA ICA
         
        for  i in xrange(len(self.InterW)):  #going through the inter weight matrices for the different inputs 
            
            if i>0:   # this only learns the input weights form the previous hidden layer, not the weights from the instruction layer i=0
                
                delta_InterW = np.sign(np.squeeze(updated_InterW[i])) * (self.learning_rate * \
                    (( np.multiply.outer(np.squeeze(layer_output**3),  np.squeeze(layer_input[i]))) - np.absolute(np.squeeze(updated_InterW[i])) ))  # hebbian learning ala OJA-ICA 
            
                updated_InterW[i] += delta_InterW  
                
                print('INTER MAX/MIN  UPDATED   ' + str(np.amax(updated_InterW[i])) +'   /   ' +str(np.amin(updated_InterW[i])))
            else:
                pass 
        
        # update intra layer weights # applied learning rule is : dW= sing(W) * eta * (x*y**3 - abs(w))        
        #--- x is input neuron activity, y is weight adjusting neuron activity 
        
        delta_IntraW = np.sign(np.squeeze(updated_IntraW)) * self.learning_rate * \
                    (( np.multiply.outer( np.squeeze(layer_input[len(layer_input)-1]**3),  np.squeeze(layer_input[len(layer_input)-1])  )  ) - np.absolute(np.squeeze(updated_IntraW)) )  # hebbian learning ala OJA-ICA 
            
        updated_IntraW += delta_IntraW         
        
        print('INTRA MAX/MIN  UPDATED   ' + str(np.amax(updated_IntraW)) +'   /   ' +str(np.amin(updated_IntraW)))
        
        
        return layer_output, new_biases, updated_IntraW, updated_InterW
        


      



# defines the network class
class basic_network():

    def __init__(self, net_params, neuron_params):
    
        self.net_params=net_params
        self.neuron_params=neuron_params        
        self.layers=[x for (x,y) in net_params['layers']]
        self.layer_ID=[y for (x,y) in net_params['layers']]
        self.layer_connect=net_params['layer_connect']
        self.bias_offsets=net_params['bias_offset']
        self.layer_act_reg=net_params['layer_act_reg']
        self.instr_strength=net_params['inst_strenght']
        self.learning_rate=neuron_params['eta']
        
        
        # create and initialize biases & wheigt matrices
        self.biases = []
        self.INTRA_layer_weights=[]
        self.INTER_layer_weights=[]
               
        
        for x in xrange(len(self.layers)):
            
            if  self.layer_ID[x]=='input':
                self.INTER_layer_weights.append([])
                self.INTRA_layer_weights.append([])
                self.biases.append([])            
            
            elif self.layer_ID[x]=='hidden':
                self.INTRA_layer_weights.append((abs((np.random.randn(self.layers[x],self.layers[x]))))*(-0.1))
                self.biases.append((np.random.randn(self.layers[x],1)))
                self.INTER_layer_weights.append([(np.random.randn((self.layers[x]),(self.layers[k]))) for k in self.layer_connect[x]])
                
            elif self.layer_ID[x]=='instr': 
                self.INTER_layer_weights.append([])
                self.INTRA_layer_weights.append([])
                self.biases.append([])
                
            else: # this is initializing the output layer
                self.INTRA_layer_weights.append((abs((np.random.randn(self.layers[x],self.layers[x]))))*(-0.1))
                self.biases.append((np.random.randn(self.layers[x],1)))
                self.INTER_layer_weights.append([(np.random.randn((self.layers[x]),(self.layers[k]))) for k in self.layer_connect[x]])
           
        
        print('Initialization OK')     
        '''#printing network properties
        print ('----------------------------------------------------')     
        print ('Layers:' + str(zip(self.layers,self.layer_ID)))                 #printing how many layers we initialize       
        print('Layer connection:'+ str(self.layer_connect))
        print('---')           #printing layer connections
        print('Biases:' + str([len(self.biases[x]) for x in xrange(len(self.biases))]))   #printing how many bises we have   
        print('INTRA layer weights:' + str([np.shape(self.INTRA_layer_weights[x]) for x in xrange(len(self.INTRA_layer_weights))]))        
        print('INTER layer weights:' + str([np.shape(self.INTER_layer_weights[x]) for x in xrange(len(self.INTER_layer_weights))]))'''
        
   
   
   
   
    # define net function that takes input and returns network output       
    def network_activation(self,input_vector):
        net_activation=[]

            # go through layers, and handle respectively to input/hidden/output
        for x in xrange(len(self.layers)):
           print ('----------------------------------------------------')
           print('Compute ' +self.layer_ID[x]  +' layer Nr: ' +str(x) +' - ' +str(self.layers[x]) +' neurons')
           layer_input=[]
           
           if self.layer_ID[x]=='instr':             
               net_activation.append(input_vector[0])
           elif self.layer_ID[x]=='input':
               net_activation.append(input_vector[1])
               
           else:
               
               activate=layer(self.biases[x], self.INTRA_layer_weights[x],self.INTER_layer_weights[x], self.bias_offsets[x], \
                       self.layer_act_reg[x], self.learning_rate[x], self.layer_ID[x])   #init layer class
               
               for h in self.layer_connect[x]:   # this grabs the layer specif input from the layer connectivity 
                   if h==0:
                       layer_input.append(net_activation[h]*self.instr_strength[x])   # the intructive layer is the first input and mulitplied with the intr. strength
                   else:
                       layer_input.append(net_activation[h])
               
               
               layer_activations=activate.layer_activation(layer_input)  # send input to layer and get back layer activity, weights etc.
               net_activation.append(layer_activations[0])               # append the activity of each layer to the net activity
              
               #getting&updating  biases&weights from layer class
                   
               self.biases[x]=layer_activations[1]
               self.INTRA_layer_weights[x]=layer_activations[2]
               self.INTER_layer_weights[x]=layer_activations[3] 

               
        return net_activation            
            
         





             
         
     
     
        
    
    
        
        
        
        
        
        
        
     
     
        
        
    