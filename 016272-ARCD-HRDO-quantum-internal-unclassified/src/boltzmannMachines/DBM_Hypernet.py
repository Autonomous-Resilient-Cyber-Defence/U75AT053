import torch
from copy import deepcopy
from boltzmannMachines.DBM_action import DBM_action
import numpy as np
import os


class DBM_Hypernet(DBM_action):
    # DBM_Hypernet class a subclass of DBM_action
    # This class aims to use the DBM framework to act as a hypernetwork for a neural network.
    # For efficiency, the feature extractor NN layer is kept. This ensures that the number of inputs remains feasible.
    # The Structure of the DBM is [layer+nodeflags] - Hidden layers - [Node outputs]
    
   
    def __init__(self,baseline_nn,net_arch:"list[int]"=[16,16],
                 beta:float=5,numReads:int=200,numParallel:int=1,DWaveSystem:str='Advantage_system4.1'):
        
        super().__init__(1,0,net_arch,beta,numReads,numParallel,DWaveSystem)
        self.loadNN(baseline_nn)
        
    def forward(self,observation):
        self.getNN()
        inputsExtracted = self.nn[0].forward(observation)
        return self.nn[1].forward(inputsExtracted)
    
    def loadNN(self,NN,trainEpochs=0,storeFrequency=0,storeLocation=None,learningRate=0.0001):
        # Train DBM against NN values

        # Exctract some data on the NN
        # Assume it is in two sections        
        # assume NN[1] is a linear object for now
        self.nn_arch = [NN[0][2*i].in_features for i in range(int(len(NN[0])/2))] + [NN[1].in_features,NN[1].out_features]
        self.nn_n_layers = len(self.nn_arch)
        maxIn = max(self.nn_arch[1:-1])
        maxOut = max(self.nn_arch)

        # Define visible/actions spaces based on NN
        n_visible = self.nn_n_layers - 1 + maxIn + maxOut
        n_actions = maxOut
        super().__init__(n_visible,n_actions,self.net_arch,self.beta,self.numReads,self.nParallelAnneals,self.DWaveSystem)
        self.currWeights = [self.hvWeights, self.hhWeights, self.vBias, self.hBias, self.cBias - 1] # Don't store actual currWeights as we need to evaluarte the DBM once

        # Do some training
        self.nn = [deepcopy(NN[0]),deepcopy(NN[1])]
        allValues = torch.tensor([])
        maxScalar = 0
        for i in range(int(len(NN[0])/2)):
            NN[0][2*i].bias.requires_grad=False
            NN[0][2*i].weight.requires_grad=False
            self.nn[0][2*i].bias.requires_grad=False
            self.nn[0][2*i].weight.requires_grad=False                
            maxScalar = max(maxScalar,NN[0][2*i].bias.max())
            maxScalar = max(maxScalar,NN[0][2*i].weight.max())
            allValues = torch.cat((allValues,NN[0][2*i].bias))
            allValues = torch.cat((allValues,NN[0][2*i].weight.flatten()))
        NN[1].bias.requires_grad=False
        NN[1].weight.requires_grad=False
        self.nn[1].bias.requires_grad=False
        self.nn[1].weight.requires_grad=False
        allValues = torch.cat((allValues,NN[1].bias))
        allValues = torch.cat((allValues,NN[1].weight.flatten()))
        self.Scalars = [allValues.mean(),allValues.std()]

        self.nn_ = [deepcopy(self.nn[0]),deepcopy(self.nn[1])]
        optimizer = torch.optim.Adam(self.parameters(),learningRate)
        self.getNN(scale=False)
        meanWeight = torch.mean(self.nn[1].bias).detach()
        with torch.no_grad():
            self.cBias[0] = -meanWeight
        
        for iE in range(trainEpochs):
            n_observations = self.nn_n_layers - 1 + self.nn_arch[1] + sum(self.nn_arch[1:-1])
            for iO in range(n_observations):
                optimizer.zero_grad()
                self.getNN(scale = False,slice=iO)
                loss = self.calcLoss(NN,slice=iO)
                loss.backward()
                optimizer.step()
            if (storeLocation is not None) and storeFrequency>0:
                if ((iE+1) % storeFrequency == 0):
                    storeLocation.joinpath(f'Epoch_{iE+1:.0f}\\').mkdir()
                    self.saveWeights(storeLocation.joinpath(f'Epoch_{iE+1:.0f}\\'))
        self.getNN(scale = True)

    def getNN(self,scale=True,slice=None):
        # Check if DBM parameters have changed
        newWeights = [self.hvWeights, self.hhWeights, self.vBias, self.hBias, self.cBias]

        tmp = 1 * self.cBias
        if all([torch.all(newWeights[i]==self.currWeights[i]) for i in range(len(newWeights))]) and \
            not (tmp.requires_grad and not self.nn[1].weight.requires_grad):
            # If weights have not changed, NN has not changed. Continue with saved one
            # But, if we now require grad but did not when we initialised the nn, then remake the nn
            return
        
        # Define observations and submit to DBM
        n_observations = self.nn_n_layers - 1 + self.nn_arch[1] + sum(self.nn_arch[1:-1])
        observation = torch.zeros(n_observations,self.nObservations)
        
        self.nn = [deepcopy(self.nn_[0]),deepcopy(self.nn_[1])] # reset self.nn with a fresh copy
        count = 0
        # Node biases
        for i in range(self.nn_n_layers-1):
            observation[i,i] = 1
            count += 1
        # First layer weights
        for i in range(self.nn_arch[1]):
            observation[count,0] = 1 # layer flag
            observation[count,i+self.nn_n_layers-1] = 1 # node flag
            count+=1
        # Other weights
        for j in range(1,len(self.nn_arch[:-1])):
            for i in range(self.nn_arch[j]):
                observation[count,j] = 1 # layer flag
                observation[count,i+self.nn_n_layers-1] = 1 # node flag
                count+=1
        if slice is None:
            outVals = self.evaluateQBM(observation)  
        else:
            outVals = torch.zeros(observation.shape[0],self.nActions)
            outVals[slice,:] = self.evaluateQBM(observation[slice,:].reshape((1,-1)))

        if scale:
            outVals = (outVals * self.Scalars[1])  + self.Scalars[0]

        # Assign outVals to NN
        count = 0
        # Assign Bias
        for i in range(self.nn_n_layers-2):
            self.nn[0][2*i].bias[:] = outVals[i,:self.nn[0][2*i].out_features]
            count += 1
        self.nn[1].bias[:] = outVals[count,:self.nn[1].out_features]
        count += 1

        # Assign first layer - backwards to allow large inputs
        for i in range(self.nn_arch[1]):
            self.nn[0][0].weight[i,:] = outVals[count,:self.nn[0][0].in_features]
            count+=1
        # Assign subsequent layers
        for j in range(1,len(self.nn_arch[1:-1])):
            for i in range(self.nn_arch[j]):
                self.nn[0][2*j].weight[:,i] = outVals[count,:self.nn[0][2*j].out_features]
                count+=1
        # Assign final layer
        for i in range(self.nn_arch[-2]):
            self.nn[1].weight[:,i] = outVals[count,:self.nn[1].out_features]
            count+=1

        # Store copy of current DBM weights for comparison
        self.currWeights = [deepcopy(newWeights[i].detach()) for i in range(len(newWeights))]

    def calcLoss(self,NN,slice=None):
        loss = 0
        count = 0

        # Error in bias
        for i in range(self.nn_n_layers-2):
            thisLoss = self.nn[0][2*i].bias - (NN[0][2*i].bias.detach() - self.Scalars[0])/self.Scalars[1]
            if slice is None or count==slice:
                loss += sum(thisLoss**2)
            count += 1
        thisLoss = self.nn[1].bias - (NN[1].bias.detach() - self.Scalars[0])/self.Scalars[1]
        if slice is None or count==slice:
            loss += sum(thisLoss**2)
        count += 1

        # Error in weights
        for j in range(len(self.nn_arch[:-2])):
            for i in range(self.nn_arch[j]):
                thisLoss = self.nn[0][2*j].weight[:,i] - (NN[0][2*j].weight[:,i].detach() - self.Scalars[0])/self.Scalars[1]
                if slice is None or count==slice:
                    loss += sum(thisLoss**2)
                count+=1
        for i in range(self.nn_arch[-2]):
            thisLoss = self.nn[1].weight[:,i] - (NN[1].weight[:,i].detach() - self.Scalars[0])/self.Scalars[1]
            if slice is None or count==slice:
                loss += sum(thisLoss**2)
            count+=1
        return loss

    def saveWeights(self,Location):
        super().saveWeights(Location)
        np.savetxt(os.path.join(Location,'scalars.txt'),self.Scalars)

    def loadWeights(self,Location):
        super().loadWeights(Location)
        self.Scalars = np.loadtxt(os.path.join(Location,'scalars.txt'))
        
