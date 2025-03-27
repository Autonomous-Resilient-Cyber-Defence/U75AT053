from dimod import BinaryQuadraticModel
import os
import json
from dwave.system import FixedEmbeddingComposite, LazyFixedEmbeddingComposite, DWaveSampler
from dimod import SimulatedAnnealingSampler
import torch
import torch.nn as nn
import numpy as np
import threading
from time import time

class DBM(nn.Module):
    # DBM class a subclass of nn.Module
    # This allows use of built-in functionality to use a Boltzmann Machine instead
    # of a Neural Network.
    # This class is built to allow simple building and sampling of a DBM or RBM for 
    # use as a function approximator. Pytorch is used within calculations so that 
    # 'backward' steps update the weights accordingly.

    # TODO:
    # - Fully connected Boltzmann machines

    def __init__(self,n_visible:int,n_actions:int=0,net_arch:"list[int]"=[16,16],
                 beta:float=5,numReads:int=200,numParallel:int=1,DWaveSystem:str='Advantage_system4.1'):
        super().__init__()
        # n_visible = Number of visible units (i.e. observations plus actions (if applicable))
        # n_actions = Number of possible actions. If zero, the output of the system is a value based on the observation only. 
        #               If non-zero, the output of the system is a vector of probabilities for each action
        # net_arch  = list, detailing the number of hidden units in each layer
        # beta      = Assumed thermodynamic beta for use in Free Energy calculations
        # numReads  = Number of samples to take from the DWave sampler
        # numParallel = Number of parallel samples to build into the Hamiltonian
        # DWaveSystem = Which DWave QPU to use

        # Set up parameters
        self.beta = beta 
        self.numReads = numReads

        # Get space information
        self.nVisible = n_visible
        self.nObservations = n_visible - n_actions
        self.nActions = n_actions

        # Sampling efficiency options
        self.nParallelAnneals = numParallel

        # Initialise samplers
        self.SA_sampler = SimulatedAnnealingSampler()
        try:
            self.QA_sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver=DWaveSystem))
            self.SimulateAnneal = False
        except:
            self.QA_sampler = []
            self.SimulateAnneal = True

        self.DWaveSystem = DWaveSystem
        self.embeddingLoaded = False
        self.QPUtime_micro_s = 0
        self.wallClock_s = time()

        # Initialise Boltzmann Machines
        self.net_arch = net_arch
        if len(net_arch)==1:
            self.initRBM(net_arch)
        else:
            self.initDBM(net_arch)
        return

    def __call__(self,observation):
        # Set __call__ such that DBM(obs) = DBM.forward(obs)
        return self.forward(observation) 

    def forward(self,observation):
        return self.evaluateQBM(observation)    

    def initRBM(self,net_arch):
        # Initialise an RBM with random weights, zero weight between hidden nodes
        self.nHidden = net_arch[0]
        self.hiddenLayerSizes = [net_arch[0]]
        self.hvWeights = torch.randn((self.nVisible,self.nHidden))
        self.hhWeights = torch.zeros((self.nHidden,self.nHidden))
        self.hBias = torch.randn((self.nHidden))
        self.vBias = torch.randn((self.nVisible))

        self.nonzeroHV = torch.ones_like(self.hvWeights)
        self.nonzeroHH = torch.zeros_like(self.hhWeights)

        # Scale weights so initial free energy is between -1 and 1
        scale = torch.sum(torch.abs(self.hvWeights)) + \
            torch.sum(torch.abs(self.hBias)) + torch.sum(torch.abs(self.vBias))
        self.hvWeights = nn.Parameter(self.hvWeights / scale)
        self.hBias = nn.Parameter(self.hBias / scale)
        self.vBias = nn.Parameter(self.vBias / scale)
        self.cBias = nn.Parameter(torch.zeros((1)))

        self.QBMinitialised = True

    def initDBM(self,net_arch):
        # Initialise a DBM with random weights
        self.nHidden = sum(net_arch)
        self.hvWeights = torch.randn((self.nVisible,self.nHidden))
        self.hhWeights = torch.randn((self.nHidden,self.nHidden))
        self.hBias = torch.randn((self.nHidden))
        self.vBias = torch.randn((self.nVisible))

        self.nonzeroHV = torch.zeros_like(self.hvWeights)
        self.nonzeroHV[0:self.nObservations,0:net_arch[0]] = 1
        self.nonzeroHV[self.nObservations:self.nVisible,-net_arch[-1]:] = 1
        self.hvWeights = torch.multiply(self.hvWeights,self.nonzeroHV)

        # Strictly upper triangular matrix to avoid duplication
        self.nonzeroHH = torch.zeros_like(self.hhWeights)
        cumsum0 = 0
        for iH, nLayer in enumerate(net_arch[:-1]):
            cumsum1 = cumsum0 + nLayer
            self.nonzeroHH[cumsum0:cumsum1,cumsum1:(cumsum1+net_arch[iH+1])] = 1
            cumsum0 = cumsum1
        self.hhWeights = torch.multiply(self.hhWeights,self.nonzeroHH)

        # Scale weights so initial free energy is between -1 and 1
        scale = torch.sum(torch.abs(self.hhWeights)) + torch.sum(torch.abs(self.hvWeights)) + \
            torch.sum(torch.abs(self.hBias)) + torch.sum(torch.abs(self.vBias))
        self.hvWeights = nn.Parameter(self.hvWeights / scale)
        self.hhWeights = nn.Parameter(self.hhWeights / scale)
        self.hBias = nn.Parameter(self.hBias / scale)
        self.vBias = nn.Parameter(self.vBias / scale)
        self.cBias = nn.Parameter(torch.zeros((1)))

        self.QBMinitialised = True

    def buildHamiltonian(self,state,action=[],nParallelAnneals=None):
        # Build a Hamiltonian on hidden nodes only and add bias for visible node connections
        if self.nActions>0:
            observation = torch.concatenate((state,action))
        else:
            observation = state
        
        # Fix weights that have become non-zero
        # Two lots of filtering as non-zero values can sometimes slip through with large learning rates
        hhWeights = self.hhWeights * self.nonzeroHH
        hvWeights = self.hvWeights * self.nonzeroHV
        hhWeights[self.nonzeroHH==0] = 0
        hvWeights[self.nonzeroHV==0] = 0
        
        # Get quadratic terms (connections) and linear terms (biases)
        quadTerms = -hhWeights
        linearTerms = -(self.hBias + torch.matmul(torch.transpose(hvWeights,0,1),observation))
        constantTerm = -(torch.dot(self.vBias,observation) + self.cBias)
        Hamiltonian_tensor = [constantTerm, linearTerms, quadTerms]

        # Duplicate for parallel anneals

        if nParallelAnneals is None:
            nParallelAnneals = self.nParallelAnneals
        Hamiltonian = self.tileHamiltonian(linearTerms,quadTerms,nParallelAnneals)
        return Hamiltonian, Hamiltonian_tensor

    def tileHamiltonian(self,linearTerms,quadTerms,nParallelAnneals):
        nFreeNodes = linearTerms.shape[0]
        # Create block off-diagonal quadratic array
        quadTermsBlock = torch.zeros((nFreeNodes * nParallelAnneals,nFreeNodes * nParallelAnneals)).detach().numpy()
        i0 = 0
        for iP in range(nParallelAnneals):
            i1 = i0 + nFreeNodes
            quadTermsBlock[i0:i1,i0:i1] = quadTerms.detach().numpy()
            i0 += nFreeNodes

        # Build Hamiltonian
        Hamiltonian = BinaryQuadraticModel(nFreeNodes, 'BINARY')
        Hamiltonian.add_linear_from_array(torch.tile(linearTerms,(nParallelAnneals,)).detach().numpy())
        Hamiltonian.add_quadratic_from_dense(quadTermsBlock)
        return Hamiltonian


    def calculateFreeEnergy(self,results,beta,Hamiltonian):
        # Take results from sampled hamiltonian and calculate mean free energy
        resAgg = self.collateSamples(results,Hamiltonian)

        # Calculate Zv
        energies = torch.concatenate([record[1].reshape(-1,1) for record in resAgg],axis=1)
        minEnergy = torch.min(energies,axis=1) # Stop overflowing exp
        minEnergy_ = minEnergy.values.repeat(energies.shape[1],1).transpose(0,1)

        proportionalProbability = torch.exp(-(energies-minEnergy_) * beta).detach()
        Zv = torch.sum(proportionalProbability,axis=1)
        Zv_ = Zv.repeat(energies.shape[1],1).transpose(0,1)

        # Get probabilities of each energy state, and take log
        pEnergy = proportionalProbability / Zv_
        p_log_pEnergy = torch.log(torch.pow(pEnergy,pEnergy)) # log(x^x)=xlog(x), makes 0log(0) less annoying to deal with

        # Calculate average energy and entropy
        HamAvg = torch.sum(pEnergy * energies,axis=1)
        entropy = torch.sum(p_log_pEnergy)

        minusF = - HamAvg - 1/beta * entropy
        return minusF

    def collateSamples(self,results,Hamiltonian):
        records = [[record[0][x*self.nHidden:(x+1)*self.nHidden],
                    self.calculateSampleEnergy(Hamiltonian,record[0][x*self.nHidden:(x+1)*self.nHidden])] for x in range(self.nParallelAnneals) for record in results]
        states = [record[0] for record in records]
        _,uInds = np.unique(states,return_index=True,axis=0)
        return [records[iU] for iU in uInds]


    def calculateSampleEnergy(self,Hamiltonian,sample):
        # Calculate the energy for a given sample for a pytorch hamiltonian,
        # split into a linear and quadratic component
        sampleTensor = torch.Tensor(sample)
        energy = Hamiltonian[0] + torch.dot(Hamiltonian[1],sampleTensor) + \
         torch.dot(sampleTensor,torch.matmul(Hamiltonian[2],sampleTensor))

        return energy

    def calculateFreeEnergyRBM(self,beta,Hamiltonian_tensor):
        # Solve Free Energy equation for a linear Hamiltonian
        # Possible explicitly
        constantTerm = Hamiltonian_tensor[0]
        linTerms = Hamiltonian_tensor[1]
        pEnergy = (torch.exp(linTerms)+1)**-1
        pEnergy = pEnergy.detach()

        pOne = pEnergy[torch.logical_and(pEnergy!=0,pEnergy!=1)]
        pZero = 1 - pOne
        entropy = torch.dot(pOne,torch.log(pOne)) + torch.dot(1-pZero,torch.log(1-pZero))
        
        minusF = - constantTerm - torch.dot(pEnergy,linTerms) - 1/beta * entropy

        return minusF

    def evaluateQBM(self,state):
        # This function receives the current state and action choice
        # It then calculates the value of -F, and the average h values for use in weight updates
        
        # Filter for unique inputs - just copy outputs for matching inputs to save some QPU time

        n_cases = state.shape[0]
        n_outputs = max((1,self.nActions))
        minusF = torch.zeros(n_cases,n_outputs)
        threads = list()
        self.sample_results = n_cases*[[]]
        # Build Hamiltonian
        self.Hamiltonians = n_cases*[[]]
        self.Hamiltonian_tensors = n_cases*[[]]
        for iC in range(n_cases):
            self.Hamiltonians[iC],self.Hamiltonian_tensors[iC] = self.buildHamiltonian(state[iC,:])       

            # If all hamiltonian terms are linear, then we are solving a clamped RBM, which has an explicit solution
            if self.Hamiltonians[iC].is_linear():
                minusF[iC,:] = self.calculateFreeEnergyRBM(self.beta,self.Hamiltonian_tensors[iC])
                continue
            if n_cases>1:
                # Sample Hamiltonian and aggregate results
                thisThread = threading.Thread(target=self.sampleHamiltonian, args=(iC,))
                threads.append(thisThread)
                thisThread.start()
            else:
                self.sampleHamiltonian(iC)

        for thread in threads:
            thread.join()

        # Process results and calculate mean energy, h
        for iC in range(n_cases):
            if not self.Hamiltonians[iC].is_linear():
                minusF[iC,:] = self.calculateFreeEnergy(self.sample_results[iC],self.beta,self.Hamiltonian_tensors[iC])
        return minusF


    def sampleHamiltonian(self,iC):
        # Perform sampling of the Hamiltonian. This function deals with choices between simulated Annealing
        # or Quantum Annealing, and catches any failed QA attempts.

        Hamiltonian = self.Hamiltonians[iC]
        if self.SimulateAnneal:
            beta0 = min(0.1,self.beta/5)
            results_sample = self.SA_sampler.sample(Hamiltonian,num_reads=self.numReads,beta_range=[beta0, self.beta],num_sweeps=20)
            results = results_sample.record.tolist()
        else:
            if not self.embeddingLoaded:
                embedding_name = '_'.join([str(x) for x in self.net_arch])+'_Net_Arch_' + \
                            str(self.nParallelAnneals)+'_Parallel_Samples'
                this_folder = os.path.dirname(os.path.abspath(__file__))
                embedding_name = os.path.join(this_folder,'embeddings',self.DWaveSystem,embedding_name+'.txt')
                if os.path.isfile(embedding_name):
                    self.loadEmbedding(embedding_name)
            try:
                results_sample = self.QA_sampler.sample(Hamiltonian,num_reads=self.numReads)
                results = results_sample.record.tolist() # Do this inside the try, except as it sometimes errors
                self.QPUtime_micro_s += results_sample.info["timing"]["qpu_access_time"]
                if not self.embeddingLoaded:
                    self.saveEmbedding(embedding_name)
                    self.embeddingLoaded = True
            except:
                beta0 = min(0.1,self.beta/5)
                results_sample = self.SA_sampler.sample(Hamiltonian,num_reads=self.numReads,beta_range=[beta0, self.beta],num_sweeps=20)
                results = results_sample.record.tolist()
        self.sample_results[iC] = results
    
    def loadEmbedding(self,fileName:str=''):
        # Load minor embedding for problem if it is already saved within the run folder
        with open(fileName, 'r') as file:
            embedding = json.loads(file.read())
        embedding_ = {int(key):embedding[key] for key in list(embedding.keys())}
        self.QA_sampler = FixedEmbeddingComposite(DWaveSampler(solver=self.DWaveSystem),embedding=embedding_)
        self.embeddingLoaded = True

    def saveEmbedding(self,fileName:str=''):
        # Save calculated minor embedding for problem within the run folder
        if not os.path.isdir(os.path.dirname(fileName)):
            os.mkdir(os.path.dirname(fileName))
        embedding = self.QA_sampler.embedding

        with open(fileName, 'w') as file:
            file.write(json.dumps(embedding))    

    def saveWeights(self,Location):
        if not os.path.isdir(Location):
            os.mkdir(Location)
        
        wc_time = time() - self.wallClock_s
        with open(os.path.join(Location,'timings.txt'),'w') as tFile:
            tFile.write(f'Wall-Clock time (s): {wc_time:.2f}\n')
            tFile.write(f'QPU access time (s): {self.QPUtime_micro_s/1e6:.8f}')
            
        np.savetxt(os.path.join(Location,'hvWeights.txt'),self.hvWeights.detach().numpy())
        np.savetxt(os.path.join(Location,'hhWeights.txt'),self.hhWeights.detach().numpy())
        np.savetxt(os.path.join(Location,'hBias.txt'),self.hBias.detach().numpy())
        np.savetxt(os.path.join(Location,'vBias.txt'),self.vBias.detach().numpy())
        np.savetxt(os.path.join(Location,'cBias.txt'),self.cBias.detach().numpy())
        


    def loadWeights(self,Location):
        self.hvWeights = nn.Parameter(torch.from_numpy(np.loadtxt(os.path.join(Location,'hvWeights.txt'))).float())
        self.hhWeights = nn.Parameter(torch.from_numpy(np.loadtxt(os.path.join(Location,'hhWeights.txt'))).float())
        self.hBias = nn.Parameter(torch.from_numpy(np.loadtxt(os.path.join(Location,'hBias.txt'))).float())
        self.vBias = nn.Parameter(torch.from_numpy(np.loadtxt(os.path.join(Location,'vBias.txt'))).float())
        self.cBias = nn.Parameter(torch.from_numpy(
            np.array([np.loadtxt(os.path.join(Location,'cBias.txt'))])
            ).float()) # single value becomes a 0x0 array unless you add some square brackets
