import torch
from boltzmannMachines.DBM import DBM

class DBM_action(DBM):
    # DBM_action class a subclass of DBM
    # This includes additonal functionality to get multiple outputs from the class
    def buildHamiltonian(self,state):       
        ## Get distinct linear hamiltonian components for each action
        action = torch.zeros(self.nActions)
        _, Hamiltonian_tensors = super().buildHamiltonian(state,action)
        
        constOffset = self.vBias[-self.nActions:]
        linOffset = torch.transpose(self.hvWeights[-self.nActions:,:],0,1)
                
        ## Build baseline Hamiltonian with no actions and add a contraint penalty term
        # There are too many actions to define an embedding for a network including the action nodes
        # Suggest applying fixed linear bias to hidden nodes. Some options:
        # - The maximum weight
        # - The minimum weight
        # - The mean weight
        # - The median
        # Try mean first - easiest to apply
        Hamiltonian_mean = self.tileHamiltonian(Hamiltonian_tensors[1] - linOffset.mean(axis=1),Hamiltonian_tensors[2],self.nParallelAnneals)
        Hamiltonian = Hamiltonian_mean

        Hamiltonian_tensors += [constOffset, linOffset]
        return Hamiltonian, Hamiltonian_tensors

    def calculateSampleEnergy(self, Hamiltonian_tensors, sample):
        sampleTensor = torch.Tensor(sample)

        constantTerm = Hamiltonian_tensors[0] - Hamiltonian_tensors[3]
        linTerms = torch.matmul(Hamiltonian_tensors[1],sampleTensor) - torch.matmul(Hamiltonian_tensors[4].transpose(1,0),sampleTensor)
        quadTerm = torch.dot(sampleTensor,torch.matmul(Hamiltonian_tensors[2],sampleTensor)).repeat(self.nActions)
        return constantTerm + linTerms + quadTerm

    def calculateFreeEnergyRBM(self,beta,Hamiltonian_tensors):
        # Solve Free Energy equation for a linear Hamiltonian
        # Possible explicitly
        constantTerm = Hamiltonian_tensors[0] - Hamiltonian_tensors[3]
        linTerms = Hamiltonian_tensors[1] - Hamiltonian_tensors[4].transpose(1,0)
        pOne = (torch.exp(linTerms)+1)**-1
        pOne = pOne.detach()
        pZero = 1 - pOne

        p_log_pOne = torch.log(torch.pow(pOne,pOne))
        p_log_pZero = torch.log(torch.pow(pZero,pZero))

        entropy = torch.sum(p_log_pOne + p_log_pZero,axis=1)
        
        minusF = - constantTerm - torch.sum(pOne*linTerms,axis=1) - 1/beta * entropy

        return minusF
    