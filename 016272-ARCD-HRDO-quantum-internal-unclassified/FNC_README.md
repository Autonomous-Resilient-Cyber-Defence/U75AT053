# FNC: README Quantum Transfer Agents

This branch is for testing the quantum transfer learning capabilities of PrimAITE AI agents. Some packages require access to the D-WAVE quantum annealer, such as dwave.system package in the src/boltzmannMachines/DBM.py file. It may be easier to setup the enviroment following the dev install option in README.md, and then installing the required packages as you attempt to run the code. There shouldn't be too many.

The TL runs were performed in src/run_TL.py. Within you can find references to what searches were performed and where the results are located.

To allow PrimAITE to train TL agents, its main code was tweaked so that it can create a fresh agent and then replace the weights of this fresh NN with the weights from a previously trained agent.

