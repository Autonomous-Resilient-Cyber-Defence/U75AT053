# FNC: README ARCD PPO Transfer Learning U75a Task 53

## General Information

This repo is for testing the transfer learning (TL) capabilities of PPO AI defence agents that use quantum solved Deep Boltzmann Machines (DBMs) and classically solved neural network (NNs).

## The Quantum Runs

We performed the quantum TL runs in src/run_TL.py. If you can't see this file then you are likely in the branch that contains teh classical runs. The classical runs were kept in a seperate branch to the dev branch due to the volumne of classical hyperparameter runs performed. Within src/run_TL.py you can find references to what searches were performed and where the results are located. These results are stored in the dev branch of the repo. Note that these runs can't be repeated because it requires access to a D-WAVE quantum sampler. 

## The Classical Runs

We performed the classical hyperparameter runs for transfer learning in src/hyperparameter_Search_TL.py. Within this file you can find references to what searches were performed and where the results are located.

We performed the classical agent evaluation runs in src/evaluate_agents_TL.py. Within this file you can find references to what searches were performed and where the results are located.

The result plots were generated in src/Plotting/transferlearning/plot_from_Hyperparameters_folder_script.ipynb. Within this file you can find references to what searches were performed and where the results are located. 

To allow PrimAITE to train TL agents, its main code was tweaked so that it can create a fresh agent and then replace the weights of this fresh NN with the weights from a previously trained agent.

## Runs from the previous task 
For the previous task,  Quantum Reinforcement Learning for Data Efficient Decision Making, we performed the quantum and classical hyperparameter runs within src/hyperparameter_Search.py. Within this file you can find references to what searches were performed.

The result plots were generated in src/Plotting/plot_from_Hyperparameters_folder_script.ipynb.

## How to set up this repo's environment

There is an environment.yaml file with an exported environment that worked for the FNC team. Although it can be used as a reference for package versions, simply installing the environment via this yaml file wasn't repeatable. Therefore, follow the below steps:

1. Create and activate conda environment with python = 3.10. For example: 

>conda create --name prim_env python=3.10

>conda activate prim_env 

2. Downgrade the pip, wheel and setuptools packages: 

>python -m pip install pip==23.0.1 --upgrade  

>pip install wheel==0.38.4 --upgrade 

>pip install setuptools==66 --upgrade 

3. Make sure you are in the 016272-ARCD-HRDO-quantum-internal-unclassified folder.

4. Perform the developer install in editable mode:

>python -m pip install -e .

5. Install the following packages:

>pip install kaleido==0.1.0post1
>pip install gitpython==3.1.40
>pip install dimod==0.12.12
>pip install dwave-ocean-sdk==6.6.0

6. You should now be able to run python src\hyperparameter_Search_TL.py. You will have to change the repo_folder parameter in:
#%% Round 1 - 100 episodes, NN,  2x64 networks, baseline, LR 
hyperparameterCase = '2X64_NN_LR_100_episodes'
runCases(hyperparameterCase,config_location,lay_down_config_baseline, repo_folder='test')

if it already exits.

7. If it still doesn't work, try running:
> primaite setup

## Note 

Please nore that this branch contains in-development code / code that needs to be ran on D-Wave. Therefore, there may be unforseen issues with this codebase that may require user edits / debugging. 
