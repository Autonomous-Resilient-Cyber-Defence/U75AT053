# FNC: README Classical transfer agents

This branch for testing the classical transfer learning capabilities of PrimAITE AI agents uses the environment in environment.yml. There is a chance that all of the contained packages may not be necessary because some were used for exploratory work. It may be easier to setup the enviroment following the dev install option in README.md, and then installing the required packages as you attempt to run the code. There shouldn't be too many.

The hyperparameter runs were performed in src/hyperparamter_Search_TL.py. Within you can find references to what searches were performed and where the results are located.

The result plots were generated in src/Plotting/transferlearning/plot_from_Hyperparameters_folder_script.ipynb.

To allow PrimAITE to train TL agents, its main code was tweaked so that it can create a fresh agent and then replace the weights of this fresh NN with the weights from a previously trained agent.
