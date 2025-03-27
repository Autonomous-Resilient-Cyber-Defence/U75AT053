#%% Setup cell
from primaite.hyperparameter_runner import runCases
from primaite import PRIMAITE_PATHS
from pathlib import Path


config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config_baseline = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'
lay_down_config_complex = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex.yaml'
previous_agent_50ep_path = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\baseline_50ep_agent'



# #%% Round 1 - 100 episodes, NN,  2x64 networks, baseline, LR 
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, repo_folder='Baseline_2X64_NN_LR_100_episodes')

# #%% Round 2 - 100 episodes, NN,  2x64 networks, complex, LR 
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes')

# #%% Round 2 - 100 episodes, NN,  2x64 networks, complex, LR, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, previous_agent_path=previous_agent_50ep_path)

#%% Round 2 - 100 episodes, NN,  2x64 networks, complex, LR 
hyperparameterCase = 'Hypernet_LR'
runCases(hyperparameterCase,config_location,lay_down_config_baseline, repo_folder='Hypernet_LR',runCases=[1,2])
