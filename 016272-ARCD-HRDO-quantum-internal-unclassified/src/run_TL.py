#%% Setup cell
from primaite.hyperparameter_runner import runCases
from primaite import PRIMAITE_PATHS
from pathlib import Path


config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config_baseline = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'
lay_down_config_alternative = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_Alternative.yaml'
lay_down_config_complex = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex.yaml'

previous_agent_NN = '.\\src\\previous_sessions\\Baseline_2X64_NN_LR_100_episodes\\sessions\\LR_3E-4\\learning\\'
previous_agent_NN_50 = '.\\src\\previous_sessions\\Baseline_2X64_50_episodes\\NN\\learning\\'
previous_agent_NN_25 = '.\\src\\previous_sessions\\Baseline_2X64_25_episodes\\sessions\\Neural_Network\\learning\\'
previous_agent_DBM = '.\\src\\previous_sessions\\Baseline_2X64_50_episodes\\DBM\\Value+Policy_DBM\\'
previous_agent_NN_Hypernet = '.\\src\\previous_sessions\\Converted_NN\\Baseline_2X64_NN_25_episodes\\sessions\\DBM_Hypernet\\learning\\'

#%% NNs  
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_alternative, previous_agent_path=previous_agent_NN,runCases=[3],repo_folder='NN_TL_100eps_alt_')
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_alternative, previous_agent_path=previous_agent_NN_50,runCases=[3],repo_folder='NN_TL_50eps_alt')
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_alternative, previous_agent_path=None,runCases=[3],repo_folder='NN_TL_untrained_alt')
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_NN,runCases=[3],repo_folder='NN_TL_100eps')
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_NN_50,runCases=[3],repo_folder='NN_TL_50eps')
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_NN_25,runCases=[3],repo_folder='NN_TL_25eps')
# runCases('2X64_NN_LR_100_episodes',config_location,lay_down_config_complex, previous_agent_path=None,runCases=[3],repo_folder='NN_TL_untrained')

#%% DBMs
# runCases('2x64_100_Episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_DBM,runCases=[1],repo_folder='DBM_TL')
# runCases('2x64_100_Episodes',config_location,lay_down_config_complex, previous_agent_path=None,runCases=[1],repo_folder='DBM_TL_untrained')
# previous_agent_DBM = '.\\src\\previous_sessions\\Baseline_2X64_25_episodes\\weights\\Value+Policy_DBM_weights\\'
# runCases('2x64_50_Episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_DBM,runCases=[1],repo_folder='DBM_25_TL')

# Alt
# runCases('2x64_100_Episodes',config_location,lay_down_config_alternative, previous_agent_path=previous_agent_DBM,runCases=[1],repo_folder='DBM_TL_alt')
# runCases('2x64_100_Episodes',config_location,lay_down_config_alternative, previous_agent_path=None,runCases=[1],repo_folder='DBM_TL_untrained_alt')

#%% Hypernets
# runCases('2x64_50_Episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_NN_Hypernet,runCases=[2],repo_folder='DBM_25_Hypernet_TL')
# runCases('Hypernet_LR',config_location,lay_down_config_complex, previous_agent_path=previous_agent_NN_Hypernet,runCases=[3],repo_folder='DBM_25_Hypernet_TL_3E-6')
# runCases('Hypernet_LR',config_location,lay_down_config_complex, previous_agent_path=previous_agent_NN_Hypernet,runCases=[1],repo_folder='DBM_25_Hypernet_TL_3E-4')
# runCases('2x64_100_Episodes',config_location,lay_down_config_complex, previous_agent_path=previous_agent_DBM,runCases=[2],repo_folder='DBM_Hypernet_TL_Untrained')



