#%% Setup cell
from primaite.hyperparameter_runner import runCases
from primaite import PRIMAITE_PATHS
from pathlib import Path


config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning'
lay_down_config_baseline = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'
lay_down_config_complex = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex.yaml'
lay_down_config_S2 = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex_no_baseline.yaml'
previous_agent_baseline = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_50eps'
previous_agent_S2 = '.\\src\\saved_agents\\S2_2X64_NN_3E-4LR_50eps'
previous_agent_baseline_then_S2 = '.\\src\\saved_agents\\baseline_50eps_S2_100ps_2x64_NN_6E-4LR'


#note agent_evaluation is not a hp case but contains the training config for agent evaluation 

# #%% Round 1 - 50 episodes, NN, 2x64 networks, baseline, 50 ep trained baseline agent
# hyperparameterCase = 'agent_evaluation'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, 
# repo_folder='EVALUATE_50ep_baseline_on_baseline', agent_to_evaluate_path=previous_agent_baseline)

# #%% Round 2 - 50 episodes, NN, 2x64 networks, baseline, 50 ep trained S2 agent
# hyperparameterCase = 'agent_evaluation'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, 
# repo_folder='EVALUATE_50ep_S2_on_baseline', agent_to_evaluate_path=previous_agent_S2)

# #%% Round 3 - 50 episodes, NN, 2x64 networks, baseline, 50 ep baseline then 100ep S2 trained agent
# hyperparameterCase = 'agent_evaluation'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, 
# repo_folder='EVALUATE_Baseline_50ep_S2_100ep_on_baseline', agent_to_evaluate_path=previous_agent_baseline_then_S2)

# #%% Round 3 - 50 episodes, NN, 2x64 networks, S2, 50 ep baseline then 100ep S2 trained agent
# hyperparameterCase = 'agent_evaluation'
# runCases(hyperparameterCase,config_location,lay_down_config_S2, 
# repo_folder='EVALUATE_Baseline_50ep_S2_100ep_on_S2', agent_to_evaluate_path=previous_agent_baseline_then_S2)
