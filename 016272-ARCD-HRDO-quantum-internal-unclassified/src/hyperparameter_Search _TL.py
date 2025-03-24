#%% Setup cell
from primaite.hyperparameter_runner import runCases
from primaite import PRIMAITE_PATHS
from pathlib import Path


config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config_baseline = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'
lay_down_config_complex = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex.yaml'
lay_down_config_complex_no_baseline = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex_no_baseline.yaml'
previous_agent_50ep_path_2x64 = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_50eps'
previous_agent_50ep_path_2x128 = '.\\src\\saved_agents\\baseline_2x128_NN_3E-4LR_50eps'
previous_agent_100ep_path_2x64 = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_100eps'

previous_agent_10ep_path_2x64 = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_10eps'
previous_agent_20ep_path_2x64 = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_20eps'
previous_agent_30ep_path_2x64 = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_30eps'
previous_agent_40ep_path_2x64 = '.\\src\\saved_agents\\baseline_2x64_NN_3E-4LR_40eps'


# #%% Round 1 - 100 episodes, NN,  2x64 networks, baseline, LR 
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, repo_folder='Baseline_2X64_NN_LR_100_episodes')

# #%% Round 2 - 100 episodes, NN,  2x128 networks, baseline, LR 
# hyperparameterCase = '2X128_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, repo_folder='Baseline_2X128_NN_LR_100_episodes')

# #%% Round 3 - 100 episodes, NN,  2x64 networks, complex, LR 
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes')

# #%% Round 4 - 100 episodes, NN,  2x128 networks, complex, LR 
# hyperparameterCase = '2X128_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X128_NN_LR_100_episodes')

# #%% Round 5 - 200 episodes, NN,  2x64 networks, complex, LR, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_200_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_200_episodes_50ep_trained_agent', previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 6 - 200 episodes, NN, 2X128 networks, complex, LR, using 50 ep baseline trained agent
# hyperparameterCase = '2X128_NN_LR_200_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X128_NN_LR_200_episodes_50ep_trained_agent', previous_agent_path=previous_agent_50ep_path_2x128)

# #%% Round 7 - 100 episodes, NN, 2X64 networks, complex, ENT, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_ENT_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_ENT_100_episodes_50ep_trained_agent', 
#          previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 8 - 100 episodes, NN, 2X64 networks, complex, VF Coef, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_VF_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_VF_100_episodes_50ep_trained_agent', 
#          previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 9 - 100 episodes, NN, 2X64 networks, complex, random seed, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_SEED_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_SEED_100_episodes_50ep_trained_agent', 
#          previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 10 - 100 episodes, NN, 2X64 networks, complex, random seed
# hyperparameterCase = '2X64_NN_SEED_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_SEED_100_episodes')

# #%% Round 11 - 200 episodes, NN,  2x64 networks, complex, LR, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_200_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_200_episodes_50ep_trained_agent_optmiser_reset', previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 12 - 200 episodes, NN,  2x64 networks, complex, LR, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_200_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_200_episodes_50ep_trained_agent_redo', previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 13 - 100 episodes, NN,  2x64 networks, complex, LR, using 50 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes_50ep_trained_agent', previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 14 - 100 episodes, NN,  2x64 networks, complex, LR, using 100 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes_100ep_trained_agent', previous_agent_path=previous_agent_100ep_path_2x64)

# #%% Round 15 - 100 episodes, NN, 2x64 networks, complex no baseline(S2), LR, using 50 ep baseline trained agent
# hyperparameterCase = '2x64_NN_LR_100_episodes'
# runCases(hyperparameterCase, config_location, lay_down_config_complex_no_baseline, repo_folder='S2_2X64_NN_LR_100_episodes_50ep_trained_agent', previous_agent_path=previous_agent_50ep_path_2x64)

# #%% Round 16 - 100 episodes, NN, 2x64 networks, S2, LR, using 100 ep baseline trained agent
# hyperparameterCase = '2x64_NN_LR_100_episodes'
# runCases(hyperparameterCase, config_location, lay_down_config_complex_no_baseline, repo_folder='S2_2X64_NN_LR_200_episodes_100ep_trained_agent', previous_agent_path=previous_agent_100ep_path_2x64)

# #%% Round 17 - 100 episodes, NN, 2x64 networks, S2, LR 
# hyperparameterCase = '2x64_NN_LR_100_episodes'
# runCases(hyperparameterCase, config_location, lay_down_config_complex_no_baseline, repo_folder='S2_2X64_NN_LR_100_episodes')

# #%% Round 18 - 100 episodes, NN,  2x64 networks, complex, LR, using 10 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes_10ep_trained_agent', previous_agent_path=previous_agent_10ep_path_2x64)

# #%% Round 19 - 100 episodes, NN,  2x64 networks, complex, LR, using 20 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes_20ep_trained_agent', previous_agent_path=previous_agent_20ep_path_2x64)

# #%% Round 20 - 100 episodes, NN,  2x64 networks, complex, LR, using 30 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes_30ep_trained_agent', previous_agent_path=previous_agent_30ep_path_2x64)

# #%% Round 21 - 100 episodes, NN,  2x64 networks, complex, LR, using 40 ep baseline trained agent
# hyperparameterCase = '2X64_NN_LR_100_episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_complex, repo_folder='Complex_2X64_NN_LR_100_episodes_40ep_trained_agent', previous_agent_path=previous_agent_40ep_path_2x64)

# #%% Round 22 - 3 episodes, NN,  2x64 networks, baseline, LR, test saving capability
# hyperparameterCase = '2x64_20_episodes_test'
# runCases(hyperparameterCase,config_location,lay_down_config_baseline, repo_folder='save_test_Baseline_2X64_NN_LR_20_episodes')