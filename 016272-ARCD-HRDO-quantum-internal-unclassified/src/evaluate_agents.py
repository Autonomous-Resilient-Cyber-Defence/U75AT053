from primaite.agent_utilities import evaluateModel

config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'

# Baseline NN
hyperparameterCase = '2x64_50_Episodes'

NN_location = ".\\src\\previous_sessions\\Converted_NN\\Baseline_2X64_NN_LR_100_episodes\\sessions\\NN\\learning\\"
DBM_location = ".\\src\\previous_sessions\\Converted_NN\\Baseline_2X64_NN_LR_100_episodes\\sessions\\DBM\\learning\\"
Hypernet_location = ".\\src\\previous_sessions\\Converted_NN\\Baseline_2X64_NN_LR_100_episodes\\sessions\\DBM_Hypernet\\learning\\"
# evaluateModel(config_location,lay_down_config,
#                 hyperparameterCase,0,NN_location,
#                 num_episodes=100)
# evaluateModel(config_location,lay_down_config,
#                 hyperparameterCase,2,Hypernet_location,
#                 num_episodes=100)
evaluateModel(config_location,lay_down_config,
                hyperparameterCase,1,DBM_location,
                num_episodes=10)
