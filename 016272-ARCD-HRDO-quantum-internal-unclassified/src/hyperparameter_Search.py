from primaite.hyperparameter_runner import runCases

config_location = '.\\src\\primaite\\config\\_package_data\\training\\'
lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\lay_down_config_1_DDOS_basic.yaml'
lay_down_config_doubleAgent = '.\\src\\primaite\\config\\_package_data\\lay_down\\1_DDOS_with_extra_red_agents\\lay_down_config_1_DDOS_basic_extra_red_POL.yaml'
lay_down_config_2 = '.\\src\\primaite\\config\\_package_data\\lay_down\\lay_down_config_2_DDOS_basic.yaml'
lay_down_config_16 = '.\\src\\primaite\\config\\_package_data\\lay_down\\custom_16_node_lay_down.yaml'
lay_down_config_20 = '.\\src\\primaite\\config\\_package_data\\lay_down\\custom_20_node_lay_down.yaml'
lay_down_config_30 = '.\\src\\primaite\\config\\_package_data\\lay_down\\custom_30_node_lay_down.yaml'
lay_down_config_realistic = '.\\src\\primaite\\config\\_package_data\\lay_down\\real_life_lay_down.yaml'

# %% Round 1 - Action Space
# # Case 1, DBM Policy Network
# hyperparameterCase = 'Action_Space_DBM'
# runCases(hyperparameterCase,config_location,lay_down_config)

# # Case 2, Baseline NN
# hyperparameterCase = 'Action_Space_NN'
# runCases(hyperparameterCase,config_location,lay_down_config)

# %% Round 2 - Learning Rate
# # Case 1, Value network
# hyperparameterCase = 'Learning_Rate_Value'
# runCases(hyperparameterCase,config_location,lay_down_config)

# # Case 2, Policy network
# hyperparameterCase = 'Learning_Rate_Policy'
# runCases(hyperparameterCase,config_location,lay_down_config)

#%% Round 3 - Beta
# # Case 1, Value network
# hyperparameterCase = 'Beta_Value'
# runCases(hyperparameterCase,config_location,lay_down_config,[0,1,2])
# runCases(hyperparameterCase,config_location,lay_down_config,[3,4])

# # Case 2, Policy network
# hyperparameterCase = 'Beta_Policy'
# runCases(hyperparameterCase,config_location,lay_down_config,[0,1,2])
# runCases(hyperparameterCase,config_location,lay_down_config,[3,4])

# #%% Round 4 - Policy and Value Network
# hyperparameterCase = 'Policy_Value'
# runCases(hyperparameterCase,config_location,lay_down_config)

# #%% Round 5- Optimised parameters, harder laydown
# hyperparameterCase = '2x64_Optimised'
# runCases(hyperparameterCase,config_location,lay_down_config_doubleAgent,[1,3])

#%% Round 6- Deeper network LR searches
# # Case 1 - NN
# hyperparameterCase = '4x64_NN_LR'
# runCases(hyperparameterCase,config_location,lay_down_config)

# #Case 2 - Value net DBM
# hyperparameterCase = '4x64_Value_LR'
# runCases(hyperparameterCase,config_location,lay_down_config)

# #Case 3 - Policy net DBM
# hyperparameterCase = '4x64_Policy_LR'
# runCases(hyperparameterCase,config_location,lay_down_config)

# #Case 4 - Policy + Value net DBM
# hyperparameterCase = '4x64_Policy_Value_LR'
# runCases(hyperparameterCase,config_location,lay_down_config)

#%% Round 7 - Random Red Agent, 2x64 networks, using optimised params
# hyperparameterCase = '2x64_Random_Red_Agent'
# runCases(hyperparameterCase,config_location,lay_down_config)

#%% Round 8 - 10 Node Laydown, 2x64 networks, using optimised params
# hyperparameterCase = '2x64_100_Episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_2)

# hyperparameterCase = '2x64_100_Episodes_5ACL'
# runCases(hyperparameterCase,config_location,lay_down_config_16)
# hyperparameterCase = '2x64_100_Episodes_5ACL'
# runCases(hyperparameterCase,config_location,lay_down_config_20)

# N.B. memory saving updates applied to DBM code at this point
# hyperparameterCase = '2x64_100_Episodes' # Trains in 700 steps with NN, 3E-5 LR
# runCases(hyperparameterCase,config_location,lay_down_config_16)
# hyperparameterCase = '2x64_1000_Episodes'
# runCases(hyperparameterCase,config_location,lay_down_config_16)

# #%% Round 9 - 73 Node Laydown, 2x64 networks, using optimised params
hyperparameterCase = '2x64_RealisticRuns'
runCases(hyperparameterCase,config_location,lay_down_config_realistic,[0])

# #%% Round 10 - 150 DBM episodes, 1000 NN, 2x64 networks, 30 ACL for 10 and 16 node networks
# hyperparameterCase = '2x64_150_DBM_1000_NN'
# runCases(hyperparameterCase,config_location,lay_down_config_16)

# hyperparameterCase = '2x64_150_DBM_1000_NN'
# runCases(hyperparameterCase,config_location,lay_down_config_2)
