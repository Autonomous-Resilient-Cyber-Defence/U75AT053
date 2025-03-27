from primaite.agent_utilities import convertModel_syntheticData, convertModel_Hypernet

config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'

# Baseline NN
OldHyperparameterCase = '2x64_50_Episodes'
NewHyperparameterCase = '2x64_50_Episodes'

model_location = ".\\src\\previous_sessions\\Baseline_2X64_NN_LR_100_episodes\\sessions\\LR_3E-4\\learning\\"
# convertModel_Hypernet(NewHyperparameterCase,2,config_location,lay_down_config,
#                                OldHyperparameterCase,0,model_location,
#                                epochs=120,save_frequency=10,lr=3E-4)
convertModel_syntheticData(NewHyperparameterCase,1,config_location,lay_down_config,
                               OldHyperparameterCase,0,model_location,
                               epochs=1000,dataDownscale=400,save_frequency=100)
