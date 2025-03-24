from primaite.model_converter import convertModel_syntheticData, convertModel_Hypernet

config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning'
lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\lay_down_config_1_DDOS_basic.yaml'

# Baseline NN
OldHyperparameterCase = '2x64_50_Episodes'
NewHyperparameterCase = '2x64_50_Episodes'

model_location = "C:\\Users\\dtk\\primaite\\2.0.0\\sessions\\2024-12-03\\2024-12-03_09-11-40\\learning\\"
convertModel_Hypernet(NewHyperparameterCase,2,config_location,lay_down_config,
                               OldHyperparameterCase,0,model_location,
                               epochs=20,save_frequency=10,lr=1E-3)
convertModel_syntheticData(NewHyperparameterCase,1,config_location,lay_down_config,
                               OldHyperparameterCase,0,model_location,
                               epochs=1000,dataDownscale=1000,save_frequency=50)
