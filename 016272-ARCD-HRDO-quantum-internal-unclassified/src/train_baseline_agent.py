from primaite.hyperparameter_runner import runCases

config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\lay_down_config_1_DDOS_basic.yaml'

hyperparameterCase = '2x64_50_Episodes'
runCases(hyperparameterCase,config_location,lay_down_config)
