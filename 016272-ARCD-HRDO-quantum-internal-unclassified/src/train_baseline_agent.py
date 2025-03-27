from primaite.hyperparameter_runner import runCases

config_location = '.\\src\\primaite\\config\\_package_data\\training\\TransferLearning\\'
lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'

# hyperparameterCase = '2x64_50_Episodes'
# runCases(hyperparameterCase,config_location,lay_down_config,[1])

hyperparameterCase = '2x64_25_Episodes'
runCases(hyperparameterCase,config_location,lay_down_config,[0],repo_folder='Baseline_2X64_25_episodes_')

