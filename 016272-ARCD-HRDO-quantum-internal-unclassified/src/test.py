from primaite.main import run

training_config = '.\\src\\primaite\\config\\_package_data\\training\\training_config_main.yaml'
lay_down_config_baseline = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_baseline.yaml'
lay_down_config_complex = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_complex.yaml'
lay_down_config_no_red = '.\\src\\primaite\\config\\_package_data\\lay_down\\TL_custom_lay_downs\\custom_15_node_lay_down_no_red.yaml'
#lay_down_config = '.\\src\\primaite\\config\\_package_data\\lay_down\\lay_down_config_1_DDOS_basic.yaml'
previous_agent_path_2ep_baseline = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\baseline_2ep_agent'
previous_agent_path_50ep_baseline = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\baseline_50ep_agent'
previous_agent_path_2ep_complex = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\complex_2ep_agent'
previous_agent_path_0ep_no_red = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\no_red_0ep_agent'
previous_agent_path_5ep_no_red = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\no_red_5ep_agent'
previous_agent_path_25ep_no_red = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\agents\\no_red_25ep_agent'

two_ddos_laydown = '.\\src\\primaite\\config\\_package_data\\lay_down\\lay_down_config_2_DDOS_basic.yaml'
#run(training_config,lay_down_config_baseline, previous_agent_path=previous_agent_path_50ep_baseline)
#run(training_config,lay_down_config_complex, previous_agent_path=previous_agent_path_50ep_baseline)
#run(training_config, lay_down_config_no_red)
#run(training_config,lay_down_config_baseline)
# run(training_config,lay_down_config_baseline)
# run(training_config,lay_down_config_complex)

run(training_config, lay_down_config_complex, agent_to_evaluate_path='C:\\Users\\js11\\Documents\\ARCD - Transfer Learning\\saved_agents\\baseline_2x64_NN_3E-4LR_50eps')

""" from pathlib import Path
previous_agent_path = 'C:\\Users\\js11\\primaite\\2.0.0\\sessions\\2024-12-05\\baseline_agent_5ep_trained'
previous_agent_path = Path(previous_agent_path)
print(list(previous_agent_path.glob('*')))
previous_agent_load_file = next(previous_agent_path.rglob("*.zip"), None) """