from pathlib import Path
from primaite.primaite_session import PrimaiteSession
from primaite import PRIMAITE_PATHS
from primaite import load_agent
from datetime import datetime
from boltzmannMachines.DBM import DBM
from boltzmannMachines.DBM_action import DBM_action
from boltzmannMachines.DBM_Hypernet import DBM_Hypernet
import shutil

# JS11: To evaluate a previous agent, you must set session_type to be EVAL in the training_config.
def runCases(hyperparameterCase,training_config_location,lay_down_config, runCases=None, 
previous_agent_path=None, repo_folder=None, agent_to_evaluate_path=None):
    # Setting repo path folder to save sessions
    if repo_folder is not None:
        repo_folder = Path(f'.\\src\\previous_sessions\\{repo_folder}')
        if not repo_folder.exists():
            repo_folder.mkdir()
        else:
            raise Exception('This folder already exists in the repo, please rename.')
        repo_session_folder = Path.joinpath(repo_folder, Path('session_agent_checkpoints')) 
        average_rewards_repo_folder = Path.joinpath(repo_folder, Path('average_rewards'))
        weights_repo_folder = Path.joinpath(repo_folder, Path('weights'))

    # Convert training data to path objects and join
    if not hyperparameterCase is Path:
        hyperparameterCase = Path(hyperparameterCase)
    if not training_config_location is Path:
        training_config_location = Path(training_config_location)
    trainingFolder = Path.joinpath(training_config_location,hyperparameterCase)

    # Get timestamp and make a results folder
    hyperparameter_run_path = Path.joinpath(PRIMAITE_PATHS.user_sessions_path.parent,Path('Hyperparameters'))
    if not hyperparameter_run_path.exists():
        hyperparameter_run_path.mkdir()
    casePath = Path.joinpath(hyperparameter_run_path,hyperparameterCase)
    if not casePath.exists():
        casePath.mkdir()
    timestamp = Path(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    resultsPath = Path.joinpath(casePath,timestamp)
    resultsPath.mkdir()

    count = -1
    for training_config in trainingFolder.glob('*.yaml'):
        count+=1
        if runCases is not None:
            if count not in runCases:
                continue
        # Run primaite for this case
        # Copied defaults and code from main.run()
        # Added in the option to use a previous agent
        session = PrimaiteSession(
            training_config, lay_down_config, None, False, 
            False, previous_agent_path=previous_agent_path, agent_to_evaluate_path=agent_to_evaluate_path
        )
        session.setup()
        session.learn()
        session.evaluate() #JS11: include evaluate capability

        # Copy training config
        shutil.copy(training_config,resultsPath)

        #setting primaite_results path depending on if we're training or evaluating 
        if agent_to_evaluate_path is not None:
            primaite_results_path = session.evaluation_path
        else:
            primaite_results_path = session.learning_path

        # Copy results files
        average_reward = Path.joinpath(primaite_results_path,Path(f"average_reward_per_episode_{session.timestamp_str}.csv"))
        #all_transactions = Path.joinpath(primaite_results_path,Path(f"all_transactions_{session.timestamp_str}.csv"))

        thisCase = training_config.name[:-5]
        shutil.copy(average_reward,Path.joinpath(resultsPath,Path(f"average_reward_per_episode_{thisCase}.csv")))
        # shutil.copy(all_transactions,Path.joinpath(resultsPath,Path(f"all_transactions_{thisCase}.csv")))

        # Save weights
        weights_folder = Path.joinpath(resultsPath,thisCase)
        load_agent.save_agent_session(session,weights_folder)

        # save session to repo if required
        if repo_folder is not None:
            #save agent checkpoints if learning
            if agent_to_evaluate_path is None:
                # note that checkpoints folder is only created if number of eps for training is greater than the checkpoint episode number.
                #session_path = session.session_path
                repo_case_agent_folder = Path.joinpath(repo_session_folder, Path(f'{thisCase}'))
                agent_checkpoints_path = Path.joinpath(primaite_results_path, 'checkpoints')
                # shutil.copytree(session_path, repo_case_folder)
                shutil.copytree(agent_checkpoints_path, repo_case_agent_folder)

            #copy the average_rewards in the repo
            if not average_rewards_repo_folder.exists():
                average_rewards_repo_folder.mkdir()
            shutil.copy(average_reward, Path.joinpath(average_rewards_repo_folder, Path(f"average_reward_per_episode_{thisCase}.csv")))

            #copy the weights folder to repo
            shutil.copytree(weights_folder, Path.joinpath(weights_repo_folder, f'{thisCase}_weights'))
