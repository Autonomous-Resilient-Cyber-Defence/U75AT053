from pathlib import Path
from primaite.primaite_session import PrimaiteSession
from primaite import PRIMAITE_PATHS
from datetime import datetime
from boltzmannMachines.DBM import DBM
from boltzmannMachines.DBM_action import DBM_action
from stable_baselines3 import PPO
from primaite import load_agent
import shutil
import pandas as pd
import numpy as np
import torch as tf

def convertModel_syntheticData(newModelCase,newCaseI,training_config_location,lay_down_config,
                               oldModelCase,oldCaseI,model_location,epochs=1000,dataDownscale=1000,save_frequency=50):
    
    # Convert paths to path objects and join
    if not newModelCase is Path:
        newModelCase = Path(newModelCase)
    if not oldModelCase is Path:
        oldModelCase = Path(oldModelCase)
    if not training_config_location is Path:
        training_config_location = Path(training_config_location)
    if not model_location is Path:
        model_location = Path(model_location)
    newFolder = Path.joinpath(training_config_location,newModelCase)
    oldFolder = Path.joinpath(training_config_location,oldModelCase)

    # Get config files for new and old model
    newConfig = [case for case in newFolder.glob('*.yaml')][newCaseI]
    oldConfig = [case for case in oldFolder.glob('*.yaml')][oldCaseI]

    # Start primaite sessions to initialise the agents
    session_new = PrimaiteSession(
        newConfig, lay_down_config, None, False, False
    )
    session_new.setup()

    # Load in the old agent and data
    session_old = load_agent.load_agent_session(oldConfig, lay_down_config,model_location)
    nFeatures = session_old._agent_session._agent.policy.mlp_extractor.value_net[0].in_features
    
    trainingSteps = pd.read_csv([file for file in (model_location.glob('all_transactions*.csv'))][0],header=0,names=[str(i) for i in range(nFeatures+5)])
    trainingSteps = trainingSteps[[str(i) for i in range(5,(nFeatures+5))]]
    
    trainingData_x = tf.from_numpy(trainingSteps.to_numpy(dtype='float32'))
    trainingData_FE_value = session_old._agent_session._agent.policy.mlp_extractor.value_net.forward(trainingData_x)
    trainingData_y_value = session_old._agent_session._agent.policy.value_net.forward(trainingData_FE_value).detach()

    trainingData_FE_policy = session_old._agent_session._agent.policy.mlp_extractor.policy_net.forward(trainingData_x)
    trainingData_y_policy = session_old._agent_session._agent.policy.action_net.forward(trainingData_FE_policy).detach()

    # Apply scaling factor to number 
    nMeas = trainingData_x.shape[0]
    nTrain = int(nMeas/dataDownscale+0.99)

    session_new.session_path.joinpath(f'learning\\checkpoints\\').mkdir()
    for iE in range(epochs):
        these_inds = np.random.random_integers(0,nMeas-1,nTrain)

        this_data_x = trainingData_x[these_inds,:]
        this_data_y_value  = trainingData_y_value[these_inds]
        this_data_y_policy = trainingData_y_policy[these_inds,:]

        pred_FE_value = session_new._agent_session._agent.policy.mlp_extractor.value_net.forward(this_data_x)
        pred_y_value = session_new._agent_session._agent.policy.value_net.forward(pred_FE_value)

        pred_FE_policy = session_new._agent_session._agent.policy.mlp_extractor.policy_net.forward(this_data_x)
        pred_y_policy = session_new._agent_session._agent.policy.action_net.forward(pred_FE_policy)

        loss_value = sum((pred_y_value - this_data_y_value)**2)
        loss_policy = sum(sum((pred_y_policy - this_data_y_policy)**2))

        loss = loss_policy + loss_value * session_new._agent_session._agent.vf_coef
        # Optimization step
        session_new._agent_session._agent.policy.optimizer.zero_grad()
        loss.backward()
        session_new._agent_session._agent.policy.optimizer.step()

        if iE % save_frequency == 0:
            print(f'Epoch {iE:.0f}, loss={loss.detach().numpy()[0]:.3f}')  
            session_new._agent_session._agent.policy.save(session_new.session_path.joinpath(f'learning\\checkpoints\\sb3ppo_{iE:.0f}.zip'))
            # session_new.evaluate()
    load_agent.save_agent_session(session_new,session_new.session_path.joinpath(f'learning\\'))
    session_new._agent_session._training_config.num_eval_episodes = 10
    session_new.evaluate()           

def convertModel_Hypernet(newModelCase,newCaseI,training_config_location,lay_down_config,
                               oldModelCase,oldCaseI,model_location,epochs=20,save_frequency=10,lr=1E-3):
    
    # Convert paths to path objects and join
    if not newModelCase is Path:
        newModelCase = Path(newModelCase)
    if not oldModelCase is Path:
        oldModelCase = Path(oldModelCase)
    if not training_config_location is Path:
        training_config_location = Path(training_config_location)
    if not model_location is Path:
        model_location = Path(model_location)
    newFolder = Path.joinpath(training_config_location,newModelCase)
    oldFolder = Path.joinpath(training_config_location,oldModelCase)


    # Get config files for new and old model
    newConfig = [case for case in newFolder.glob('*.yaml')][newCaseI]
    oldConfig = [case for case in oldFolder.glob('*.yaml')][oldCaseI]

    # Start primaite sessions to initialise the agents
    session_new = PrimaiteSession(
        newConfig, lay_down_config, None, False, False
    )
    session_new.setup()

    # Load in old agent
    session_old = load_agent.load_agent_session(oldConfig, lay_down_config,model_location)

    # Get the trained NNs
    policy_extractor = session_old._agent_session._agent.policy.mlp_extractor.policy_net
    value_extractor = session_old._agent_session._agent.policy.mlp_extractor.value_net
    policy_net = session_old._agent_session._agent.policy.action_net
    value_net = session_old._agent_session._agent.policy.value_net

    # Make output paths for training hypernet
    session_new.session_path.joinpath(f'learning\\checkpoints\\').mkdir()
    action_path = session_new.session_path.joinpath(f'learning\\checkpoints\\action_net\\')
    value_path = session_new.session_path.joinpath(f'learning\\checkpoints\\value_net\\')
    action_path.mkdir()
    value_path.mkdir()

    # Train hypernets to match NNs
    session_new._agent_session._agent.policy.action_net.loadNN([policy_extractor, policy_net],trainEpochs=epochs,
                                                               storeFrequency=save_frequency,storeLocation=action_path,learningRate=lr)
    session_new._agent_session._agent.policy.value_net.loadNN([value_extractor, value_net],trainEpochs=epochs,
                                                              storeFrequency=save_frequency,storeLocation=value_path,learningRate=lr)
    
    # Save outputs
    load_agent.save_agent_session(session_new,session_new.session_path.joinpath(f'learning\\'))
    session_new._agent_session._training_config.num_eval_episodes = 10
    session_new.evaluate()           