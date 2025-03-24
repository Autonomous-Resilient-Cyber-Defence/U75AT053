from primaite.primaite_session import PrimaiteSession
from stable_baselines3 import PPO
from boltzmannMachines.DBM import DBM
from boltzmannMachines.DBM_action import DBM_action
from boltzmannMachines.DBM_Hypernet import DBM_Hypernet
from primaite.common.enums import AgentIdentifier
from pathlib import Path

def load_agent_session(config,laydown,model_location):
    
    session = PrimaiteSession(
        config, laydown, None, False, False
    )
    session.setup()

    # Load in the agent and data
    if session._agent_session._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_POLICY_PPO or \
        session._agent_session._training_config.agent_identifier == AgentIdentifier.DBM_HYPERNET_PPO:
        session._agent_session._agent.policy.action_net.loadWeights(model_location.joinpath('Policy'))
        session._agent_session._agent.policy.value_net.loadWeights(model_location.joinpath('Value'))
    else:
        session._agent_session._agent = PPO.load(model_location.joinpath('SB3_PPO.zip'))
    return session


def save_agent_session(session,resultsPath):
        if not Path.is_dir(resultsPath):
            resultsPath.mkdir()

        policy_folder = Path.joinpath(resultsPath,'Policy')
        policy_folder.mkdir()    
        value_folder = Path.joinpath(resultsPath,'Value')
        value_folder.mkdir()

        if type(session._agent_session._agent.policy.action_net) is DBM_action or \
            type(session._agent_session._agent.policy.action_net) is DBM_Hypernet:
            session._agent_session._agent.policy.action_net.saveWeights(str(policy_folder))

        if type(session._agent_session._agent.policy.value_net) is DBM or \
            type(session._agent_session._agent.policy.action_net) is DBM_Hypernet:
            session._agent_session._agent.policy.value_net.saveWeights(str(value_folder))

        session._agent_session._agent.policy.save(str(resultsPath)+'\\PPO.zip')
