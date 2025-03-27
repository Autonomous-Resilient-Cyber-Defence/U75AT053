from primaite.primaite_session import PrimaiteSession
from stable_baselines3 import PPO
from boltzmannMachines.DBM import DBM
from boltzmannMachines.DBM_action import DBM_action
from boltzmannMachines.DBM_Hypernet import DBM_Hypernet
from primaite.common.enums import AgentIdentifier
from pathlib import Path

def load_agent_session(config,laydown,model_location):
    if not model_location is Path:
        model_location = Path(model_location)

    session = PrimaiteSession(
        config, laydown, None, False, False
    )

    # Load in the agent and data
    if session._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_POLICY_PPO or \
        session._training_config.agent_identifier == AgentIdentifier.DBM_HYPERNET_PPO:

        session.setup()

        session._agent_session._agent.policy.action_net.loadWeights(model_location.joinpath('Policy'))
        session._agent_session._agent.policy.value_net.loadWeights(model_location.joinpath('Value'))
        session._agent_session._agent.policy.optimizer = \
            session._agent_session._agent.policy.optimizer_class(
                session._agent_session._agent.policy.parameters(),
                session._agent_session._agent.policy.optimizer.defaults['lr'],
                **session._agent_session._agent.policy.optimizer_kwargs)
    else:
        session = PrimaiteSession(
            config, laydown, None, False, previous_agent_path=model_location
        )
        session.setup()
        session._agent_session._agent.policy.optimizer = \
            session._agent_session._agent.policy.optimizer_class(
                session._agent_session._agent.policy.parameters(),
                session._agent_session._agent.policy.optimizer.defaults['lr'],
                **session._agent_session._agent.policy.optimizer_kwargs)
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
            try:
                session._agent_session._agent.policy.action_net.saveWeights(str(policy_folder))
            except:
                pass

        if type(session._agent_session._agent.policy.value_net) is DBM or \
            type(session._agent_session._agent.policy.action_net) is DBM_Hypernet:
            try:
                session._agent_session._agent.policy.value_net.saveWeights(str(value_folder))
            except:
                pass
        session._agent_session._agent.policy.save(str(resultsPath)+'\\PPO.zip')
