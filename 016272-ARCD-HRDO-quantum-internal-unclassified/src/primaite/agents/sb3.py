# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
from __future__ import annotations

import json
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlp

from primaite import getLogger
from primaite.agents.agent_abc import AgentSessionABC
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite

from boltzmannMachines.DBM import DBM
from boltzmannMachines.DBM_action import DBM_action
from boltzmannMachines.DBM_Hypernet import DBM_Hypernet
from torch import nn
import torch as th

_LOGGER: Logger = getLogger(__name__)


class SB3Agent(AgentSessionABC):
    """An AgentSession class that implements a Stable Baselines3 agent."""

    def __init__(
        self,
        training_config_path: Optional[Union[str, Path]] = None,
        lay_down_config_path: Optional[Union[str, Path]] = None,
        session_path: Optional[Union[str, Path]] = None,
        legacy_training_config: bool = False,
        legacy_lay_down_config: bool = False,
        DBM_parameters_dict: dict = None,
        PPO_parameters_dict: dict = None,
        previous_agent_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialise the SB3 Agent training session.

        :param training_config_path: YAML file containing configurable items defined in
            `primaite.config.training_config.TrainingConfig`
        :type training_config_path: Union[path, str]
        :param lay_down_config_path: YAML file containing configurable items for generating network laydown.
        :type lay_down_config_path: Union[path, str]
        :param legacy_training_config: True if the training config file is a legacy file from PrimAITE < 2.0,
            otherwise False.
        :param legacy_lay_down_config: True if the lay_down config file is a legacy file from PrimAITE < 2.0,
            otherwise False.
        :raises ValueError: If the training config contains an unexpected value for agent_framework (should be "SB3")
        :raises ValueError: If the training config contains an unexpected value for agent_identifies (should be `PPO`
            or `A2C`)
        """
        self.DBM_parameters_dict = DBM_parameters_dict
        self.PPO_parameters_dict = PPO_parameters_dict
        super().__init__(
            training_config_path, lay_down_config_path, session_path, legacy_training_config, legacy_lay_down_config, previous_agent_path
        )
        if not self._training_config.agent_framework == AgentFramework.SB3:
            msg = f"Expected SB3 agent_framework, " f"got {self._training_config.agent_framework}"
            _LOGGER.error(msg)
            raise ValueError(msg)
        self._agent_class: Union[PPO, A2C]
        if self._training_config.agent_identifier in {AgentIdentifier.PPO, AgentIdentifier.DBM_VALUE_PPO, AgentIdentifier.DBM_POLICY_PPO, AgentIdentifier.DBM_VALUE_POLICY_PPO, AgentIdentifier.DBM_HYPERNET_PPO}:
            self._agent_class = PPO
        elif self._training_config.agent_identifier == AgentIdentifier.A2C:
            self._agent_class = A2C
        else:
            msg = "Expected PPO or A2C agent_identifier, " f"got {self._training_config.agent_identifier}"
            _LOGGER.error(msg)
            raise ValueError(msg)

        self._tensorboard_log_path = self.learning_path / "tensorboard_logs"
        self._tensorboard_log_path.mkdir(parents=True, exist_ok=True)

        _LOGGER.debug(
            f"Created {self.__class__.__name__} using: "
            f"agent_framework={self._training_config.agent_framework}, "
            f"agent_identifier="
            f"{self._training_config.agent_identifier}"
        )

        self.is_eval = False

        self._setup()

    def _setup(self) -> None:
        """Set up the SB3 Agent."""
        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
            legacy_training_config=self.legacy_training_config,
            legacy_lay_down_config=self.legacy_lay_down_config,
        )
        # check if there is a zip file that needs to be loaded.
        # JS11: convert previous_agent_path to path type if it exists
        if self.previous_agent_path is not None:
            if not isinstance(self.previous_agent_path, Path):
                self.previous_agent_path = Path(self.previous_agent_path)
            previous_agent_load_file = next(self.previous_agent_path.rglob("SB3*.zip"), None)
        else:
            previous_agent_load_file = None
        load_file = next(self.session_path.rglob("*.zip"), None)

        #we use load file to continue a previous session.
        #we use previous_agent_load_file to start a new session with a trained agent.
        if not load_file:
            # create a new env and agent

            self._agent = self._agent_class(
                PPOMlp,
                self._env,
                verbose=self.sb3_output_verbose_level,
                n_steps=self._training_config.num_train_steps,
                tensorboard_log=str(self._tensorboard_log_path),
                seed=self._training_config.seed,
                **self.PPO_parameters_dict
            )
            #If we want to change the value networks we do so here
            if self._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_PPO or \
                self._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_POLICY_PPO:
                # Get current input layer size
                input_layer_size = self._agent.policy.mlp_extractor.value_net[0].in_features
                # Clear exisiting extractor network
                self._agent.policy.mlp_extractor.value_net = nn.Identity()

                # Setting the PPO value network to a DBM with the parameters specified in the training conig file
                self._agent.policy.value_net = DBM(n_visible=input_layer_size,**self.DBM_parameters_dict)
                print(f'DBM PARAMETERS IN SB PPO agent Value Network: {self.DBM_parameters_dict}')
            
            #If we want to change the policy networks we do so here
            if self._training_config.agent_identifier == AgentIdentifier.DBM_POLICY_PPO or \
                self._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_POLICY_PPO:
                # Get current input layer size
                input_layer_size = self._agent.policy.mlp_extractor.policy_net[0].in_features
                # Clear exisiting extractor network
                self._agent.policy.mlp_extractor.policy_net = nn.Identity()

                # Setting the PPO value network to a DBM with the parameters specified in the training config file
                output_layer_size = self._agent.policy.action_net.out_features
                total_visible_units = input_layer_size+output_layer_size
                self._agent.policy.action_net = DBM_action(n_visible=total_visible_units,n_actions=output_layer_size,**self.DBM_parameters_dict)
                print(f'DBM PARAMETERS IN SB PPO agent Policy Network: {self.DBM_parameters_dict}')
            
            #If we want to use hypernets we do so here
            if self._training_config.agent_identifier == AgentIdentifier.DBM_HYPERNET_PPO:
                policy_extractor = self._agent.policy.mlp_extractor.policy_net
                value_extractor = self._agent.policy.mlp_extractor.value_net
                policy_net = self._agent.policy.action_net
                value_net = self._agent.policy.value_net

                # Clear exisiting extractor networks
                self._agent.policy.mlp_extractor.policy_net = nn.Identity()
                self._agent.policy.mlp_extractor.value_net = nn.Identity()

                # Setting the PPO value network to a DBM with the parameters specified in the training conig file
                self._agent.policy.action_net = DBM_Hypernet([policy_extractor,policy_net],**self.DBM_parameters_dict)
                self._agent.policy.value_net = DBM_Hypernet([value_extractor,value_net],**self.DBM_parameters_dict)
                print(f'DBM PARAMETERS IN SB PPO agent Policy+Value Hypernetworks: {self.DBM_parameters_dict}')

            if self._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_PPO or \
                self._training_config.agent_identifier == AgentIdentifier.DBM_POLICY_PPO or \
                self._training_config.agent_identifier == AgentIdentifier.DBM_VALUE_POLICY_PPO or \
                self._training_config.agent_identifier == AgentIdentifier.DBM_HYPERNET_PPO:
                # Update the optimiser if required
                self._agent.policy.optimizer = \
                    self._agent.policy.optimizer_class(
                        self._agent.policy.parameters(),
                        self._agent.policy.optimizer.defaults['lr'],
                        **self._agent.policy.optimizer_kwargs)
            
            if previous_agent_load_file is not None:
                #JS11: set the new agent's policy to be equal to the poilcy of the loaded agent
                #JS11: We will need to adjust if the agent's previous env and current env have different obs/action spaces,
                #JS11: but it works if they have different reward functions.
                previous_agent = self._agent_class.load(previous_agent_load_file, env=self._env)
                previous_agent_parameters = previous_agent.get_parameters()
                previous_agent_policy = {'policy': previous_agent_parameters['policy']}
                self._agent.set_parameters(previous_agent_policy, exact_match = False)

        else:
            # set env values from session metadata
            with open(self.session_path / "session_metadata.json", "r") as file:
                md_dict = json.load(file)

            # load environment values
            if self.is_eval:
                # evaluation always starts at 0
                self._env.episode_count = 0
                self._env.total_step_count = 0
            else:
                # carry on from previous learning sessions
                self._env.episode_count = md_dict["learning"]["total_episodes"]
                self._env.total_step_count = md_dict["learning"]["total_time_steps"]

            # load the file
            self._agent = self._agent_class.load(load_file, env=self._env)

            # set agent values
            self._agent.verbose = self.sb3_output_verbose_level
            self._agent.tensorboard_log = self.session_path / "learning/tensorboard_logs"

        super()._setup()

    def _save_checkpoint(self) -> None:
        checkpoint_n = self._training_config.checkpoint_every_n_episodes
        episode_count = self._env.episode_count
        save_checkpoint = False
        if checkpoint_n:
            save_checkpoint = episode_count % checkpoint_n == 0
        if episode_count and save_checkpoint:
            checkpoint_path = self.checkpoints_path / f"sb3ppo_{episode_count}.zip"
            self._agent.save(checkpoint_path)
            _LOGGER.debug(f"Saved agent checkpoint: {checkpoint_path}")

    def _get_latest_checkpoint(self) -> None:
        pass

    def learn(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Train the agent.

        :param kwargs: Any agent-specific key-word args to be passed.
        """
        time_steps = self._training_config.num_train_steps
        episodes = self._training_config.num_train_episodes
        self.is_eval = False
        _LOGGER.info(f"Beginning learning for {episodes} episodes @" f" {time_steps} time steps...")
        for i in range(episodes):
            self._agent.learn(total_timesteps=time_steps)
            self._save_checkpoint()
        self._env._write_av_reward_per_episode()  # noqa
        self.save()
        self._env.close()
        super().learn()

        # save agent
        self.save()

        self._plot_av_reward_per_episode(learning_session=True)

    def evaluate(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Evaluate the agent.

        :param kwargs: Any agent-specific key-word args to be passed.
        """
        time_steps = self._training_config.num_eval_steps
        episodes = self._training_config.num_eval_episodes
        self._env.set_as_eval()
        self.is_eval = True
        if self._training_config.deterministic:
            deterministic_str = "deterministic"
        else:
            deterministic_str = "non-deterministic"
        _LOGGER.info(
            f"Beginning {deterministic_str} evaluation for " f"{episodes} episodes @ {time_steps} time steps..."
        )
        for episode in range(episodes):
            obs = self._env.reset()

            for step in range(time_steps):
                action, _states = self._agent.predict(obs, deterministic=self._training_config.deterministic)
                if isinstance(action, np.ndarray):
                    action = np.int64(action)
                obs, rewards, done, info = self._env.step(action)
        self._env._write_av_reward_per_episode()  # noqa
        self._env.close()
        super().evaluate()

    def save(self) -> None:
        """Save the agent."""
        self._agent.save(self._saved_agent_path)

    def export(self) -> None:
        """Export the agent to transportable file format."""
        raise NotImplementedError
