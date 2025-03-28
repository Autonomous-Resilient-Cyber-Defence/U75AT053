# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
"""Main environment module containing the PRIMmary AI Training Evironment (Primaite) class."""
import copy
import logging
import uuid as uuid
from logging import Logger
from pathlib import Path
from random import choice, randint, sample, uniform
from typing import Any, Dict, Final, List, Tuple, Union

import networkx as nx
import numpy as np
from gym import Env, spaces
from matplotlib import pyplot as plt

from primaite import getLogger
from primaite.acl.access_control_list import AccessControlList
from primaite.agents.utils import is_valid_acl_action_extra, is_valid_node_action
from primaite.common.custom_typing import NodeUnion
from primaite.common.enums import (
    ActionType,
    AgentFramework,
    AgentIdentifier,
    FileSystemState,
    HardwareState,
    NodePOLInitiator,
    NodePOLType,
    NodeType,
    ObservationType,
    Priority,
    SessionType,
    SoftwareState,
)
from primaite.common.service import Service
from primaite.config import training_config
from primaite.config.lay_down_config import load
from primaite.config.training_config import TrainingConfig
from primaite.environment.observations import ObservationsHandler
from primaite.environment.reward import calculate_reward_function
from primaite.links.link import Link
from primaite.nodes.active_node import ActiveNode
from primaite.nodes.node import Node
from primaite.nodes.node_state_instruction_green import NodeStateInstructionGreen
from primaite.nodes.node_state_instruction_red import NodeStateInstructionRed
from primaite.nodes.passive_node import PassiveNode
from primaite.nodes.service_node import ServiceNode
from primaite.pol.green_pol import apply_iers, apply_node_pol
from primaite.pol.ier import IER
from primaite.pol.red_agent_pol import apply_red_agent_iers, apply_red_agent_node_pol
from primaite.transactions.transaction import Transaction
from primaite.utils.session_output_writer import SessionOutputWriter

_LOGGER: Logger = getLogger(__name__)


class Primaite(Env):
    """PRIMmary AI Training Evironment (Primaite) class."""

    #  Action Space contants
    ACTION_SPACE_NODE_PROPERTY_VALUES: int = 5
    ACTION_SPACE_NODE_ACTION_VALUES: int = 4
    ACTION_SPACE_ACL_ACTION_VALUES: int = 3
    ACTION_SPACE_ACL_PERMISSION_VALUES: int = 2

    def __init__(
        self,
        training_config_path: Union[str, Path],
        lay_down_config_path: Union[str, Path],
        session_path: Path,
        timestamp_str: str,
        legacy_training_config: bool = False,
        legacy_lay_down_config: bool = False,
    ) -> None:
        """
        The Primaite constructor.

        :param training_config_path: The training config filepath.
        :param lay_down_config_path: The lay down config filepath.
        :param session_path: The directory path the session is writing to.
        :param timestamp_str: The session timestamp in the format: <yyyy-mm-dd>_<hh-mm- ss>.
        :param legacy_training_config: True if the training config file is a legacy file from PrimAITE < 2.0,
            otherwise False.
        :param legacy_lay_down_config: True if the lay_down config file is a legacy file from PrimAITE < 2.0,
            otherwise False.
        """
        self.session_path: Final[Path] = session_path
        self.timestamp_str: Final[str] = timestamp_str
        self._training_config_path: Union[str, Path] = training_config_path
        self._lay_down_config_path: Union[str, Path] = lay_down_config_path
        self.legacy_training_config = legacy_training_config
        self.legacy_lay_down_config = legacy_lay_down_config

        self.training_config: TrainingConfig = training_config.load(training_config_path, self.legacy_training_config)
        _LOGGER.info(f"Using: {str(self.training_config)}")

        # Number of steps in an episode
        self.episode_steps: int
        if self.training_config.session_type == SessionType.TRAIN:
            self.episode_steps = self.training_config.num_train_steps
        elif self.training_config.session_type == SessionType.EVAL:
            self.episode_steps = self.training_config.num_eval_steps
        else:
            self.episode_steps = self.training_config.num_train_steps

        super(Primaite, self).__init__()

        # The agent in use
        self.agent_identifier: AgentIdentifier = self.training_config.agent_identifier

        # Create a dictionary to hold all the nodes
        self.nodes: Dict[str, NodeUnion] = {}

        # Create a dictionary to hold a reference set of nodes
        self.nodes_reference: Dict[str, NodeUnion] = {}

        # Create a dictionary to hold all the links
        self.links: Dict[str, Link] = {}

        # Create a dictionary to hold a reference set of links
        self.links_reference: Dict[str, Link] = {}

        # Create a dictionary to hold all the green IERs (this will come from an external source)
        self.green_iers: Dict[str, IER] = {}
        self.green_iers_reference: Dict[str, IER] = {}

        # Create a dictionary to hold all the node PoLs (this will come from an external source)
        self.node_pol: Dict[str, NodeStateInstructionGreen] = {}

        # Create a dictionary to hold all the red agent IERs (this will come from an external source)
        self.red_iers: Dict[str, IER] = {}

        # Create a dictionary to hold all the red agent node PoLs (this will come from an external source)
        self.red_node_pol: Dict[str, NodeStateInstructionRed] = {}

        # Create the Access Control List
        self.acl: AccessControlList = AccessControlList(
            self.training_config.implicit_acl_rule,
            self.training_config.max_number_acl_rules,
        )
        # Sets limit for number of ACL rules in environment
        self.max_number_acl_rules: int = self.training_config.max_number_acl_rules

        # Create a list of services (enums)
        self.services_list: List[str] = []

        # Create a list of ports
        self.ports_list: List[str] = []

        # Create graph (network)
        self.network: nx.Graph = nx.MultiGraph()

        # Create a graph (network) reference
        self.network_reference: nx.Graph = nx.MultiGraph()

        # Create step count
        self.step_count: int = 0

        self.total_step_count: int = 0
        """The total number of time steps completed."""

        # Create step info dictionary
        self.step_info: Dict[Any] = {}

        # Total reward
        self.total_reward: float = 0

        # Average reward
        self.average_reward: float = 0

        # Episode count
        self.episode_count: int = 0

        # Number of nodes - gets a value by examining the nodes dictionary after it's been populated
        self.num_nodes: int = 0

        # Number of links - gets a value by examining the links dictionary after it's been populated
        self.num_links: int = 0

        # Number of services - gets a value when config is loaded
        self.num_services: int = 0

        # Number of ports - gets a value when config is loaded
        self.num_ports: int = 0

        # The action type
        # TODO: confirm type
        self.action_type: int = 0

        # TODO fix up with TrainingConfig
        # stores the observation config from the yaml, default is NODE_LINK_TABLE
        self.obs_config: dict = {"components": [{"name": "NODE_LINK_TABLE"}]}
        if self.training_config.observation_space is not None:
            self.obs_config = self.training_config.observation_space

        # Observation Handler manages the user-configurable observation space.
        # It will be initialised later.
        self.obs_handler: ObservationsHandler

        self._obs_space_description: List[str] = None
        "The env observation space description for transactions writing"

        self.lay_down_config = load(self._lay_down_config_path, self.legacy_lay_down_config)
        self.load_lay_down_config()

        # Store the node objects as node attributes
        # (This is so we can access them as objects)
        for node in self.network:
            self.network.nodes[node]["self"] = node

        for node in self.network_reference:
            self.network_reference.nodes[node]["self"] = node

        self.num_nodes = len(self.nodes)
        self.num_links = len(self.links)

        # Visualise in PNG
        try:
            plt.tight_layout()
            plt.figure(1, figsize=(75, 75))
            nx.draw_networkx(self.network, with_labels=True, node_size=80,font_size=11)

            file_path = session_path / f"network_{timestamp_str}.png"
            plt.savefig(file_path, format="PNG")
            plt.clf()
        except Exception:
            _LOGGER.error("Could not save network diagram", exc_info=True)

        # Initiate observation space
        self.observation_space: spaces.Space
        self.env_obs: np.ndarray
        self.observation_space, self.env_obs = self.init_observations()

        # Define Action Space - depends on action space type (Node or ACL)
        self.action_dict: Dict[int, List[int]]
        self.action_space: spaces.Space
        if self.training_config.action_type == ActionType.NODE:
            _LOGGER.debug("Action space type NODE selected")
            # Terms (for node action space):
            # [0, num nodes] - node ID (0 = nothing, node ID)
            # [0, 4] - what property it's acting on (0 = nothing, state, SoftwareState, service state, file system state) # noqa
            # [0, 3] - action on property (0 = nothing, On / Scan, Off / Repair, Reset / Patch / Restore) # noqa
            # [0, num services] - resolves to service ID (0 = nothing, resolves to service) # noqa
            self.action_dict = self.create_node_action_dict()
            self.action_space = spaces.Discrete(len(self.action_dict))
        elif self.training_config.action_type == ActionType.ACL:
            _LOGGER.debug("Action space type ACL selected")
            # Terms (for ACL action space):
            # [0, 2] - Action (0 = do nothing, 1 = create rule, 2 = delete rule)
            # [0, 1] - Permission (0 = DENY, 1 = ALLOW)
            # [0, num nodes] - Source IP (0 = any, then 1 -> x resolving to IP addresses)
            # [0, num nodes] - Dest IP (0 = any, then 1 -> x resolving to IP addresses)
            # [0, num services] - Protocol (0 = any, then 1 -> x resolving to protocol)
            # [0, num ports] - Port (0 = any, then 1 -> x resolving to port)
            self.action_dict = self.create_acl_action_dict()
            self.action_space = spaces.Discrete(len(self.action_dict))
        elif self.training_config.action_type == ActionType.ANY:
            _LOGGER.debug("Action space type ANY selected - Node + ACL")
            self.action_dict = self.create_node_and_acl_action_dict()
            self.action_space = spaces.Discrete(len(self.action_dict))
        else:
            _LOGGER.error(f"Invalid action type selected: {self.training_config.action_type}")

        self.episode_av_reward_writer: SessionOutputWriter = SessionOutputWriter(
            self, transaction_writer=False, learning_session=True
        )
        self.transaction_writer: SessionOutputWriter = SessionOutputWriter(
            self, transaction_writer=True, learning_session=True
        )

        self.is_eval = False

    @property
    def actual_episode_count(self) -> int:
        """Shifts the episode_count by -1 for RLlib learning session."""
        if self.training_config.agent_framework is AgentFramework.RLLIB and not self.is_eval:
            return self.episode_count - 1
        return self.episode_count

    def set_as_eval(self) -> None:
        """Set the writers to write to eval directories."""
        self.episode_av_reward_writer = SessionOutputWriter(self, transaction_writer=False, learning_session=False)
        self.transaction_writer = SessionOutputWriter(self, transaction_writer=True, learning_session=False)
        self.episode_count = 0
        self.step_count = 0
        self.total_step_count = 0
        self.episode_steps = self.training_config.num_eval_steps
        self.is_eval = True

    def _write_av_reward_per_episode(self) -> None:
        if self.actual_episode_count > 0:
            csv_data = self.actual_episode_count, self.average_reward
            self.episode_av_reward_writer.write(csv_data)

    def reset(self) -> np.ndarray:
        """
        AI Gym Reset function.

        Returns:
             Environment observation space (reset)
        """
        self._write_av_reward_per_episode()
        self.episode_count += 1

        # Don't need to reset links, as they are cleared and recalculated every
        # step

        # Clear the ACL
        self.init_acl()

        # Reset the node statuses and recreate the ACL from config
        # Does this for both live and reference nodes
        self.reset_environment()

        # Create a random red agent to use for this episode
        if self.training_config.random_red_agent:
            self._create_random_red_agent()

        # Reset counters and totals
        self.total_reward = 0.0
        self.step_count = 0
        self.average_reward = 0.0

        # Update observations space and return
        self.update_environent_obs()

        return self.env_obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        AI Gym Step function.

        Args:
            action: Action space from agent

        Returns:
             env_obs: Observation space
             reward: Reward value for this step
             done: Indicates episode is complete if True
             step_info: Additional information relating to this step
        """
        # TEMP
        done = False
        self.step_count += 1
        self.total_step_count += 1

        # Need to clear traffic on all links first
        for link_key, link_value in self.links.items():
            link_value.clear_traffic()

        for link in self.links_reference.values():
            link.clear_traffic()

        # Create a Transaction (metric) object for this step
        transaction = Transaction(self.agent_identifier, self.actual_episode_count, self.step_count)
        # Load the initial observation space into the transaction
        transaction.obs_space = self.obs_handler._flat_observation

        # Set the transaction obs space description
        transaction.obs_space_description = self._obs_space_description

        # Load the action space into the transaction
        transaction.action_space = copy.deepcopy(action)

        # 1. Implement Blue Action
        self.interpret_action_and_apply(action)
        # Take snapshots of nodes and links
        self.nodes_post_blue = copy.deepcopy(self.nodes)
        self.links_post_blue = copy.deepcopy(self.links)

        # 2. Perform any time-based activities (e.g. a component moving from patching to good)
        self.apply_time_based_updates()

        # 3. Apply PoL
        apply_node_pol(self.nodes, self.node_pol, self.step_count)  # Node PoL
        apply_iers(
            self.network,
            self.nodes,
            self.links,
            self.green_iers,
            self.acl,
            self.step_count,
        )  # Network PoL
        # Take snapshots of nodes and links
        self.nodes_post_pol = copy.deepcopy(self.nodes)
        self.links_post_pol = copy.deepcopy(self.links)
        # Reference
        apply_node_pol(self.nodes_reference, self.node_pol, self.step_count)  # Node PoL
        apply_iers(
            self.network_reference,
            self.nodes_reference,
            self.links_reference,
            self.green_iers_reference,
            self.acl,
            self.step_count,
        )  # Network PoL

        # 4. Implement Red Action
        apply_red_agent_iers(
            self.network,
            self.nodes,
            self.links,
            self.red_iers,
            self.acl,
            self.step_count,
        )
        apply_red_agent_node_pol(self.nodes, self.red_iers, self.red_node_pol, self.step_count)
        # Take snapshots of nodes and links
        self.nodes_post_red = copy.deepcopy(self.nodes)
        self.links_post_red = copy.deepcopy(self.links)

        # 5. Calculate reward signal (for RL)
        reward = calculate_reward_function(
            self.nodes_post_pol,
            self.nodes_post_red,
            self.nodes_reference,
            self.green_iers,
            self.green_iers_reference,
            self.red_iers,
            self.step_count,
            self.training_config,
        )
        _LOGGER.debug(f"Episode: {self.actual_episode_count}, " f"Step {self.step_count}, " f"Reward: {reward}")
        self.total_reward += reward
        if self.step_count == self.episode_steps:
            self.average_reward = self.total_reward / self.step_count
            if self.training_config.session_type is SessionType.EVAL:
                # For evaluation, need to trigger the done value = True when
                # step count is reached in order to prevent neverending episode
                done = True
            _LOGGER.info(f"Episode: {self.actual_episode_count}, " f"Average Reward: {self.average_reward}")
            # Load the reward into the transaction
        transaction.reward = reward

        # 6. Output Verbose
        # self.output_link_status()

        # 7. Update env_obs
        self.update_environent_obs()

        # Write transaction to file
        if self.actual_episode_count > 0:
            self.transaction_writer.write(transaction)

        # Return
        return self.env_obs, reward, done, self.step_info

    def close(self) -> None:
        """Override parent close and close writers."""
        # Close files if last episode/step
        # if self.can_finish:
        super().close()

        self.transaction_writer.close()
        self.episode_av_reward_writer.close()

    def init_acl(self) -> None:
        """Initialise the Access Control List."""
        self.acl.remove_all_rules()

    def output_link_status(self) -> None:
        """Output the link status of all links to the console."""
        for link_key, link_value in self.links.items():
            _LOGGER.debug("Link ID: " + link_value.get_id())
            for protocol in link_value.protocol_list:
                print("    Protocol: " + protocol.get_name().name + ", Load: " + str(protocol.get_load()))

    def interpret_action_and_apply(self, _action: int) -> None:
        """
        Applies agent actions to the nodes and Access Control List.

        Args:
            _action: The action space from the agent
        """
        # At the moment, actions are only affecting nodes
        if self.training_config.action_type == ActionType.NODE:
            self.apply_actions_to_nodes(_action)
        elif self.training_config.action_type == ActionType.ACL:
            self.apply_actions_to_acl(_action)
        elif len(self.action_dict[_action]) == 7:  # ACL actions in multidiscrete form have len 7
            self.apply_actions_to_acl(_action)
        elif len(self.action_dict[_action]) == 4:  # Node actions in multdiscrete (array) from have len 4
            self.apply_actions_to_nodes(_action)
        else:
            logging.error("Invalid action type found")

    def apply_actions_to_nodes(self, _action: int) -> None:
        """
        Applies agent actions to the nodes.

        Args:
            _action: The action space from the agent
        """
        readable_action = self.action_dict[_action]
        node_id = readable_action[0]
        node_property = readable_action[1]
        property_action = readable_action[2]
        service_index = readable_action[3]

        # Check that the action is requesting a valid node
        try:
            node = self.nodes[str(node_id)]
        except Exception:
            return

        if node_property == 0:
            # This is the do nothing action
            return
        elif node_property == 1:
            # This is an action on the node Hardware State
            if property_action == 0:
                # Do nothing
                return
            elif property_action == 1:
                # Turn on (only applicable if it's OFF, not if it's patching)
                if node.hardware_state == HardwareState.OFF:
                    node.turn_on()
            elif property_action == 2:
                # Turn off
                node.turn_off()
            elif property_action == 3:
                # Reset (only applicable if it's ON)
                if node.hardware_state == HardwareState.ON:
                    node.reset()
            else:
                return
        elif node_property == 2:
            if isinstance(node, ActiveNode) or isinstance(node, ServiceNode):
                # This is an action on the node Software State
                if property_action == 0:
                    # Do nothing
                    return
                elif property_action == 1:
                    # Patch (valid action if it's good or compromised)
                    node.software_state = SoftwareState.PATCHING
            else:
                # Node is not of Active or Service Type
                return
        elif node_property == 3:
            # This is an action on a node Service State
            if isinstance(node, ServiceNode):
                # This is an action on a node Service State
                if property_action == 0:
                    # Do nothing
                    return
                elif property_action == 1:
                    # Patch (valid action if it's good or compromised)
                    node.set_service_state(self.services_list[service_index], SoftwareState.PATCHING)
            else:
                # Node is not of Service Type
                return
        elif node_property == 4:
            # This is an action on a node file system state
            if isinstance(node, ActiveNode):
                if property_action == 0:
                    # Do nothing
                    return
                elif property_action == 1:
                    # Scan
                    node.start_file_system_scan()
                elif property_action == 2:
                    # Repair
                    # You cannot repair a destroyed file system - it needs restoring
                    if node.file_system_state_actual != FileSystemState.DESTROYED:
                        node.set_file_system_state(FileSystemState.REPAIRING)
                elif property_action == 3:
                    # Restore
                    node.set_file_system_state(FileSystemState.RESTORING)
            else:
                # Node is not of Active Type
                return
        else:
            return

    def apply_actions_to_acl(self, _action: int) -> None:
        """
        Applies agent actions to the Access Control List [TO DO].

        Args:
            _action: The action space from the agent
        """
        # Convert discrete value back to multidiscrete
        readable_action = self.action_dict[_action]

        action_decision = readable_action[0]
        action_permission = readable_action[1]
        action_source_ip = readable_action[2]
        action_destination_ip = readable_action[3]
        action_protocol = readable_action[4]
        action_port = readable_action[5]
        acl_rule_position = readable_action[6]

        if action_decision == 0:
            # It's decided to do nothing
            return
        else:
            # It's decided to create a new ACL rule or remove an existing rule
            # Permission value
            if action_permission == 0:
                acl_rule_permission = "DENY"
            else:
                acl_rule_permission = "ALLOW"
            # Source IP value
            if action_source_ip == 0:
                acl_rule_source = "ANY"
            else:
                node = list(self.nodes.values())[action_source_ip - 1]
                if isinstance(node, ServiceNode) or isinstance(node, ActiveNode):
                    acl_rule_source = node.ip_address
                else:
                    return
            # Destination IP value
            if action_destination_ip == 0:
                acl_rule_destination = "ANY"
            else:
                node = list(self.nodes.values())[action_destination_ip - 1]
                if isinstance(node, ServiceNode) or isinstance(node, ActiveNode):
                    acl_rule_destination = node.ip_address
                else:
                    return
            # Protocol value
            if action_protocol == 0:
                acl_rule_protocol = "ANY"
            else:
                acl_rule_protocol = self.services_list[action_protocol - 1]
            # Port value
            if action_port == 0:
                acl_rule_port = "ANY"
            else:
                acl_rule_port = self.ports_list[action_port - 1]

            # Now add or remove
            if action_decision == 1:
                # Add the rule
                self.acl.add_rule(
                    acl_rule_permission,
                    acl_rule_source,
                    acl_rule_destination,
                    acl_rule_protocol,
                    acl_rule_port,
                    acl_rule_position,
                )
            elif action_decision == 2:
                # Remove the rule
                self.acl.remove_rule(
                    acl_rule_permission,
                    acl_rule_source,
                    acl_rule_destination,
                    acl_rule_protocol,
                    acl_rule_port,
                )
            else:
                return

    def apply_time_based_updates(self) -> None:
        """
        Updates anything that needs to count down and then change state.

        e.g. reset / patching status
        """
        for node_key, node in self.nodes.items():
            if node.hardware_state == HardwareState.RESETTING:
                node.update_resetting_status()
            else:
                pass
            if isinstance(node, ActiveNode) or isinstance(node, ServiceNode):
                node.update_file_system_state()
                if node.software_state == SoftwareState.PATCHING:
                    node.update_os_patching_status()
                else:
                    pass
            else:
                pass
            if isinstance(node, ServiceNode):
                node.update_services_patching_status()
            else:
                pass

        for node_key, node in self.nodes_reference.items():
            if node.hardware_state == HardwareState.RESETTING:
                node.update_resetting_status()
            else:
                pass
            if isinstance(node, ActiveNode) or isinstance(node, ServiceNode):
                node.update_file_system_state()
                if node.software_state == SoftwareState.PATCHING:
                    node.update_os_patching_status()
                else:
                    pass
            else:
                pass
            if isinstance(node, ServiceNode):
                node.update_services_patching_status()
            else:
                pass

    def init_observations(self) -> Tuple[spaces.Space, np.ndarray]:
        """
        Create the environment's observation handler.

        :return: The observation space, initial observation (zeroed out array with the correct shape)
        :rtype: Tuple[spaces.Space, np.ndarray]
        """
        self.obs_handler = ObservationsHandler.from_config(self, self.obs_config)

        if not self._obs_space_description:
            self._obs_space_description = self.obs_handler.describe_structure()

        return self.obs_handler.space, self.obs_handler.current_observation

    def update_environent_obs(self) -> None:
        """Updates the observation space based on the node and link status."""
        self.obs_handler.update_obs()
        self.env_obs = self.obs_handler.current_observation

    def load_lay_down_config(self) -> None:
        """Loads config data in order to build the environment configuration."""
        for item in self.lay_down_config:
            if item["item_type"] == "NODE":
                # Create a node
                self.create_node(item)
            elif item["item_type"] == "LINK":
                # Create a link
                self.create_link(item)
            elif item["item_type"] == "GREEN_IER":
                # Create a Green IER
                self.create_green_ier(item)
            elif item["item_type"] == "GREEN_POL":
                # Create a Green PoL
                self.create_green_pol(item)
            elif item["item_type"] == "RED_IER":
                # Create a Red IER
                self.create_red_ier(item)
            elif item["item_type"] == "RED_POL":
                # Create a Red PoL
                self.create_red_pol(item)
            elif item["item_type"] == "ACL_RULE":
                # Create an ACL rule
                self.create_acl_rule(item)
            elif item["item_type"] == "SERVICES":
                # Create the list of services
                self.create_services_list(item)
            elif item["item_type"] == "PORTS":
                # Create the list of ports
                self.create_ports_list(item)
            else:
                item_type = item["item_type"]
                _LOGGER.error(f"Invalid item_type: {item_type}")
                pass

        _LOGGER.info("Environment configuration loaded")
        print("Environment configuration loaded")

    def create_node(self, item: Dict) -> None:
        """
        Creates a node from config data.

        Args:
            item: A config data item
        """
        # All nodes have these parameters
        node_id = item["node_id"]
        node_name = item["name"]
        node_class = item["node_class"]
        node_type = NodeType[item["node_type"]]
        node_priority = Priority[item["priority"]]
        node_hardware_state = HardwareState[item["hardware_state"]]

        if node_class == "PASSIVE":
            node = PassiveNode(
                node_id,
                node_name,
                node_type,
                node_priority,
                node_hardware_state,
                self.training_config,
            )
        elif node_class == "ACTIVE":
            # Active nodes have IP address, Software State and file system state
            node_ip_address = item["ip_address"]
            node_software_state = SoftwareState[item["software_state"]]
            node_file_system_state = FileSystemState[item["file_system_state"]]
            node = ActiveNode(
                node_id,
                node_name,
                node_type,
                node_priority,
                node_hardware_state,
                node_ip_address,
                node_software_state,
                node_file_system_state,
                self.training_config,
            )
        elif node_class == "SERVICE":
            # Service nodes have IP address, Software State, file system state and list of services
            node_ip_address = item["ip_address"]
            node_software_state = SoftwareState[item["software_state"]]
            node_file_system_state = FileSystemState[item["file_system_state"]]
            node = ServiceNode(
                node_id,
                node_name,
                node_type,
                node_priority,
                node_hardware_state,
                node_ip_address,
                node_software_state,
                node_file_system_state,
                self.training_config,
            )
            node_services = item["services"]
            for service in node_services:
                service_protocol = service["name"]
                service_port = service["port"]
                service_state = SoftwareState[service["state"]]
                node.add_service(Service(service_protocol, service_port, service_state))
        else:
            # Bad formatting
            pass

        # Copy the node for the reference version
        node_ref = copy.deepcopy(node)

        # Add node to node dictionary
        self.nodes[node_id] = node

        # Add reference node to reference node dictionary
        self.nodes_reference[node_id] = node_ref

        # Add node to network
        self.network.add_nodes_from([node])

        # Add node to network (reference)
        self.network_reference.add_nodes_from([node_ref])

    def create_link(self, item: Dict) -> None:
        """
        Creates a link from config data.

        Args:
            item: A config data item
        """
        link_id = item["id"]
        link_name = item["name"]
        link_bandwidth = item["bandwidth"]
        link_source = item["source"]
        link_destination = item["destination"]

        source_node: Node = self.nodes[link_source]
        dest_node: Node = self.nodes[link_destination]

        # Add link to network
        self.network.add_edge(source_node, dest_node, id=link_name)

        # Add link to link dictionary
        self.links[link_name] = Link(
            link_id,
            link_bandwidth,
            source_node.name,
            dest_node.name,
            self.services_list,
        )

        # Reference
        source_node_ref: Node = self.nodes_reference[link_source]
        dest_node_ref: Node = self.nodes_reference[link_destination]

        # Add link to network (reference)
        self.network_reference.add_edge(source_node_ref, dest_node_ref, id=link_name)

        # Add link to link dictionary (reference)
        self.links_reference[link_name] = Link(
            link_id,
            link_bandwidth,
            source_node_ref.name,
            dest_node_ref.name,
            self.services_list,
        )

    def create_green_ier(self, item: Dict) -> None:
        """
        Creates a green IER from config data.

        Args:
            item: A config data item
        """
        ier_id = item["id"]
        ier_start_step = item["start_step"]
        ier_end_step = item["end_step"]
        ier_load = item["load"]
        ier_protocol = item["protocol"]
        ier_port = item["port"]
        ier_source = item["source"]
        ier_destination = item["destination"]
        ier_mission_criticality = item["mission_criticality"]

        # Create IER and add to green IER dictionary
        self.green_iers[ier_id] = IER(
            ier_id,
            ier_start_step,
            ier_end_step,
            ier_load,
            ier_protocol,
            ier_port,
            ier_source,
            ier_destination,
            ier_mission_criticality,
        )
        self.green_iers_reference[ier_id] = IER(
            ier_id,
            ier_start_step,
            ier_end_step,
            ier_load,
            ier_protocol,
            ier_port,
            ier_source,
            ier_destination,
            ier_mission_criticality,
        )

    def create_red_ier(self, item: Dict) -> None:
        """
        Creates a red IER from config data.

        Args:
            item: A config data item
        """
        ier_id = item["id"]
        ier_start_step = item["start_step"]
        ier_end_step = item["end_step"]
        ier_load = item["load"]
        ier_protocol = item["protocol"]
        ier_port = item["port"]
        ier_source = item["source"]
        ier_destination = item["destination"]
        ier_mission_criticality = item["mission_criticality"]

        # Create IER and add to red IER dictionary
        self.red_iers[ier_id] = IER(
            ier_id,
            ier_start_step,
            ier_end_step,
            ier_load,
            ier_protocol,
            ier_port,
            ier_source,
            ier_destination,
            ier_mission_criticality,
        )

    def create_green_pol(self, item: Dict) -> None:
        """
        Creates a green PoL object from config data.

        Args:
            item: A config data item
        """
        pol_id = item["id"]
        pol_start_step = item["start_step"]
        pol_end_step = item["end_step"]
        pol_node = item["nodeId"]
        pol_type = NodePOLType[item["type"]]

        # State depends on whether this is Operating, Software, file system or Service PoL type
        if pol_type == NodePOLType.OPERATING:
            pol_state = HardwareState[item["state"]]
            pol_protocol = ""
        elif pol_type == NodePOLType.FILE:
            pol_state = FileSystemState[item["state"]]
            pol_protocol = ""
        else:
            pol_protocol = item["protocol"]
            pol_state = SoftwareState[item["state"]]

        self.node_pol[pol_id] = NodeStateInstructionGreen(
            pol_id,
            pol_start_step,
            pol_end_step,
            pol_node,
            pol_type,
            pol_protocol,
            pol_state,
        )

    def create_red_pol(self, item: Dict) -> None:
        """
        Creates a red PoL object from config data.

        Args:
            item: A config data item
        """
        pol_id = item["id"]
        pol_start_step = item["start_step"]
        pol_end_step = item["end_step"]
        pol_target_node_id = item["targetNodeId"]
        pol_initiator = NodePOLInitiator[item["initiator"]]
        pol_type = NodePOLType[item["type"]]
        pol_protocol = item["protocol"]

        # State depends on whether this is Operating, Software, file system or Service PoL type
        if pol_type == NodePOLType.OPERATING:
            pol_state = HardwareState[item["state"]]
        elif pol_type == NodePOLType.FILE:
            pol_state = FileSystemState[item["state"]]
        else:
            pol_state = SoftwareState[item["state"]]

        pol_source_node_id = item["sourceNodeId"]
        pol_source_node_service = item["sourceNodeService"]
        pol_source_node_service_state = item["sourceNodeServiceState"]

        self.red_node_pol[pol_id] = NodeStateInstructionRed(
            pol_id,
            pol_start_step,
            pol_end_step,
            pol_target_node_id,
            pol_initiator,
            pol_type,
            pol_protocol,
            pol_state,
            pol_source_node_id,
            pol_source_node_service,
            pol_source_node_service_state,
        )

    def create_acl_rule(self, item: Dict) -> None:
        """
        Creates an ACL rule from config data.

        Args:
            item: A config data item
        """
        acl_rule_permission = item["permission"]
        acl_rule_source = item["source"]
        acl_rule_destination = item["destination"]
        acl_rule_protocol = item["protocol"]
        acl_rule_port = item["port"]
        acl_rule_position = item.get("position")

        self.acl.add_rule(
            acl_rule_permission,
            acl_rule_source,
            acl_rule_destination,
            acl_rule_protocol,
            acl_rule_port,
            acl_rule_position,
        )

    # TODO: confirm typehint using runtime
    def create_services_list(self, services: Dict) -> None:
        """
        Creates a list of services (enum) from config data.

        Args:
            item: A config data item representing the services
        """
        service_list = services["service_list"]

        for service in service_list:
            service_name = service["name"]
            self.services_list.append(service_name)

        # Set the number of services
        self.num_services = len(self.services_list)

    def create_ports_list(self, ports: Dict) -> None:
        """
        Creates a list of ports from config data.

        Args:
            item: A config data item representing the ports
        """
        ports_list = ports["ports_list"]

        for port in ports_list:
            port_value = port["port"]
            self.ports_list.append(port_value)

        # Set the number of ports
        self.num_ports = len(self.ports_list)

    # TODO: this is not used anymore, write a ticket to delete it
    def get_observation_info(self, observation_info: Dict) -> None:
        """
        Extracts observation_info.

        :param observation_info: Config item that defines which type of observation space to use
        :type observation_info: str
        """
        self.observation_type = ObservationType[observation_info["type"]]

    # TODO: this is not used anymore, write a ticket to delete it.
    def get_action_info(self, action_info: Dict) -> None:
        """
        Extracts action_info.

        Args:
            item: A config data item representing action info
        """
        self.action_type = ActionType[action_info["type"]]

    def save_obs_config(self, obs_config: dict) -> None:
        """
        Cache the config for the observation space.

        This is necessary as the observation space can't be built while reading the config,
        it must be done after all the nodes, links, and services have been initialised.

        :param obs_config: Parsed config relating to the observation space. The format is described in
        :py:meth:`primaite.environment.observations.ObservationsHandler.from_config`
        :type obs_config: dict
        """
        self.obs_config = obs_config

    def reset_environment(self) -> None:
        """
        Resets environment.

        Uses config data config data in order to build the environment configuration.
        """
        for item in self.lay_down_config:
            if item["item_type"] == "NODE":
                # Reset a node's state (normal and reference)
                self.reset_node(item)
            elif item["item_type"] == "ACL_RULE":
                # Create an ACL rule (these are cleared on reset, so just need to recreate them)
                self.create_acl_rule(item)
            else:
                # Do nothing (bad formatting or not relevant to reset)
                pass

        # Reset the IER status so they are not running initially
        # Green IERs
        for ier_key, ier_value in self.green_iers.items():
            ier_value.set_is_running(False)
        # Red IERs
        for ier_key, ier_value in self.red_iers.items():
            ier_value.set_is_running(False)

    def reset_node(self, item: Dict) -> None:
        """
        Resets the statuses of a node.

        Args:
            item: A config data item
        """
        # All nodes have these parameters
        node_id = item["node_id"]
        node_class = item["node_class"]
        node_hardware_state: HardwareState = HardwareState[item["hardware_state"]]

        node: NodeUnion = self.nodes[node_id]
        node_ref = self.nodes_reference[node_id]

        # Reset the hardware state (common for all node types)
        node.hardware_state = node_hardware_state
        node_ref.hardware_state = node_hardware_state

        if node_class == "ACTIVE":
            # Active nodes have Software State
            node_software_state = SoftwareState[item["software_state"]]
            node_file_system_state = FileSystemState[item["file_system_state"]]
            node.software_state = node_software_state
            node_ref.software_state = node_software_state
            node.set_file_system_state(node_file_system_state)
            node_ref.set_file_system_state(node_file_system_state)
        elif node_class == "SERVICE":
            # Service nodes have Software State and list of services
            node_software_state = SoftwareState[item["software_state"]]
            node_file_system_state = FileSystemState[item["file_system_state"]]
            node.software_state = node_software_state
            node_ref.software_state = node_software_state
            node.set_file_system_state(node_file_system_state)
            node_ref.set_file_system_state(node_file_system_state)
            # Update service states
            node_services = item["services"]
            for service in node_services:
                service_protocol = service["name"]
                service_state = SoftwareState[service["state"]]
                # Update node service state
                node.set_service_state(service_protocol, service_state)
                # Update reference node service state
                node_ref.set_service_state(service_protocol, service_state)
        else:
            # Bad formatting
            pass

    def create_node_action_dict(self) -> Dict[int, List[int]]:
        """
        Creates a dictionary mapping each possible discrete action to more readable multidiscrete action.

        Note: Only actions that have the potential to change the state exist in the mapping (except for key 0)

        example return:
        {0: [1, 0, 0, 0],
        1: [1, 1, 1, 0],
        2: [1, 1, 2, 0],
        3: [1, 1, 3, 0],
        4: [1, 2, 1, 0],
        5: [1, 3, 1, 0],
        ...
        }
        """
        # Terms (for node action space):
        # [0, num nodes] - node ID (0 = nothing, node ID)
        # [0, 4] - what property it's acting on (0 = nothing, state, SoftwareState, service state, file system state) # noqa
        # [0, 3] - action on property (0 = nothing, On / Scan, Off / Repair, Reset / Patch / Restore) # noqa
        # [0, num services] - resolves to service ID (0 = nothing, resolves to service) # noqa
        # reserve 0 action to be a nothing action
        actions = {0: [1, 0, 0, 0]}
        action_key = 1
        for node in range(1, self.num_nodes + 1):
            # 4 node properties (NONE, OPERATING, OS, SERVICE)
            for node_property in range(4):
                # Node Actions either:
                # (NONE, ON, OFF, RESET) - operating state OR (NONE, PATCH) - OS/service state
                # Use MAX to ensure we get them all
                for node_action in range(4):
                    for service_state in range(self.num_services):
                        action = [node, node_property, node_action, service_state]
                        # check to see if it's a nothing action (has no effect)
                        if is_valid_node_action(action):
                            actions[action_key] = action
                            action_key += 1

        return actions

    def create_acl_action_dict(self) -> Dict[int, List[int]]:
        """Creates a dictionary mapping each possible discrete action to more readable multidiscrete action."""
        # Terms (for ACL action space):
        # [0, 2] - Action (0 = do nothing, 1 = create rule, 2 = delete rule)
        # [0, 1] - Permission (0 = DENY, 1 = ALLOW)
        # [0, num nodes] - Source IP (0 = any, then 1 -> x resolving to IP addresses)
        # [0, num nodes] - Dest IP (0 = any, then 1 -> x resolving to IP addresses)
        # [0, num services] - Protocol (0 = any, then 1 -> x resolving to protocol)
        # [0, num ports] - Port (0 = any, then 1 -> x resolving to port)
        # [0, max acl rules - 1] - Position (0 = first index, then 1 -> x index resolving to acl rule in acl list)
        # reserve 0 action to be a nothing action
        actions = {0: [0, 0, 0, 0, 0, 0, 0]}

        action_key = 1
        # 3 possible action decisions, 0=NOTHING, 1=CREATE, 2=DELETE
        for action_decision in range(3):
            # 2 possible action permissions 0 = DENY, 1 = CREATE
            for action_permission in range(2):
                # Number of nodes + 1 (for any)
                for source_ip in range(self.num_nodes + 1):
                    for dest_ip in range(self.num_nodes + 1):
                        for protocol in range(self.num_services + 1):
                            for port in range(self.num_ports + 1):
                                for position in range(self.max_number_acl_rules - 1):
                                    action = [
                                        action_decision,
                                        action_permission,
                                        source_ip,
                                        dest_ip,
                                        protocol,
                                        port,
                                        position,
                                    ]
                                    # Check to see if it is an action we want to include as possible
                                    # i.e. not a nothing action
                                    if is_valid_acl_action_extra(action):
                                        actions[action_key] = action
                                        action_key += 1

        return actions

    def create_node_and_acl_action_dict(self) -> Dict[int, List[int]]:
        """
        Create a dictionary mapping each possible discrete action to a more readable mutlidiscrete action.

        The dictionary contains actions of both Node and ACL action types.
        """
        node_action_dict = self.create_node_action_dict()
        acl_action_dict = self.create_acl_action_dict()

        # Change node keys to not overlap with acl keys
        # Only 1 nothing action (key 0) is required, remove the other
        new_node_action_dict = {k + len(acl_action_dict) - 1: v for k, v in node_action_dict.items() if k != 0}

        # Combine the Node dict and ACL dict
        combined_action_dict = {**acl_action_dict, **new_node_action_dict}
        return combined_action_dict

    def _create_random_red_agent(self) -> None:
        """Decide on random red agent for the episode to be called in env.reset()."""
        # Reset the current red iers and red node pol
        self.red_iers = {}
        self.red_node_pol = {}

        # Decide how many nodes become compromised
        node_list = list(self.nodes.values())
        computers = [node for node in node_list if node.node_type == NodeType.COMPUTER]
        max_num_nodes_compromised = len(computers)  # only computers can become compromised
        # random select between 1 and max_num_nodes_compromised
        num_nodes_to_compromise = randint(1, max_num_nodes_compromised)

        # Decide which of the nodes to compromise
        nodes_to_be_compromised = sample(computers, num_nodes_to_compromise)

        # choose a random compromise node to be source of attacks
        source_node = choice(nodes_to_be_compromised)

        # For each of the nodes to be compromised decide which step they become compromised
        max_step_compromised = self.episode_steps // 2  # always compromise in first half of episode

        # Bandwidth for all links
        bandwidths = [i.get_bandwidth() for i in list(self.links.values())]

        if len(bandwidths) < 1:
            msg = "Random red agent cannot be used on a network without any links"
            _LOGGER.error(msg)
            raise Exception(msg)

        servers = [node for node in node_list if node.node_type == NodeType.SERVER]

        for n, node in enumerate(nodes_to_be_compromised):
            # 1: Use Node PoL to set node to compromised

            _id = str(uuid.uuid4())
            _start_step = randint(2, max_step_compromised + 1)  # step compromised
            pol_service_name = choice(list(node.services.keys()))

            source_node_service = choice(list(source_node.services.values()))

            red_pol = NodeStateInstructionRed(
                _id=_id,
                _start_step=_start_step,
                _end_step=_start_step,  # only run for 1 step
                _target_node_id=node.node_id,
                _pol_initiator="DIRECT",
                _pol_type=NodePOLType["SERVICE"],
                pol_protocol=pol_service_name,
                _pol_state=SoftwareState.COMPROMISED,
                _pol_source_node_id=source_node.node_id,
                _pol_source_node_service=source_node_service.name,
                _pol_source_node_service_state=source_node_service.software_state,
            )

            self.red_node_pol[_id] = red_pol

            # 2: Launch the attack from compromised node - set the IER

            ier_id = str(uuid.uuid4())
            # Launch the attack after node is compromised, and not right at the end of the episode
            ier_start_step = randint(_start_step + 2, int(self.episode_steps * 0.8))
            ier_end_step = self.episode_steps

            # Randomise the load, as a percentage of a random link bandwith
            ier_load = uniform(0.4, 0.8) * choice(bandwidths)
            ier_protocol = pol_service_name  # Same protocol as compromised node
            ier_service = node.services[pol_service_name]
            ier_port = ier_service.port
            ier_mission_criticality = 0  # Red IER will never be important to green agent success
            # We choose a node to attack based on the first that applies:
            # a. Green IERs, select dest node of the red ier based on dest node of green IER
            # b. Attack a random server that doesn't have a DENY acl rule in default config
            # c. Attack a random server
            possible_ier_destinations = [
                ier.get_dest_node_id()
                for ier in list(self.green_iers.values())
                if ier.get_source_node_id() == node.node_id
            ]
            if len(possible_ier_destinations) < 1:
                for server in servers:
                    if not self.acl.is_blocked(
                        node.ip_address,
                        server.ip_address,
                        ier_service,
                        ier_port,
                    ):
                        possible_ier_destinations.append(server.node_id)
            if len(possible_ier_destinations) < 1:
                # If still none found choose from all servers
                possible_ier_destinations = [server.node_id for server in servers]
            ier_dest = choice(possible_ier_destinations)
            self.red_iers[ier_id] = IER(
                ier_id,
                ier_start_step,
                ier_end_step,
                ier_load,
                ier_protocol,
                ier_port,
                node.node_id,
                ier_dest,
                ier_mission_criticality,
            )

            overwhelm_pol = red_pol
            overwhelm_pol.id = str(uuid.uuid4())
            overwhelm_pol.end_step = self.episode_steps

            # 3: Make sure the targetted node can be set to overwhelmed - with node pol
            # # TODO remove duplicate red pol for same targetted service - must take into account start step

            o_pol_id = str(uuid.uuid4())
            o_red_pol = NodeStateInstructionRed(
                _id=o_pol_id,
                _start_step=ier_start_step,
                _end_step=self.episode_steps,
                _target_node_id=ier_dest,
                _pol_initiator="DIRECT",
                _pol_type=NodePOLType["SERVICE"],
                pol_protocol=ier_protocol,
                _pol_state=SoftwareState.OVERWHELMED,
                _pol_source_node_id=source_node.node_id,
                _pol_source_node_service=source_node_service.name,
                _pol_source_node_service_state=source_node_service.software_state,
            )
            self.red_node_pol[o_pol_id] = o_red_pol
