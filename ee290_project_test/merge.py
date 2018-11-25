"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,  # TODO: get good number
}


class WaveAttenuationMergePOEnv(Env):
    # TODO: fix docstring
    """Partially observable merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

        In order to account for variability in the number of autonomous
        vehicles, if n_AV < "num_rl" the additional actions provided by the
        agent are not assigned to any vehicle. Moreover, if n_AV > "num_rl",
        the additional vehicles are not provided with actions from the learning
        agent, and instead act as human-driven vehicles as well.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params["num_rl"]
        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()
        # names of the rl vehicles controlled at any step
        self.rl_veh = []
        # used for visualization
        self.leader = []
        self.follower = []
        self.communication = [0 for _ in range(self.num_rl)]

        super().__init__(env_params, sumo_params, scenario)

    @property
    def action_space(self):
        """See class definition."""
        # TODO: bind by 0->1, and denormalize based on what each
        # TODO: of acceleration, lane change, and communication want: DONE
        return Box(
            low=0,
            high=1,
            shape=(3 * self.num_rl, ),
            dtype=np.float32)

    def denormalize(self, value, action_type):
        '''Helper function used to denormalized the 0-1 value of action space
        value: value between 0 and 1
        type: type of the action: 1 for acceleration, 2 for lane change and 3 for communication'''
        if action_type == 1:
            return value * (self.env_params.additional_params["max_accel"] +
                            abs(self.env_params.additional_params["max_decel"])) +\
                   -abs(self.env_params.additional_params["max_decel"])
        elif action_type == 2:
            if value < 1/3:
                return -1
            elif value <= 2/3:
                return 0
            else:
                return 1
        elif action_type == 3:
            # TODO: define communication variable range
            return np.floor(value * 10)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(7 * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.communication = []
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.vehicles.get_rl_ids():
                continue
            self.apply_acceleration([rl_id], [self.denormalize(rl_actions[i], 1)])
            # TODO: add lane change action: DONE
            self.apply_lane_change([rl_id], [self.denormalize(rl_actions[1 * self.num_rl + i], 2)])
            self.communication.append(self.denormalize(rl_actions[2 * self.num_rl + i], 3))

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        self.leader = []
        self.follower = []
        # number of leaders and followers  = num lanes
        # length of a single vehicle's observation space
        num_lanes = 2
        lanes = 5 * num_lanes

        # normalizing constants
        max_speed = self.scenario.max_speed
        max_length = self.scenario.length

        leads_speed = [max_speed for _ in range(num_lanes)]
        leads_head = [max_length for _ in range(num_lanes)]
        follows_speed = [0 for _ in range(num_lanes)]
        follows_head = [max_length for _ in range(num_lanes)]

        observation = [0 for _ in range((6 + lanes) * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.vehicles.get_speed(rl_id)
            leads_id = self.vehicles.get_lane_leaders(rl_id)
            followers = self.vehicles.get_lane_followers(rl_id)

            count = 0
            for lead_id in leads_id:
                if lead_id in ["", None]:
                    # in case leader is not visible
                    leads_speed[count] = max_speed
                    leads_head[count] = max_length
                else:
                    self.leader.append(lead_id)
                    leads_speed[count] = self.vehicles.get_speed(lead_id)
                    leads_head[count] = self.get_x_by_id(lead_id) \
                                        - self.get_x_by_id(rl_id) - self.vehicles.get_length(rl_id)

            count = 0
            for follower in followers:
                if follower in ["", None]:
                    # in case follower is not visible
                    follows_speed[count] = 0
                    follows_head[count] = max_length
                else:
                    self.follower.append(follower)
                    follows_speed[count] = self.vehicles.get_speed(follower)
                    follows_head[count] = self.vehicles.get_headway(follower)

            # TODO: try local communication (so maybe nearest neighbors in each lane in front and behind)
            comm = (np.sum(self.communication) - self.communication[i]) / (len(self.communication) - 1)
            position = self.vehicles.get_absolute_position(rl_id)
            lane = self.vehicles.get_lane(rl_id)
            route = int(self.vehicles.get_state(rl_id, "type"))

            # this vel's normalized speed
            observation[5 * i + 0] = this_speed / max_speed
            # lane leaders' normalized speed
            observation[5 * i + 1: 5 * i + 1 + 1 * lanes] = np.subtract(leads_speed, this_speed) / max_speed
            # lane leaders' normalized headway
            observation[5 * i + 1 + 1 * lanes: 5 * i + 1 + 2 * lanes] = np.divide(leads_head, max_length)
            # lane followers' normalized speed
            observation[5 * i + 1 + 2 * lanes: 5 * i + 1 + 3 * lanes] = \
                np.subtract(this_speed, follows_speed) / max_speed
            # lane followers' normalized headway
            observation[5 * i + 1 + 3 * lanes: 5 * i + 1 + 4 * lanes] = \
                np.divide(follows_head, max_length)
            # TODO: try position, lane, lane leaders and followers, route (1, 2, 3, 4): DONE
            observation[5 * i + 4 * lanes + 1] = comm
            observation[5 * i + 4 * lanes + 2] = position
            observation[5 * i + 4 * lanes + 3] = lane
            observation[5 * i + 4 * lanes + 4] = route
            observation[5 * i + 4 * lanes + 5] = self.only_numerics(rl_id)  # FIXME: this is a string, not int: DONE

        return observation

    @staticmethod
    def only_numerics(seq):
        seq_type = type(seq)
        return seq_type().join(filter(seq_type.isdigit, seq))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return -1

    def sort_by_position(self):
        """See parent class.

        Sorting occurs by the ``get_x_by_id`` method instead of
        ``get_absolute_position``.
        """
        # vehicles are sorted by their get_x_by_id value
        sorted_ids = sorted(self.vehicles.get_ids(), key=self.get_x_by_id)
        return sorted_ids, None

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.vehicles.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.vehicles.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.vehicles.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.vehicles.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()
