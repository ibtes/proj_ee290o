# import matplotlib.pyplot as plt
import numpy as np
from hw3_utils import get_params, HORIZON
from merge import WaveAttenuationMergePOEnv
from train import REINFORCE

class TestEnvironment(WaveAttenuationMergePOEnv):
    def compute_reward(self, rl_actions, **kwargs):
        return np.mean(self.vehicles.get_speed(self.vehicles.get_ids()))


# create the gym environment
sumo_params, env_params, scenario = get_params(render=True)

env = TestEnvironment(env_params=env_params,
                      sumo_params=sumo_params,
                      scenario=scenario)
env._max_episode_steps = HORIZON

# run the environment for one rollout with the trained policy
alg = REINFORCE(env, stochastic=True)
alg.load_checkpoint("REINFORCE" + ".ckpt")
samples = alg.rollout()


# # plot the results from that rollout
# avg_speeds = samples["reward"]
#
# plt.figure(figsize=(12,8))
# plt.title("Stabilizing Controller Performance", fontsize=25)
# plt.xlabel("time step", fontsize=20)
# plt.ylabel("average speed (m/s)", fontsize=20)
# plt.plot(avg_speeds, c='k', label="average speed with one AV")
# plt.plot([4.82] * len(avg_speeds), '--', c='b', label="non-perturbed equilibrium")
# plt.plot([3.28] * len(avg_speeds), '--', c='r', label="perturbed equilibrium")
# plt.legend(fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.show()