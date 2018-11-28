import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import os
from merge import WaveAttenuationMergePOEnv


class DirectPolicyAlgorithm(object):

    def __init__(self,
                 env,
                 linear=False,
                 stochastic=False,
                 hidden_size=[64, 64],
                 nonlinearity=tf.nn.relu):
        """Instantiate the policy iteration class.

        This initializes the policy model with a set of trainable 
        parameters, and creates a tensorflow session and a saver 
        to save checkpoints during training.

        In order to train an algorithm using this class, type:

            >>> alg = DirectPolicyAlgorithm(...)
            >>> alg.train(...)

        Note that the "train" method is abstract, and needs to be
        filled in by a child class.

        Attributes
        ----------
        env : gym.Env
            the environment that the policy will be trained on
        linear : bool, optional
            specifies whether to use a linear or neural network 
            policy, defaults to False (i.e. use a fully-connected
            network)
        stochastic : bool, optional
            specifies whether to use a stochastic or deterministic 
            policy, defaults to False (i.e. deterministic policy)
        hidden_size : list of int, optional
            list of hidden layers, with each value corresponding 
            to the number of nodes in that layer 
        nonlinearity : tf.nn.*
            activation nonlinearity
        """
        # clear any previous computation graph
        tf.reset_default_graph()

        # set a random seed
        tf.set_random_seed(1234)

        # start a tensorflow session
        self.sess = tf.Session()

        # environment to train on
        self.env = env

        # number of elements in the action space
        self.ac_dim = env.action_space.shape[0]
        # number of elements in the observation space
        self.obs_dim = env.observation_space.shape[0]

        # mean state (substracted from the state before computing 
        # actions). This is used in problem 3.
        self.mean_state = np.zeros(self.obs_dim)

        # actions placeholder
        self.a_t_ph = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.ac_dim])
        # state placeholder
        self.s_t_ph = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.obs_dim])
        # expected reward placeholder
        self.rew_ph = tf.placeholder(dtype=tf.float32,
                                     shape=[None])

        # specifies whether the policy is stochastic
        self.stochastic = stochastic

        # policy that the agent executes during training/testing
        self.policy = self.create_model(
            args={
                "num_actions": self.ac_dim,
                "hidden_size": hidden_size,
                "linear": linear,
                "nonlinearity": nonlinearity,
                "stochastic": stochastic,
                "scope": "policy",
            }
        )

        self.symbolic_action = self.action_op()

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # create saver to save model variables
        self.saver = tf.train.Saver()

    def create_model(self, args):
        """Create a model for your policy or other components.

        This model may be linear or a fully-connected network. In
        addition, a logstd variable may be specified if the policy
        is stochastic, otherwise, the logstd output is set to None.

        Parameters
        ----------
        args : dict
            model-specific arguments, with keys:
              - "stochastic": a boolean operator that specifies
                whether the policy is meant to be stoachstic or
                deterministic. If it is stochastic, an additional
                trainable variable is created to compute the logstd 
                of an action given. This variable is not dependent 
                on the input state.
              - "hidden_size": a list that specified the shape 
                of the neural networ (if "linear" is False)
              - "num_actions" number of output actions
              - "scope": scope of the model

        Returns
        -------
        tf.Variable
            mean actions of the policy
        tf.Variable or None
            logstd of the policy actions. If the policy is deterministic,
            this term is None
        """
        with tf.variable_scope(args["scope"]):
            # create the hidden layers
            last_layer = self.s_t_ph
            for i, hidden_size in enumerate(args["hidden_size"]):
                last_layer = tf.layers.dense(
                    inputs=last_layer,
                    units=hidden_size,
                    activation=args["nonlinearity"])

            # create the output layer
            output_mean = tf.layers.dense(
                inputs=last_layer,
                units=args["num_actions"],
                activation=tf.nn.tanh)

        if args["stochastic"]:
            ################################################################
            # Create a trainable variable whose output is the same size    #
            # as the action space. This variable will represent the output #
            # log standard deviation of your stochastic policy.            #
            #                                                              #
            # Refer to the __init__ method in when choosing an appropriate #
            # shape of your variable.                                      #
            #                                                              #
            # Note: To create this variable, use the `tf.get_variable`     #
            # function.                                                    #
            ################################################################
            output_logstd = tf.get_variable('output_logstd', shape=[self.ac_dim], dtype=tf.float32)  ### FILL IN ###
        else:
            output_logstd = None

        return output_mean, output_logstd

    def action_op(self):
        """Create a symbolic expression that will be used to compute 
        actions from observations.

        If the policy is determistic, the action is simply dictated by
        the output of the policy mean.

        Alternatively, if the policy is stochastic, the action is:

            a_t = output_mean + exp(output_logstd) * z; z ~ N(0,1)
        """
        if self.stochastic:
            output_mean, output_logstd = self.policy

            ##############################################################
            # Implement a stochastic version of computing actions.       #
            #                                                            #
            # Remember, the action in a stochastic policy represented by #
            # a diagonal Gaussian distribution with mean "M" and log     #
            # standard deviation "lstd" is computed as follows:          #
            #                                                            #
            #     a = M + exp(lstd) * z                                  #
            #                                                            #
            # where z is a random normal value, i.e. z ~ N(0,1)          #
            #                                                            #
            # In order to generate numbers from a normal distribution,   #
            # use the `tf.random_normal` function.                       #
            ##############################################################
            symbolic_action = output_mean + tf.exp(output_logstd) * tf.random_normal(
                shape=[self.ac_dim])  ### FILL IN ###
        else:
            symbolic_action, _ = self.policy

        return symbolic_action

    def compute_action(self, obs):
        """Returns a list of actions for a given observation.

        Parameters
        ----------
        obs : np.ndarray
            observations

        Returns
        -------
        np.ndarray
            actions by the policy for a given observation
        """
        return self.sess.run(self.symbolic_action,
                             feed_dict={self.s_t_ph: obs})

    def cleanState(self, state):
        numObservation = 7
        cleanState = []
        for i in range(int(len(state) / numObservation)):
            cleanState.append(state[i * numObservation:i * numObservation + (numObservation - 1)])
        return cleanState

    def rollout(self):
        """Collect samples from one rollout of the policy.

        Returns
        -------
        dict
            dictionary conta.ining trajectory information for the rollout,
            specifically containing keys for "state", "action", 
            "next_state", "reward", and "done"
        """
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        # start a new rollout by resetting the environment and 
        # collecting the initial state
        state = self.env.reset()
        steps = 0
        while True:
            steps += 1

            # compute the action given the state
            # TODO: define 1 2 3 4. How many actions are there???
            # ok... only 5 rls, but get_rl_ids gives 20 rls?
            a1 = []
            a2 = []
            a3 = []
            a4 = []  # communication?
            action = []
            cleanState = self.cleanState(state)
            for cs in cleanState:
                ac = self.compute_action([cs])
                a1.append(ac[0][0])
                a2.append(ac[0][1])
                a3.append(ac[0][2])
                a4.append(ac[0][0])
                a4.append(ac[0][1])
                a4.append(ac[0][2])
            action = a1 + a2 + a3
            # action = self.compute_action([self.cleanState(state)])
            # action=action[0]
            # input(action)
            # advance the environment once and collect the next state, 
            # reward, done, and info parameters from the environment
            next_state, reward, done, info = self.env.step(action)
            # v,l_v,l_h,f_v,f_h,comm,id
            # add to the samples list
            states.append(state)
            actions.append(a4)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            # print(state)
            state = next_state

            # if the environment returns a True for the done parameter,
            # end the rollout before the time horizon is met
            if done or steps > self.env._max_episode_steps:
                break

        veh_traj_obj = {}
        for i in range(len(states)):
            for j in range(int(len(states[i]) / 7)):
                num_nonzero = np.count_nonzero(states[i][j * 7:j * 7 + 7])
                if (num_nonzero == 0):
                    pass
                else:
                    veh_id = states[i][j * 7 + 6]
                    if veh_id not in veh_traj_obj:
                        veh_traj_obj[veh_id] = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}
                    veh_traj_obj[veh_id]["state"].append(states[i][j * 7:j * 7 + 6])
                    veh_traj_obj[veh_id]["done"].append(dones[i])
                    veh_traj_obj[veh_id]["action"].append(actions[i][j * 3:j * 3 + 3])
                    veh_traj_obj[veh_id]["reward"].append(rewards[i])
                    veh_traj_obj[veh_id]["next_state"].append(next_states[i][j * 7:j * 7 + 6])

        final_stage_ids = []
        for i in range(int(len(states[len(states) - 1]) / 7)):
            if states[len(states) - 1][i * 7 + 6] != 0:
                final_stage_ids.append(states[len(states) - 1][i * 7 + 6])
        for i in range(int(len(states[0]) / 7)):
            if states[0][i * 7 + 6] != 0:
                final_stage_ids.append(states[0][i * 7 + 6])
        trajectory = []
        for elem in veh_traj_obj:
            if elem not in final_stage_ids:
                veh_traj_obj[elem]["state"] = np.array(veh_traj_obj[elem]["state"], dtype=np.float32)
                veh_traj_obj[elem]["reward"] = np.array(veh_traj_obj[elem]["reward"], dtype=np.float32)
                veh_traj_obj[elem]["action"] = np.array(veh_traj_obj[elem]["action"], dtype=np.float32)
                veh_traj_obj[elem]["done"] = np.array(veh_traj_obj[elem]["done"], dtype=np.float32)
                veh_traj_obj[elem]["next_state"] = np.array(veh_traj_obj[elem]["next_state"], dtype=np.float32)
                trajectory.append(veh_traj_obj[elem])

        # create the output trajectory
        #         trajectory = {"state": np.array(states, dtype=np.float32),
        #                       "reward": np.array(rewards, dtype=np.float32),
        #                       "action": np.array(actions, dtype=np.float32),
        #                       "next_state": np.array(next_states, dtype=np.float32),
        #                       "done": np.array(dones, dtype=np.float32)}

        return trajectory

    def train(self, args):
        """Abstract training method.

        This method will be filled in by algorithm-specific
        training operations in subsequent problems.

        Parameters
        ----------
        args : dict
            algorithm-specific hyperparameters
        """
        raise NotImplementedError

    def save_checkpoint(self, filename):
        """Save a checkpoint for later viewing."""
        current_dir = os.getcwd()
        save_loc = os.path.join(current_dir, filename)
        self.saver.save(self.sess, save_loc)

    def load_checkpoint(self, filename):
        """Load the model from a specific checkpoint."""
        current_dir = os.getcwd()
        save_loc = os.path.join(current_dir, filename)
        self.saver.restore(self.sess, save_loc)


import tensorflow as tf
import numpy as np
import time


class REINFORCE(DirectPolicyAlgorithm):

    def train(self,
              num_iterations=100,
              steps_per_iteration=1000,
              learning_rate=0.001,
              gamma=0.95,
              **kwargs):
        """Perform the REINFORE training operation.

        Parameters
        ----------
        num_iterations : int
            number of training iterations
        steps_per_iteration : int
            number of individual samples collected every training
            iteration
        learning_rate : float
            optimizer learning rate
        gamma : float
            discount rate
        kwargs : dict
            additional arguments

        Returns
        -------
        list of float
            average return per iteration
        """
        # set the discount as an attribute
        self.gamma = gamma

        # set the learning rate as an attribute
        self.learning_rate = learning_rate

        # create a symbolic expression to compute the log-likelihoods 
        log_likelihoods = self.log_likelihoods()

        # create a symbolic expression for updating the parameters of 
        # your policy
        #
        # Note: the second operator will be used in problem 2.b, please 
        # ignore when solving 2.a
        self.opt, self.opt_baseline = self.define_updates(log_likelihoods)

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # average return per training iteration
        ret_per_iteration = []

        samples = []
        for i in range(num_iterations):
            # collect samples from the current policy
            samples.clear()
            steps_so_far = 0
            while steps_so_far < steps_per_iteration:
                new_samples = self.rollout()
                # print(len(new_samples))
                for n_s in new_samples:
                    steps_so_far += n_s["action"].shape[0]
                    samples.append(n_s)
                # print(steps_so_far,steps_per_iteration)

            # compute the expected returns
            v_s = self.compute_expected_return(samples)

            # compute and apply the gradients
            self.call_updates(log_likelihoods, samples, v_s, **kwargs)

            # compute the average cumulative return per iteration
            average_rew = np.mean([sum(s["reward"]) for s in samples])
            # print([sum(s["reward"]) for s in samples])
            # display iteration statistics
            print("Iteration {} return: {}".format(i, average_rew))
            ret_per_iteration.append(average_rew)

        return ret_per_iteration

    def log_likelihoods(self):
        """Create a tensorflow operation that computes the log-likelihood 
        of each performed action.

        Remember, the actions in this case are not deterministic, but 
        rather sampled from a Gaussian distribution; accordingly, there 
        is a probability associated with each action occuring.
        """
        # tensors representing the mean and standard deviation of
        # performing a specific action given a state (these are the 
        # parameters of your policy)
        output_mean, output_logstd = self.policy

        ##############################################################
        # Create a tf operation to compute the log-likelihood of     #
        # each action that was performed by the policy during the    #
        # trajectories.                                              #
        #                                                            #
        # The log likelihood in the continuous case where the policy #
        # is expressed by a multivariate gaussian can be computing   #
        # using the tensorflow object:                               #
        #                                                            #
        #    p = tf.contrib.distributions.MultivariateNormalDiag(    #
        #        loc=...,                                            #
        #        scale_diag=...,                                     #
        #    )                                                       #
        #                                                            #
        # This method takes as input a mean (loc) and standard       #
        # deviation (scale_diag), and then can be used to compute    #
        # the log-likelihood of a variable as follows:               #
        #                                                            #
        #    log_likelihoods = p.log_prob(...)                       #
        #                                                            #
        # For this operation, you will want to use placeholders      #
        # created in the __init__ method of problem 1.               #
        ##############################################################
        p = tf.contrib.distributions.MultivariateNormalDiag(loc=output_mean, scale_diag=tf.exp(output_logstd))
        log_likelihoods = p.log_prob(self.a_t_ph)  ### FILL IN ###

        return log_likelihoods

    def compute_expected_return(self, samples):
        """Compute the expected return from a given starting state.

        This is done by using the reward-to-go method.

        Parameters
        ----------
        rewards : list of list of float
            a list of N trajectories, with each trajectory contain T 
            returns values (one for each step in the trajectory)

        Returns
        -------
        list of float, or np.ndarray
            expected returns for each step in each trajectory
        """
        rewards = [s["reward"] for s in samples]

        ##############################################################
        # Estimate the expected return from any given starting state #
        # using the reward-to-go method.                             #
        #                                                            #
        # Using this method, the reward is estimated at every step   #
        # of the trajectory as follows:                              #
        #                                                            #
        #   r = sum_{t'=t}^T gamma^(t'-t) * r_{t'}                   #
        #                                                            #
        # where T is the time horizon at t is the index of the       #
        # current reward in the trajectory. For example, for a given #
        # set of rewards r = [1,1,1,1] and discount rate gamma = 1,  #
        # the expected reward-to-go would be:                        #
        #                                                            #
        #   v_s = [4, 3, 2, 1]                                       #
        #                                                            #
        # You will be able to test this in one of the cells below!   #
        ##############################################################
        v_s = []  ### FILL IN ###
        for r_i in rewards:
            r = []
            size = len(r_i)
            for i in range(size):
                if i == 0:
                    r.append(r_i[size - i - 1])
                else:
                    r.append(self.gamma * r[i - 1] + r_i[size - i - 1])
            v_s.append(r[::-1])
        # flatten
        v_s = [i for j in v_s for i in j]
        # print(v_s)
        return v_s

    def define_updates(self, log_likelihoods):
        """Create a tensorflow operation to update the parameters of 
        your policy.

        Parameters
        ----------
        log_likelihoods : tf.Operation
            the symbolic expression you created to estimate the log 
            likelihood of a set of actions

        Returns
        -------
        tf.Operation
            a tensorflow operation for computing and applying the 
            gradients to the parameters of the policy
        None
            the second component is used in problem 2.b, please ignore 
            for this problem
        """
        ##############################################################
        # Specify a loss function that can be used to compute the    #
        # gradient of denoted at the start of problem 2.             #
        #                                                            #
        # Note: remember we are trying to **maximize** this value    #
        #                                                            #
        # For this operation, you will want to use placeholders      #
        # created in the __init__ method of problem 1, as well as    #
        # the operations provided as inputs to this problem.         #
        ##############################################################
        loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(self.rew_ph, log_likelihoods)))  ### FILL IN ###
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        return opt, None

    def call_updates(self,
                     log_likelihoods,
                     samples,
                     v_s,
                     **kwargs):
        """Apply the gradient update methods in a tensorflow session.

        Parameters
        ----------
        log_likelihoods: tf.Operation
            the symbolic expression you created to estimate the log 
            likelihood of a set of actions
        samples : list of dict
            a list of N trajectories, with each trajectory containing 
            a dictionary of trajectory data (see self.rollout)
        v_s : list of float, or np.ndarray
            the estimated expected returns from your
            `comput_expected_return` function 
        kwargs : dict
            additional arguments (used in question 3)
        """
        # concatenate the states
        states = np.concatenate([s["state"] for s in samples])

        # concatenate the actions
        actions = np.concatenate([s["action"] for s in samples])

        ##############################################################
        # Fill in the feed_dict component below to properly execute  #
        # the optimization step. Refer to the variables formed in    #
        # this problem as well as the __init__ method in problem 1   # 
        # when doing so.                                             #
        ##############################################################
        self.sess.run(self.opt,
                      feed_dict={self.s_t_ph: states, self.a_t_ph: actions, self.rew_ph: v_s})  ### FILL IN ###
        self.sess.run(self.opt,
                      feed_dict={self.s_t_ph: states, self.a_t_ph: actions, self.rew_ph: v_s})  ### FILL IN ###


def train_REINFORCE():
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
        "num_rl": 50,  # TODO: get good number
    }

    ### creating the gym environment
    from hw3_utils import get_params, HORIZON


if __name__ == "__main__":
    ### creating the gym environment
    from hw3_utils import get_params, HORIZON

    sumo_params, env_params, scenario = get_params(render=False)
    env_params.additional_params = {"max_accel": 3, "max_decel": 3, "target_velocity": 25, "num_rl": 50}

    env = WaveAttenuationMergePOEnv(env_params=env_params,
                                    sumo_params=sumo_params,
                                    scenario=scenario)
    env._max_episode_steps = HORIZON

    alg = REINFORCE(env, stochastic=True)

    # feel free to modify the hyperparameters
    cum_rewards = alg.train(learning_rate=0.1, gamma=0.99,
                            steps_per_iteration=1000, num_iterations=300)

    alg.save_checkpoint("REINFORCE.ckpt")
    np.savetxt("REINFORCE.csv", cum_rewards, delimiter=",")
    sumo_params, env_params, scenario = get_params()
    env_params.additional_params = {"max_accel": 3, "max_decel": 3, "target_velocity": 25, "num_rl": 50}

    env = WaveAttenuationMergePOEnv(env_params=env_params,
                                    sumo_params=sumo_params,
                                    scenario=scenario)
    env._max_episode_steps = HORIZON

    ### training on REINFORCE
    import numpy as np

    alg = REINFORCE(env, stochastic=True)

    # feel free to modify the hyperparameters
    cum_rewards = alg.train(learning_rate=0.1, gamma=0.99,
                            steps_per_iteration=3000, num_iterations=300)

    alg.save_checkpoint("REINFORCE.ckpt")
    np.savetxt("REINFORCE.csv", cum_rewards, delimiter=",")

import numpy as np
from hw3_utils import plot_results

res_REINFORCE = np.array([np.genfromtxt("REINFORCE.csv")])
all_results = [res_REINFORCE]
labels = ["REINFORCE"]
plot_results(all_results, labels)

from hw3_utils import get_params, HORIZON

sumo_params, env_params, scenario = get_params(render=True)
env_params.additional_params = {"max_accel": 3, "max_decel": 3, "target_velocity": 25, "num_rl": 50}
env = WaveAttenuationMergePOEnv(env_params=env_params,
                                sumo_params=sumo_params,
                                scenario=scenario)
env._max_episode_steps = 1500

alg = DirectPolicyAlgorithm(env, stochastic=True)
alg.load_checkpoint("REINFORCE" + ".ckpt")
samples = alg.rollout()
