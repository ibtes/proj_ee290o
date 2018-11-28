"""Utility objects and methods for homework 3."""
from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows
from flow.core.vehicles import Vehicles
from flow.core.experiment import SumoExperiment
from weave import EntryExitScenario, ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController, RLController
from routing_controllers import WeaveRouter
from merge import WaveAttenuationMergePOEnv, ADDITIONAL_ENV_PARAMS

# import matplotlib.pyplot as plt
import numpy as np

HORIZON = 750/2
FLOW_RATE=300


def get_params(render=False):
    """Create flow-specific parameters for stabilizing the ring experiments.

    Parameters
    ----------
    render : bool, optional
        specifies whether the visualizer is active

    Returns
    -------
    flow.core.params.SumoParams
        sumo-specific parameters
    flow.core.params.EnvParams
        environment-speciifc parameters
    flow.scenarios.Scenario
        a flow-compatible scenario object
    """
    sumo_params = SumoParams(
        # render=True,
        sim_step=0.4,
        sumo_binary="sumo-gui" if render else "sumo",
        seed=0,
        restart_instance=True
    )
    # sumo_params = SumoParams(
    #     render=True,
    #     emission_path="./data/",
    #     sim_step=0.2,
    #     restart_instance=False)

    vehicles = Vehicles()
    vehicles.add(
        veh_id="1",
        acceleration_controller=(RLController, {
            
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)
    vehicles.add(
        veh_id="2",
        acceleration_controller=(RLController, {
            
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)
    vehicles.add(
        veh_id="3",
        acceleration_controller=(RLController, {
            
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)
    vehicles.add(
        veh_id="4",
        acceleration_controller=(RLController, {
            
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)

    additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
    additional_env_params["num_rl"] = 20

    env_params = EnvParams(
        additional_params=additional_env_params,
        sims_per_step=5,
        warmup_steps=0,)

    
    inflow = InFlows()
    inflow.add(
        veh_type="1",
        edge="inflow_highway",
        vehs_per_hour=FLOW_RATE * 0.6,  # TODO: change
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="2",
        edge="inflow_highway",
        vehs_per_hour=FLOW_RATE * 0.4,  # TODO: change
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="3",
        edge="inflow_merge",
        vehs_per_hour=FLOW_RATE * 0.1,  # TODO: change
        departLane="free",
        departSpeed=7.5)
    inflow.add(
        veh_type="4",
        edge="inflow_merge",
        vehs_per_hour=FLOW_RATE * 0.1,  # TODO: change
        departLane="free",
        departSpeed=7.5)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["diverge_lanes"] = 1
    additional_net_params["highway_lanes"] = 2
    additional_net_params["pre_merge_length"] = 300
    additional_net_params["post_merge_length"] = 200
    additional_net_params["post_diverge_length"] = 400
    additional_net_params["merge_length"] = 100
    additional_net_params["diverge_length"] = 200
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", perturbation=5.0)

    scenario = EntryExitScenario(
        name="merge-baseline",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    return sumo_params, env_params, scenario


def plot_results(results, labels):
    """Plot the training curves of multiple algorithms.

    Parameters
    ----------
    results : list of np.ndarray
        results from each algorithms
    labels : list of str
        name of each algorithm
    """
    colors = plt.cm.get_cmap('tab10', len(labels)+1)
    fig = plt.figure(figsize=(16, 9))
    for i, (label, result) in enumerate(zip(labels, results)):
        plt.plot(np.arange(result.shape[1]), np.mean(result, 0),
                 color=colors(i), linewidth=2, label=label)
        plt.fill_between(np.arange(len(result[0])),
                         np.mean(result, 0) - np.std(result, 0),
                         np.mean(result, 0) + np.std(result, 0),
                         alpha=0.25, color=colors(i))
    plt.title("Training Performance of Different Algorithms", fontsize=25)
    plt.ylabel('Cumulative Return', fontsize=20)
    plt.xlabel('Training Iteration', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=20)

    return fig
