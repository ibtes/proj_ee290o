"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""

from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows
from flow.core.vehicles import Vehicles
from flow.core.experiment import SumoExperiment
from weave import EntryExitScenario, ADDITIONAL_NET_PARAMS
from flow.controllers import IDMController
from routing_controllers import WeaveRouter
from merge import WaveAttenuationMergePOEnv, ADDITIONAL_ENV_PARAMS

# inflow rate at the highway
FLOW_RATE = 250


def merge_example(render=None):
    """
    Perform a simulation of vehicles on a merge.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a merge.
    """
    sumo_params = SumoParams(
        render=True,
        emission_path="./data/",
        sim_step=0.2,
        restart_instance=False)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()
    vehicles.add(
        veh_id="1",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)
    vehicles.add(
        veh_id="2",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)
    vehicles.add(
        veh_id="3",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)
    vehicles.add(
        veh_id="4",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(WeaveRouter, {}),
        speed_mode="all_checks",
        lane_change_mode="strategic",
        num_vehicles=5)

    env_params = EnvParams(
        additional_params=ADDITIONAL_ENV_PARAMS,
        sims_per_step=5,
        warmup_steps=0)

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
        vehs_per_hour=50,  # TODO: change
        departLane="free",
        departSpeed=7.5)
    inflow.add(
        veh_type="4",
        edge="inflow_merge",
        vehs_per_hour=50,  # TODO: change
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

    env = WaveAttenuationMergePOEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = merge_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 3600, convert_to_csv=False)
