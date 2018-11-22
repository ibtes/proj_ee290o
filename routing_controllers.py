"""Contains a list of custom routing controllers."""

from flow.controllers.base_routing_controller import BaseRouter


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed loop.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.
    """

    def choose_route(self, env):
        """Adopt the current edge's route if about to leave the network."""
        if env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            return env.available_routes[env.vehicles.get_edge(self.veh_id)]
        else:
            return None


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle within a grid environment."""

    def choose_route(self, env):
        if env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            new_route = [env.vehicles.get_edge(self.veh_id)]
        else:
            new_route = None

        return new_route


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge scenario.

    Extension to the Continuous Router.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.vehicles.get_edge(self.veh_id)
        lane = env.vehicles.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"]
        else:
            new_route = super().choose_route(env)

        return new_route


class WeaveRouter(BaseRouter):
    """fsjafhen u"""

    def choose_route(self, env):
        """

        :param env:
        :return:
        """
        route = env.vehicles.get_route(self.veh_id)
        type_id = env.vehicles.get_state(self.veh_id, "type")
        edge = env.vehicles.get_edge(self.veh_id)
        if type_id == '1':
            expected_route = ["inflow_highway", "left", "center", "right"]
        elif type_id == '2':
            expected_route = ["inflow_highway", "left", "center", "diverge"]
        elif type_id == '3':
            expected_route = ["inflow_merge", "merge", "center", "right"]
        elif type_id == '4':
            expected_route = ["inflow_merge", "merge", "center", "diverge"]
        else:
            raise TypeError("oops")

        # print(route, expected_route)

        if route != expected_route and (
                (type_id in ['1', '2'] and edge == 'inflow_highway')
                or (type_id in ['3', '4'] and edge == 'inflow_merge')):
            # print("woop")
            return expected_route
        else:
            return None
