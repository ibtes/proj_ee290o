"""Contains the merge scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.merge import INFLOW_EDGE_LEN
from numpy import pi
from numpy import cos
from numpy import sin

ADDITIONAL_NET_PARAMS = {
    # length of the merge edge
    "merge_length": 100,
    # length of the highway leading to the merge
    "pre_merge_length": 200,
    # length of the highway past the merge and before the diverge
    "post_merge_length": 100,
    # number of lanes in the merge
    "merge_lanes": 1,
    # number of lanes in the highway
    "highway_lanes": 1,
    # max speed limit of the network
    "speed_limit": 30,
    # length of the diverge edge
    "diverge_length": 100,
    # length of the highway past the diverge
    "post_diverge_length": 200,
    # number of lanes in the diverge
    "diverge_lanes": 1,
}


class EntryExitScenario(Scenario):
    """Scenario class for highways with a single in-merge."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a merge scenario.

        Requires from net_params:
        - merge_length: length of the merge edge
        - diverge_length: length of the diverge edge
        - pre_merge_length: length of the highway leading to the merge
        - post_merge_length: length of the highway past the merge and before the diverge
        - post_diverge_length: length of the highway past the diverge
        - merge_lanes: number of lanes in the merge
        - diverge_lanes: number of lanes in the diverge
        - highway_lanes: number of lanes in the highway
        - speed_limit: max speed limit of the network

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        merge = net_params.additional_params["merge_length"]
        diverge = net_params.additional_params["diverge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]
        postdiverge = net_params.additional_params["post_diverge_length"]
        length = merge + diverge + premerge + postmerge + postdiverge + 2 * INFLOW_EDGE_LEN + 8.1  # TODO
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        d_lanes = net_params.additional_params["diverge_lanes"]
        self.name = "{}-{}m{}l-{}l-{}l".format(name, length, h_lanes, m_lanes, d_lanes)

        super().__init__(name, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        # the total length of the network is defined within this function
        self.length = 0

        edgestarts = []
        for edge_id in self._edge_list:
            # the current edge starts (in 1D position) where the last edge
            # ended
            edgestarts.append((edge_id, self.length))
            # increment the total length of the network with the length of the
            # current edge
            self.length += self._edges[edge_id]["length"]

        return edgestarts

    def specify_nodes(self, net_params):
        """See parent class."""
        angle = pi / 4
        merge = net_params.additional_params["merge_length"]
        diverge = net_params.additional_params["diverge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]
        postdiverge = net_params.additional_params["post_diverge_length"]

        nodes = [
            {
                "id": "inflow_highway",
                "x": repr(-INFLOW_EDGE_LEN),
                "y": repr(0)
            },
            {
                "id": "left",
                "y": repr(0),
                "x": repr(0)
            },
            {
                "id": "center_left",
                "y": repr(0),
                "x": repr(premerge)
            },
            {
                "id": "center_right",
                "y": repr(0),
                "x": repr(premerge + postmerge)
            },
            {
                "id": "right",
                "y": repr(0),
                "x": repr(premerge + postmerge + postdiverge)
            },
            {
                "id": "inflow_merge",
                "x": repr(premerge - (merge + INFLOW_EDGE_LEN) * cos(angle)),
                "y": repr(-(merge + INFLOW_EDGE_LEN) * sin(angle))
            },
            {
                "id": "merge",
                "x": repr(premerge - merge * cos(angle)),
                "y": repr(-merge * sin(angle))
            },
            {
                "id": "diverge",
                "x": repr(premerge + postmerge + diverge * cos(angle)),
                "y": repr(-diverge * sin(angle))
            }
        ]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        merge = net_params.additional_params["merge_length"]
        diverge = net_params.additional_params["diverge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]
        postdiverge = net_params.additional_params["post_diverge_length"]

        edges = [{
            "id": "inflow_highway",
            "type": "highwayType",
            "from": "inflow_highway",
            "to": "left",
            "length": repr(INFLOW_EDGE_LEN)
        }, {
            "id": "left",
            "type": "highwayType",
            "from": "left",
            "to": "center_left",
            "length": repr(premerge)
        }, {
            "id": "inflow_merge",
            "type": "mergeType",
            "from": "inflow_merge",
            "to": "merge",
            "length": repr(INFLOW_EDGE_LEN)
        }, {
            "id": "merge",
            "type": "mergeType",
            "from": "merge",
            "to": "center_left",
            "length": repr(merge)
        }, {
            "id": "center",
            "type": "highwayType",
            "from": "center_left",
            "to": "center_right",
            "length": repr(postmerge)
        }, {
            "id": "right",
            "type": "highwayType",
            "from": "center_right",
            "to": "right",
            "length": repr(postdiverge)
        }, {
            "id": "diverge",
            "type": "divergeType",
            "from": "center_right",
            "to": "diverge",
            "length": repr(diverge)
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        h_lanes = net_params.additional_params["highway_lanes"]
        m_lanes = net_params.additional_params["merge_lanes"]
        d_lanes = net_params.additional_params["diverge_lanes"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "numLanes": repr(h_lanes),
            "speed": repr(speed)
        }, {
            "id": "mergeType",
            "numLanes": repr(m_lanes),
            "speed": repr(speed)
        }, {
            "id": "divergeType",
            "numLanes": repr(d_lanes),
            "speed": repr(speed)
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            # TODO
            "1": ["inflow_highway", "left", "center", "right"],
            "2": ["inflow_highway", "left", "center", "diverge"],
            "3": ["inflow_merge", "merge", "center", "right"],
            "4": ["inflow_merge", "merge", "center", "diverge"],
            "inflow_highway": ["inflow_highway", "left", "center", "right"],
            "left": ["left", "center", "right"],  # TODO: complete routes
            "center": ["center", "right"],
            "right": ["right"],
            "inflow_merge": ["inflow_merge", "merge", "center", "right"],
            "merge": ["merge", "center", "right"],
            "diverge": ["diverge"]
        }

        return rts
