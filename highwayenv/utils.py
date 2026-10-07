from gymnasium.envs.registration import registry, register

from highwayenv.intersection_class import MultiAgentIntersectionEnv


def patch_intersection_env():


    original_agent_rewards = MultiAgentIntersectionEnv._agent_rewards

    def patched_agent_rewards(self, vehicle):
        if "reward_speed_range" not in self.config:
            print("Adding missing reward_speed_range parameter to environment config")
            self.config["reward_speed_range"] = [7.0, 9.0]
        return original_agent_rewards(self, vehicle)

    MultiAgentIntersectionEnv._agent_rewards = patched_agent_rewards
    print("Successfully patched MultiAgentIntersectionEnv._agent_rewards method")


def register_intersection_env():
    if "RELintersection-v0" not in registry:
        register(
            id="RELintersection-v0",
            entry_point="highwayenv.intersection_class:MultiAgentIntersectionEnv",
        )


def register_roundabout_env():
    if "RELroundabout-v0" not in registry:
        register(
            id="RELroundabout-v0",
            entry_point="highwayenv.roundabout_class:MultiAgentRoundaboutEnv",
        )


def register_double_intersection_env():
    if "RELdouble-intersection-v0" not in registry:
        register(
            id="RELdouble-intersection-v0",
            entry_point="highwayenv.double_intersection_class:MultiAgentDoubleIntersectionEnv",
        )


def register_chain_intersection_env():
    if "RELchain-intersection-v0" not in registry:
        register(
            id="RELchain-intersection-v0",
            entry_point="highwayenv.chain_intersection_class:MultiAgentChainIntersectionEnv",
        )
