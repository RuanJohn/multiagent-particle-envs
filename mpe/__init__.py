from gym.envs.registration import register

# Import all possible environments

from mpe.scenarios.climbing_spread import Scenario as climbing_spread_scenario
from mpe.scenarios.multi_speaker_listener import Scenario as multi_speaker_listener_scenario
from mpe.scenarios.simple_adversary import Scenario as simple_adversary_scenario
from mpe.scenarios.simple_crypto import Scenario as simple_crypto_scenario
from mpe.scenarios.simple_doublespread import Scenario as simple_doublespread_scenario
from mpe.scenarios.simple_push import Scenario as simple_push_scenario
from mpe.scenarios.simple_reference import Scenario as simple_reference_scenario
from mpe.scenarios.simple_speaker_listener import Scenario as simple_speaker_listener_scenario
from mpe.scenarios.simple_spread import Scenario as simple_spread_scenario
from mpe.scenarios.simple_tag import Scenario as simple_tag_scenario
from mpe.scenarios.simple_world_comm import Scenario as simple_world_comm_scenario
from mpe.scenarios.simple import Scenario as simple_scenario
from mpe.scenarios.sparse_predator_prey import Scenario as sparse_predator_prey_scenario

scenarios = {
    'climbing_spread': climbing_spread_scenario,
    'multi_speaker_listener': multi_speaker_listener_scenario,
    'simple_adversary': simple_adversary_scenario,
    'simple_crypto': simple_crypto_scenario,
    'simple_doublespread': simple_doublespread_scenario,
    'simple_push': simple_push_scenario,
    'simple_reference': simple_reference_scenario,
    'simple_speaker_listener': simple_speaker_listener_scenario,
    'simple_spread': simple_spread_scenario,
    'simple_tag': simple_tag_scenario,
    'simple_world_comm': simple_world_comm_scenario,
    'simple': simple_scenario,
    'sparse_predator_prey': sparse_predator_prey_scenario,
}

# Multiagent envs
# ----------------------------------------

_particles = {
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
    "climbing_spread": "ClimbingSpread-v0",
}

for scenario_name, gymkey in _particles.items():
    scenario_module = scenarios[scenario_name]
    scenario = scenario_module()
    world = scenario.make_world()

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )

# Registers the custom double spread environment:

for N in range(2, 11, 2):
    scenario_name = "simple_doublespread"
    gymkey = f"DoubleSpread-{N}ag-v0"
    scenario_module = scenarios[scenario_name]
    scenario = scenario_module()
    world = scenario.make_world(N)

    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )