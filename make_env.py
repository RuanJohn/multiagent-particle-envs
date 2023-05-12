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

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from mpe.environment import MultiAgentEnv

    print(f"SCENARIO NAME: {scenario_name}")
    # load scenario from script
    scenario_module = scenarios[scenario_name]

    print(f"SCENARIO MODULE: {scenario_module}")

    scenario = scenario_module()

    print(f"SCENARIO: {scenario}")

    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
