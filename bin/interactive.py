#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from mpe.environment import MultiAgentEnv
from mpe.policy import InteractivePolicy

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


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario_module = scenarios[args.scenario]
    scenario = scenario_module()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
