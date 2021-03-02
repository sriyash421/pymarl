import os
import multiagent

def make_env(scenario_name, benchmark=False):    
    from multiagent.environment import MultiAgentEnv as _MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    path = os.path.abspath(_MultiAgentEnv.__file__)
    scenario = scenarios.load(path+"/scenarios/"+scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = _MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = _MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)
    return env

class MultiAgentEnv(object):
    
    def __init__(self, scenario_name):
        super().__init__()
        self.env = make_env(scenario_name)
        self.n_agents = self.env.n
        self.episode_limit = self.env.episode_limit

    def step(self, actions):
        """ Returns reward, terminated, info """
        return self.env.step(actions)

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs = [self.env._get_obs(self.env.world.agents[i]) for i in range(self.env.n)]
        return obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.env._get_obs(self.env.world.agents[agent_id])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env.observation_space.shape

    def get_state(self):
        return self.get_obs()

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size()

    def get_avail_actions(self):
        return self.env.action_space.shape

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        self.env.reset()

    def render(self):
        self.env.render(close=False)

    def close(self):
        self.env.render(close=True)

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
