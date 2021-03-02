from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from wrapper import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

REGISTRY["pymarl_simple"] = partial(env_fn, env=MultiAgentEnv, scenario="simple")
REGISTRY["pymarl_simple_tag"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_tag")
REGISTRY["pymarl_simple_adversary"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_adversary")
REGISTRY["pymarl_simple_reference"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_reference")
REGISTRY["pymarl_simple_world_comm"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_world_comm")
REGISTRY["pymarl_simple_crypto"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_crypto")
REGISTRY["pymarl_simple_speaker_listener"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_speaker_listener")
REGISTRY["pymarl_warehouse"] = partial(env_fn, env=MultiAgentEnv, scenario="warehouse")
REGISTRY["pymarl_simple_push"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_push")
REGISTRY["pymarl_simple_spread"] = partial(env_fn, env=MultiAgentEnv, scenario="simple_spread")