"""Definitions of Gym environments."""

from typing import cast
from functools import partial
import importlib

from gym import Env, ObservationWrapper, RewardWrapper, Wrapper
from gym.spaces import Box, Discrete, MultiDiscrete
from nonmarkov_envs.discrete_env import MultiDiscreteEnv
from nonmarkov_envs.rdp_env import RDPEnv as RDPEnv0
from nonmarkov_envs.specs import driving_agent
import numpy as np

EnvSpecT = driving_agent.DrivingAgent  # Or any other env spec


class NonMarkovEnvs(Wrapper):
    """A wrapper class for all NonMarkovian environments."""

    def __init__(self, config: dict):
        """Initialize.

        :param config: a specification dictionary with fields:
            name: the name of a whitemech/nonmarkov-envs environment.
            spec: a rdp specification (see nonmarkov_envs)
            rdp: rdp options (see nonmarkov_envs)
            seed: environment seed.
        :return: a gym environment
        """
        env = _build_nonmarkov_env(config["name"], config["params"], config["seed"])
        super().__init__(env=env)


class _RDPEnv(RDPEnv0):
    """RDPEnv, with the initial observation visible.

    Currently, RDPEnv from nonmarkov_envs hides the initial observation.
    This class disables this feature.
    """
    def reset(self, **kwargs):
        """Reset the environment."""
        # Grandparent, skip super
        assert issubclass(RDPEnv0, MultiDiscreteEnv), (
            "Update the superclass here")
        state = MultiDiscreteEnv.reset(self, **kwargs)
        state = self._process(state)
        self.steps = 0
        return state


def _build_nonmarkov_env(name: str, params: dict, seed: int) -> Env:
    """Instantiate a nonmarkov env to be used in TensorForce.

    :param name: the name of a whitemech/nonmarkov-envs environment
        specification (eg. rotating_maze).
    :param params: a dictionary of additional parameters with format:
        { "spec": dict(), "rdp": dict }
    :param seed: seed to set for the gym environment.
    :return: a TensorForce environment instance.
    """
    # Instantiate env
    try:
        env_module = cast(driving_agent, importlib.import_module(
            f"nonmarkov_envs.specs.{name}"))
    except ImportError:
        raise ValueError("Not a valid environment specification module")

    env_spec = env_module.instantiate_env(params["spec"])
    env = _RDPEnv(env_spec, **params["rdp"])

    # Maybe the agent observe its actions
    if "observe_actions" in params and params["observe_actions"]:
        env = SeeActions(env)

    # Flatten observations (to avoid a probable TensorForce bug)
    env = Flatten(env)

    # Set seed
    env.seed(seed)

    return env


class SeeActions(Wrapper):
    """Let the agent observe its own actions."""

    def __init__(self, env: Env):
        """Initialize.

        :param env: environment to wrap.
        """
        # Super
        super().__init__(env)

        # Check
        assert isinstance(env.observation_space, MultiDiscrete)

        # Add actions (+1 for None)
        n_actions = cast(Discrete, env.action_space).n
        self.observation_space = MultiDiscrete(
            np.append(env.observation_space.nvec, n_actions + 1))
        self._none_action = n_actions

    def reset(self):
        """Gym reset."""
        obs = self.env.reset()
        return self._combine(obs, self._none_action)

    def step(self, action):
        """Gym step."""
        obs, reward, done, info = self.env.step(action)
        obs = self._combine(obs, action)
        return obs, reward, done, info

    def _combine(self, observation, action):
        """Append action to observation."""
        return np.append(observation, action)


class Flatten(ObservationWrapper):
    """Flatten observation vector to scalar."""

    def __init__(self, env: Env):
        """Initialize.

        :param env: environment to wrap.
        """
        # Super
        super().__init__(env)

        # Check
        assert isinstance(env.observation_space, MultiDiscrete)

        # Set
        self._original_obs_space = env.observation_space
        self.encoder = partial(
            np.ravel_multi_index, dims=self._original_obs_space.nvec)
        self.decoder = partial(
            np.unravel_index, shape=self._original_obs_space.nvec)

        self.observation_space = Discrete(np.prod(env.observation_space.nvec))

    def observation(self, observation):
        """Combine an observation by combining indexed to scalar.

        See self.encoder, self.decoder.
        """
        return self.encoder(observation)


class RewardScale(RewardWrapper):
    """Just multiply rewards by a constant."""

    def __init__(self, env: Env, scale: float):
        """Initialize."""
        super().__init__(env)
        self._scale = scale

    def reward(self, reward):
        """Scale reward."""
        return reward * self._scale
