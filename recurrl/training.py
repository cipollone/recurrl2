"""Training main file."""

import yaml

import ray
from ray import tune
from ray.rllib.agents.impala import ImpalaTrainer

from .envs import NonMarkovEnvs


class Trainer:
    """Training class."""

    def __init__(self, params_file: str):
        """Initialize.

        :param params_file: path to a file of configurations.
            See run-options.yaml format of cipollone/rl-experiments.
        """

        # Read parameters
        with open(params_file) as f:
            params = yaml.safe_load(f)
        env_params = params["environment"]
        alg_params = params["algorithm"]
        self.logs_dir = params["logs-dir"]
        self.models_dir = params["model-dir"]

        # Trainer config
        self.agent_type: str = alg_params["params"]["agent"]
        self.agent_conf: dict = alg_params["params"]["config"]

        # Env config
        self.agent_conf["env"] = NonMarkovEnvs
        self.agent_conf["env_config"] = dict(
            name=env_params["name"],
            params=dict(
                spec=env_params["params"]["spec"],
                rdp=env_params["params"]["rdp"],
            ),
            seed=20358179,  # TODO
        )

        # Init library
        ray.init()

        # Set seed
        # TODO

    def train(self):
        """Start training."""
        # Start via Tune
        tune.run(
            self.agent_type,
            config=self.agent_conf,
            local_dir=self.logs_dir,
        )