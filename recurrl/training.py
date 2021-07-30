"""Training main file."""

import random

import numpy as np
import ray
from ray import tune
from ray.tune.logger import UnifiedLogger
import tensorflow as tf
import yaml

from .envs import EscapeRoom, NonMarkovEnvs


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
        self.env_params = params["environment"]
        self.alg_params = params["algorithm"]
        self.logs_dir = params["logs-dir"]
        self.models_dir = params["model-dir"]

        # Trainer config
        self.agent_type: str = self.alg_params["params"]["agent"]
        self.agent_conf: dict = self.alg_params["params"]["config"]

        # Add choices to the configuration (see with_grid_search docstring)
        with_grid_search(self.agent_conf, self.alg_params["params"]["tune"])

        # Env configs
        if self.env_params["name"] == "nonmarkov-envs":
            self.agent_conf["env"] = NonMarkovEnvs
            self.agent_conf["env_config"] = dict(
                name=self.env_params["params"]["name"],
                params=dict(
                    spec=self.env_params["params"]["spec"],
                    rdp=self.env_params["params"]["rdp"],
                ),
            )
        elif self.env_params["name"] == "escape-room1":
            self.agent_conf["env"] = EscapeRoom
            self.agent_conf["env_config"] = self.env_params["params"]
        else:
            raise ValueError(f"{self.env_params['name']} is not an environment")

        # Set seed
        seed = params["seed"]
        random.seed(seed)
        np.random.seed(seed)  # type: ignore
        tf.random.set_seed(seed)
        self.agent_conf["seed"] = seed
        self.agent_conf["env_config"]["seed"] = seed

        # Init library
        ray.init()

    def train(self):
        """Start training."""
        # Start via Tune
        tune.run(
            self.agent_type,
            config=self.agent_conf,
            local_dir=self.logs_dir,
            loggers=[UnifiedLogger],
            name="experiment",
            **self.alg_params["params"]["run"],
        )


def with_grid_search(conf, tune_conf):
    """Compose the parameters to tune in the main configuration.

    conf is any configuration of an agent (a dictionary). tune_conf is another
    configuration in which values are lists instead of single elements. Any
    list define a sequence of experiments, one for each value in the list.

    NOTE: At least one list in tune_conf is always needed (even if of size 1)
    otherwise tune library won't execute any experiment. I don't know why.
    """
    # Base case
    if not isinstance(conf, dict) or not isinstance(tune_conf, dict):
        return

    # Scan conf
    for key in conf:
        if key in tune_conf:

            # Iterate
            with_grid_search(conf[key], tune_conf[key])

            # Transform to search space
            vals = tune_conf[key]
            assert isinstance(vals, list), "'tune_conf' should contain lists"
            conf[key] = tune.grid_search(vals)
