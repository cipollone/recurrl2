"""Training main file."""

import yaml


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

        # Make environment spec
        # TODO

        # Set seed
        # TODO

        # Trainer settings
        # TODO

    def train(self):
        """Start training."""
        # TODO
