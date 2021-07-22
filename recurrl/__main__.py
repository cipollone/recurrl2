"Main script: entry point"

import argparse

from . import training


def main():
    """Main function."""

    # Args
    parser = argparse.ArgumentParser(
        description="Experiments with Recurrent models in RL"
    )
    subparsers = parser.add_subparsers(dest="op", help="What to do")

    train_parser = subparsers.add_parser("train", help="Train the net")
    train_parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to a config file. "
        "See run-options.yaml format of cipollone/rl-experiments.",
    )

    # TODO: maybe also evaluation command

    args = parser.parse_args()

    # Do
    if args.op == "train":
        training.Trainer(args.params).train()


if __name__ == "__main__":
    main()
