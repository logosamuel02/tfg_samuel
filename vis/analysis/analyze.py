import argparse
from export import Config
from visualization import (
    anova,
    algorithm,
    data_split,
    messages,
    convergence,
    inference,
    network,
)


def analyze(config: Config) -> None:
    anova.generate(config=config, action="generate")
    # algorithm.generate(config=config, action="download")
    # data_split.generate(config=config, action="download")
    # messages.generate(config=config, action="download")
    # convergence.generate(config=config, action="download")
    # inference.generate(config=config, action="download")
    # network.generate(config=config, action="download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate figures and download into .svg filetype"
    )
    parser.add_argument(
        "-s",
        "--source_path",
        help="Path to the experiments folders to use the in multi-source data figures",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        nargs="?",
        default="images",
        help="Where to store the downloaded images. Default: images",
    )
    parser.add_argument(
        "-p",
        "--plots",
        nargs="?",
        default="config_files/config.yml",
        help="Path to file with the configuration for figures",
    )

    args = vars(parser.parse_args())
    config = Config(**args)
    analyze(config)
