import argparse
from pydantic import FilePath
from config import Config
import anova
import algorithm
import data_split
import messages
import convergence
import inference
import network


def analyze(input: FilePath, output: FilePath, config_file: FilePath) -> None:
    config = Config(source_path=input, output_path=output, plots=config_file)
    anova.generate(config=config, action="download")
    algorithm.generate(config=config, action="download")
    data_split.generate(config=config, action="download")
    messages.generate(config=config, action="download")
    convergence.generate(config=config, action="download")
    inference.generate(config=config, action="download")
    network.generate(config=config, action="download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-c", "--config_file")

    args = vars(parser.parse_args())
    analyze(**args)
