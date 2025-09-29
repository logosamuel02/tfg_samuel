import argparse
from pydantic import FilePath
from vis.analysis.export import Config
import visualization as vis


def analyze(input: FilePath, output: FilePath, config_file: FilePath) -> None:
    config = Config(source_path=input, output_path=output, plots=config_file)
    vis.anova.generate(config=config, action="download")
    vis.algorithm.generate(config=config, action="download")
    vis.data_split.generate(config=config, action="download")
    vis.messages.generate(config=config, action="download")
    vis.convergence.generate(config=config, action="download")
    vis.inference.generate(config=config, action="download")
    vis.network.generate(config=config, action="download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-c", "--config_file")

    args = vars(parser.parse_args())
    analyze(**args)
