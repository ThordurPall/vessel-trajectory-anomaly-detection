# -*- coding: utf-8 -*-
import click
from pathlib import Path
import logging
from src.visualisation.VisualiseTrajectories import VisualiseTrajectories
import datetime
import configparser


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    src_dir = Path(__file__).resolve().parents[1]
    secret_config = configparser.ConfigParser()
    secret_config.read(src_dir / "secret_config.ini")

    # Initialize the visualise trajectories class
    region = "Denmark"
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(region, True, False, fig_size)

    logger.info("Creating a static Google Maps map of the ROI")
    visualise_trajectories.create_static_map(secret_config["GoogleMaps"]["api_key"])


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
