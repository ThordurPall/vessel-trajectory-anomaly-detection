# -*- coding: utf-8 -*-
import logging

import click

from src.models.TrainEvaluate import TrainEvaluate


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():
    file_name = "RegionBornholm_01062019_30092019_FishCargTank_14400_86400_600"
    fishing_file = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    fishing_new_file = "RegionBornholm_01052019_31052019_Fish_14400_86400_600"
    # print(
    #    "Train Default FishCargTank: --------------------------------------------------------"
    # )
    # train1 = TrainEvaluate(
    #    file_name, fishing_file=fishing_file, fishing_new_file=fishing_new_file
    # )
    # train1.train_VRNN()

    # file_name = "RegionBornholm_01062019_30092019_CargTank_14400_86400_600"
    # print(
    #    "Train Default CargTank: --------------------------------------------------------"
    # )
    # train2 = TrainEvaluate(
    #    file_name, fishing_file=fishing_file, fishing_new_file=fishing_new_file
    # )
    # train2.train_VRNN()

    # file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    # print(
    #    "Train Default Fish: --------------------------------------------------------"
    # )
    # train3 = TrainEvaluate(
    #    file_name, fishing_file=fishing_file, fishing_new_file=fishing_new_file
    # )
    # train3.train_VRNN()

    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600_Injected"
    print(
        "Train Default Fish with Cargo : --------------------------------------------------------"
    )
    train4 = TrainEvaluate(
        file_name, fishing_file=fishing_file, fishing_new_file=fishing_new_file
    )
    train4.train_VRNN()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
