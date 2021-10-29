# -*- coding: utf-8 -*-
import click
import logging
from src.data.ProcessData import ProcessData
import datetime


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    process_data = ProcessData(
        # ["cargo", "tanker"],
        ["fishing"],
        # ["fishing", "cargo", "tanker"],
        "All",
        datetime.datetime.strptime("2019-04-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.datetime.strptime("2020-03-31 23:59:59", "%Y-%m-%d %H:%M:%S"),
        # datetime.datetime.strptime("2019-06-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        # datetime.datetime.strptime("2019-06-30 23:59:59", "%Y-%m-%d %H:%M:%S"),
    )
    result_file_name = process_data.process_into_trajectories(
        600, 60 * 5259487, 60, 1800
    )
    print(result_file_name)
    # result_file_name = "RegionAll_01042019_31032020_Fish_600_2678400_60"
    # result_file_name = "RegionAll_01042019_31032020_CargTank_600_2678400_60"
    summary_file_name = process_data.process_trajectories_for_summary_analysis(
        result_file_name
    )
    print(summary_file_name)
    lat_lon_file_name = process_data.process_trajectories_geographic_points(
        result_file_name
    )
    print(lat_lon_file_name)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
