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
        # ["fishing"],
        ["fishing", "cargo", "tanker"],
        "Bornholm",
        datetime.datetime.strptime("2019-06-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        datetime.datetime.strptime("2019-09-30 23:59:59", "%Y-%m-%d %H:%M:%S"),
        # datetime.datetime.strptime("2019-04-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        # datetime.datetime.strptime("2020-03-31 23:59:59", "%Y-%m-%d %H:%M:%S"),
        # datetime.datetime.strptime("2019-06-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        # datetime.datetime.strptime("2019-06-30 23:59:59", "%Y-%m-%d %H:%M:%S"),
    )
    # Note that all times are in seconds
    # Old settings: 600, 60 * 5259487, 60, 1800
    # "The detection starts if the track is long enough to be meaningful, here greater or equal to 4 hours."
    # "Discontiguous voyages (voyages that have the maximum interval between two successive AIS messages
    # longer than a threshold, here 2 hours) were split into contiguous ones. We re-sampled all voyages
    # to a resolution of 10 minutes (i.e., {t + 1} − {t} = 10 mins) using a linear interpolation.
    # Very long voyages were split into smaller tracks from 4 to 24 hours each."
    min_track_length = 240 * 60  # 240 minutes (4 hours)
    max_track_Length = 24 * 60 * 60  # 24 hours
    resample_frequency = 10 * 60  # 10 mintues
    split_track_length = 120 * 60  # 120 minutes (2 hours)

    result_file_name = process_data.process_into_trajectories(
        min_track_length, max_track_Length, resample_frequency, split_track_length
    )
    print(result_file_name)
    # result_file_name = "RegionAll_01042019_31032020_Fish_600_2678400_60"
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
