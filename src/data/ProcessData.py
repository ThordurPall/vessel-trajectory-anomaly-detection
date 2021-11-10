import datetime
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import progressbar

import src.utils.createDataset as createDataset
from src.Config import Config, ROISCOGConfig


class ProcessData:
    """
    A class used to handle everything related to processing the data
    for exploratory analysis and modelling

    ...

    Attributes
    ----------
    raw_data_dir : pathlib.WindowsPath
        Directory where the raw (unprocessed) data is located

    processed_data_dir : pathlib.WindowsPath
        Directory where the processed data is / should be located

    ship_types_included : list
        List of ship type strings to include in the data set

    region : str
        One of the specified regions in the config file (like "Bornholm")

    start_time : datetime
        Start date (and time) of the data to process

    end_time : datetime
        End date (and time) of the data to process

    Methods
    -------
    process_into_trajectories(min_track_length, max_track_Length,
                              resample_frequency, split_track_length)
        Process raw data into trajectories

    process_trajectories_for_summary_analysis(file_name)
        Process trajectories into data for exploratory analysis

    process_trajectories_geographic_points(file_name)
        Process trajectories into geographic latitude and longitude points

    ship_type_to_file_name(ship_type)
        Defines a mapping between ship type and the start of file name
    """

    def __init__(self, ship_types_included, region, start_time, end_time):
        """
        Parameters
        ----------
        ship_types_included : list
            List of ship type strings to include in the data set

        region : str
            One of the predefined regions in the config file (like "Bornholm")

        start_time : datetime
            Start date (and time) of the data to process

        end_time : datetime
            End date (and time) of the data to proces
        """

        super().__init__()
        self.ship_types_included = ship_types_included
        self.region = region
        self.start_time = start_time
        self.end_time = end_time

        # Setup the correct foldure structure
        project_dir = Path(__file__).resolve().parents[2]
        self.raw_data_dir = project_dir / "data" / "raw"
        self.processed_data_dir = project_dir / "data" / "processed"

        # Make sure that the processed path exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def process_into_trajectories(
        self,
        min_track_length,
        max_track_Length,
        resample_frequency,
        split_track_length,
        train_proportion=0.8,
        test_proportion=0.1,
        inject_cargo_proportion=0.0,
    ):
        """Process raw data into trajectories

        Parameters
        ----------
        min_track_length : int
            Minimum track length in seconds. There is a lower bound for the track lenght
            since it needs to be long enought to enable learn something useful

        max_track_Length : int
            Maximum track length in seconds

        resample_frequency : int
            Time between samples in seconds. Together max_track_Length and resample_frequency
            determine the max number of AIS attribute updates (max_track_Length/resample_frequency).
            Overly long sequence (too many updates) can hinder learning

        split_track_length : int
            Split tracks into two trajectories when the time difference between AIS update messages
            is greater than split_track_length seconds. So, then the trajectory has ended and new one started

        train_proportion : float (Defaults to 0.8)
            Proportion (0.0 <= train_proportion <= 1.0) of data to use for training

        test_proportion : float (Defaults to 0.1)
            Proportion (0.0 <= test_proportion <= 1.0) of data to use for testing.
            The rest (1.0 - train_proportion - test_proportion) is used for validation

        inject_cargo_proportion : float (Defaults to 0.0)
            Inject cargo vessel MMSIS in inject_cargo_proportion proportion to the training MMSIs

        Returns
        -------
        str
            Returns the main part of the file name where the results are stored

        """
        logger = logging.getLogger(__name__)  # For logging information

        # Map the requested ship types to how they are written in the JSON data set file names
        ship_type_files = [
            self.ship_type_to_file_name(ship_type)
            for ship_type in self.ship_types_included
        ]

        # Define the file name to save the results
        cargo_injected = (
            "_Injected" + str(inject_cargo_proportion).replace(".", "")
            if inject_cargo_proportion != 0.0
            else ""
        )
        result_file_name = (
            "Region"
            + self.region
            + "_"
            + self.start_time.strftime("%d%m%Y")
            + "_"
            + self.end_time.strftime("%d%m%Y")
            + "_"
            + "".join(ship_type_files)
            + "_"
            + str(min_track_length)
            + "_"
            + str(max_track_Length)
            + "_"
            + str(resample_frequency)
            + cargo_injected
        )
        logger.info("Main part of data results file name: " + result_file_name)

        # Setup the parameter dictionary that the createDatset util function expects
        ROISCOG_config = ROISCOGConfig(self.region)  # ROI, SOG, and COG configurations
        config = Config()  # General project configuration
        params = {
            "ROI": ROISCOG_config.roi,
            "timeperiod": (self.start_time, self.end_time),
            "maxspeed": ROISCOG_config.SOG_MAX,
            "navstatuses": config.get_property("MOV_NAV_STATUSES"),
            "shiptypes": ship_type_files,
            "binedges": ROISCOG_config.binedges,
            "minTrackLength": min_track_length,
            "maxTrackLength": max_track_Length,
            "resampleFrequency": resample_frequency,
            "splitTrackTimeDiff": split_track_length,
        }
        logger.info("Calling createDataset.createDataset to process trajectories")
        trajectories = createDataset.createDataset(
            params,
            str(self.raw_data_dir),
            str(self.processed_data_dir),
            result_file_name,
            inject_cargo_proportion,
        )

        # With the trajectories, now randomly make the train, validation and test set splits
        n = int(len(trajectories["indicies"]) * (1.0 - train_proportion))
        val_test_indices = np.random.choice(
            trajectories["indicies"], size=n, replace=False
        )
        trajectories["trainIndicies"] = [
            index for index in trajectories["indicies"] if index not in val_test_indices
        ]

        # Split the rest of the data set into validation and test sets
        n = int(len(trajectories["indicies"]) * test_proportion)
        test_indices = list(np.random.choice(val_test_indices, size=n, replace=False))
        trajectories["testIndicies"] = test_indices
        trajectories["valIndicies"] = [
            index for index in val_test_indices if index not in test_indices
        ]

        # Write the data set information to a file
        with open(
            str(self.processed_data_dir) + "/datasetInfo_" + result_file_name + ".pkl",
            "wb",
        ) as file:
            pickle.dump(trajectories, file)
        return result_file_name

    def process_trajectories_for_summary_analysis(self, file_name):
        """Process trajectories into data for exploratory analysis

        Parameters
        ----------
        file_name : str
            Name of the main part of the file where the results are saved

        Returns
        -------
        str
            Returns the file path to the processed summary results
        """
        logger = logging.getLogger(__name__)  # For logging information
        data_file = self.processed_data_dir / ("data_" + file_name + ".pkl")
        data_info_file = self.processed_data_dir / ("datasetInfo_" + file_name + ".pkl")

        # Read the info file to know how to read the data file
        logger.info(
            "Processing data using the following info file: " + str(data_info_file)
        )
        with open(data_info_file, "rb") as f:
            data_info = pickle.load(f)

        logger.info("Processing the data file for analysis, one index at a time")
        trajectories = []  # Keep track of all trajectories and its relevant information
        for index in progressbar.progressbar(data_info["indicies"]):
            # Get the current track - Read the data file from the current index
            with open(data_file, "rb") as f:
                f.seek(index)
                track = pickle.load(f)
                track_df = pd.DataFrame(track)

            # Store information about this trajectory
            trajectories.append(
                [
                    track_df["mmsi"][0],
                    track_df["shiptype"][0],
                    track_df["track_length"][0],
                    track_df["speed"].mean(),
                    track_df["timestamp"].mean(),
                    track_df["timestamp"][0],
                    track_df["timestamp"].iloc[-1],
                    track_df["lat"][0],
                    track_df["lat"].iloc[-1],
                    track_df["lon"][0],
                    track_df["lon"].iloc[-1],
                ]
            )

        df = pd.DataFrame(
            trajectories,
            columns=[
                "MMSI",
                "ShipType",
                "TrackLength",
                "MeanSpeed",
                "DateTime",
                "DateTimeStart",
                "DateTimeEnd",
                "StartLat",
                "EndLat",
                "StartLon",
                "EndLon",
            ],
        )
        summary_file_name = (
            str(self.processed_data_dir) + "/" + file_name + "_trajectories_summary.csv"
        )

        logger.info(
            "Writing the data frame to a comma-separated values (csv) file: "
            + summary_file_name
        )
        df.to_csv(summary_file_name, index=False)
        return summary_file_name

    def process_trajectories_geographic_points(self, file_name):
        """Process trajectories into geographic latitude and longitude points

        Goes through each trajectory and stores all the latitude and
        longitude AIS point updates in a pickle file for analysis

        Parameters
        ----------
        file_name : str
            Name of the main part of the file where the results are saved

        Returns
        -------
        str
            Returns the file path to the processed latitude and longitude point results
        """
        logger = logging.getLogger(__name__)  # For logging information
        data_file = self.processed_data_dir / ("data_" + file_name + ".pkl")
        data_info_file = self.processed_data_dir / ("datasetInfo_" + file_name + ".pkl")

        # Read the info file to know how to read the data file
        logger.info(
            "Processing data using the following info file: " + str(data_info_file)
        )
        with open(data_info_file, "rb") as f:
            data_info = pickle.load(f)

        logger.info(
            "Processing the data file for geographic point-based analysis, one index at a time"
        )
        # Keep track of the lat/lon for all AIS updates
        lats = np.array([])
        lons = np.array([])

        for index in progressbar.progressbar(data_info["indicies"]):
            # Get the current track - Read the data file from the current index
            with open(data_file, "rb") as f:
                f.seek(index)
                track = pickle.load(f)
                track_df = pd.DataFrame(track)

            # Keep track of the lat/lon for all AIS updates
            lats = np.append(lats, track_df["lat"])
            lons = np.append(lons, track_df["lon"])
        summary_file_name = self.processed_data_dir / (file_name + "_lats_lons.pickle")

        logger.info(
            "Writing the latitude and longitude values to a pickle file: "
            + str(summary_file_name)
        )
        with open(summary_file_name, "wb") as f:
            # Pickle the data using the highest protocol available
            pickle.dump(
                {"Latitudes": lats, "Longitudes": lons}, f, pickle.HIGHEST_PROTOCOL
            )
        return summary_file_name

    def ship_type_to_file_name(self, ship_type):
        """Defines a mapping between ship type and the start of file name

            Defaults to None if no mapping was successful

        Parameters
        ----------
        ship_type : str
            Ship type string to map to file name
        """
        config = Config()
        return {
            "fishing": config.get_property("SHIPTYPE_FISHING"),
            "towing": config.get_property("SHIPTYPE_TOWING"),
            "dredging": config.get_property("SHIPTYPE_DREDGING"),
            "diving": config.get_property("SHIPTYPE_DIVING"),
            "military": config.get_property("SHIPTYPE_MILITARY"),
            "sailing": config.get_property("SHIPTYPE_SAILING"),
            "pleasure": config.get_property("SHIPTYPE_PLEASURE"),
            "passenger": config.get_property("SHIPTYPE_PASSENGER"),
            "cargo": config.get_property("SHIPTYPE_CARGO"),
            "tanker": config.get_property("SHIPTYPE_TANKER"),
        }.get(
            ship_type
        )  # Return None if not found in the dictionary
