# Code for processing data samples, such that the data set code is decoupled from
# the model training code for better readability and modularity. The data sets defined
# stores the samples and their corresponding labels, and a PyTorch DataLoader can then
# be used to wrap an iterable around the data sets to enable easy access to the samples

import logging
import pickle

import numpy as np
import pandas as pd
import torch

import src.utils.dataset_utils as dataset_utils


class AISDiscreteRepresentation(torch.utils.data.Dataset):
    """A custom Dataset class for processing discrete AIS data samples

    That is, use this class when the AIS attributes should be represented
    as a four hot encoded vector

    ...

    Attributes
    ----------
    train_mean : Tensor (Defaults to None)
        Mean of the times a certain bin was activated. When train_mean is not
        None, either a validation or test Dataset is created

    data_info : dict
        Information about the data

    data_path : str
        Path to the actual data set location

    indicies : list
        Train, val, or test indicies where the tracks begin in the data file

    data_dim : int
        Dimension of the four hot encoded vector

    Methods
    -------
    __len__()
        Returns the number of dataset samples

    __getitem__(idx)
        Gets a trajectory sample from the data set at the given index idx

    compute_mean()
        Computs the mean of how often the different bins are activated
    """

    def __init__(self, data_info_file_path, train_mean=None, validation=False):
        """
        Parameters
        ----------
        data_info_file_path : pathlib.WindowsPath
            Path to where the summary data is located

        train_mean : Tensor (Defaults to None)
            Mean of the times a certain bin was activated. When train_mean is not
            None, either a validation or test Dataset is created

        validation : bool (Defaults to False)
            When validation is True, a validation Dataset is created, but when it
            is False a test set is created (assuming train_mean is None)
        """
        logger = logging.getLogger(__name__)  # For logging information

        # Read the data info pickle file into memory
        logger.info("Processing data from the info file: " + str(data_info_file_path))
        with open(data_info_file_path, "rb") as f:
            self.data_info = pickle.load(f)
        self.data_path = self.data_info["dataFileName"]

        # Get the requested data set (one of train/val/test)
        if train_mean == None:
            self.indicies = self.data_info["trainIndicies"]
        else:
            if validation:
                self.indicies = self.data_info["valIndicies"]
            else:
                self.indicies = self.data_info["testIndicies"]

        # Get the dimension of the four hot encoded vector (#Bins = #edges - 1)
        lat_edges, lon_edges, speed_edges, course_edges = self.data_info["binedges"]
        self.data_dim = (
            len(lat_edges) + len(lon_edges) + len(speed_edges) + len(course_edges) - 4
        )

        # Compute the mean from the training set but otherwise use the training mean
        if train_mean == None:
            logger.info("Computing training mean values using self.compute_mean()")
            self.mean = self.compute_mean()
        else:
            self.mean = train_mean

    def __len__(self):
        """Returns the number of dataset samples

        Returns
        -------
        int
            Data set length
        """
        return len(self.indicies)

    def __getitem__(self, idx):
        """Gets a trajectory sample from the data set at the given index idx

        Parameters
        ----------
        idx : int
            Data set index idx to retrive

        Returns
        -------
        set
            Set of trajectory information as well as inputs and targets
        """
        logger = logging.getLogger(__name__)  # For logging information

        # Load the data file for the requested index
        logger.info("Processing data from the data file: " + str(self.data_path))
        with open(self.data_path, "rb") as file:
            # Read the track from the corresponding point in memory
            file.seek(self.indicies[idx])
            track = pickle.load(file)
        df = pd.DataFrame(track)

        # Four hot encode the current trajectory
        logger.info("Four hot encode trajectory using dataset_utils.FourHotEncode")
        encodedTrack = dataset_utils.FourHotEncode(df, self.data_info["binedges"])

        # Return the shipType as a label that could be useful later (e.g. for plotting)
        logger.info("Convert shipType using dataset_utils.convertShipTypeToLabel")
        ship_type_label = dataset_utils.convertShipTypeToLabel(track["shiptype"])

        # The targets are just to reconstruct the input sequence
        targets = torch.tensor(encodedTrack, dtype=torch.float)  # seq_len X data_dim
        inputs = targets - self.mean  # Center the inputs

        # Return the trajectory with other (potentially) useful information
        return (
            torch.tensor(track["mmsi"]),
            np.array(track["timestamp"]),
            torch.tensor(ship_type_label),
            torch.tensor(track["track_length"], dtype=torch.int),
            inputs,
            targets,
        )

    def compute_mean(self):
        """Computs the mean of how often the different bins are activated

        Returns
        -------
        Tensor
            The mean values of how often the different bins are activated
        """
        sum_all = np.zeros((self.data_dim))
        total_updates = 0

        # Loop over evry observation, read into memory, do four hot encoding and sum up all encodings
        for index in self.indicies:
            with open(self.data_path, "rb") as file:
                file.seek(index)
                track = pickle.load(file)
            df = pd.DataFrame(track)

            encodedTrack = dataset_utils.FourHotEncode(
                df, self.data_info["binedges"]
            )  # seq_len X data_dim
            sum_all += np.sum(encodedTrack, axis=0)  # Sum over all time points
            total_updates += track["track_length"]

        # Mean of all the times a certain bin was activated. Used as a normalization factor (Centering in getimtem)
        return torch.tensor(sum_all / total_updates, dtype=torch.float)
