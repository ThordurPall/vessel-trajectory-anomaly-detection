# Code for processing data samples, such that the data set code is decoupled from
# the model training code for better readability and modularity. The data sets defined
# stores the samples and their corresponding labels, and a PyTorch DataLoader can then
# be used to wrap an iterable around the data sets to enable easy access to the samples

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.utils.dataset_utils as dataset_utils


class AISDataset(torch.utils.data.Dataset):
    """A custom Dataset class for processing discrete and continuous AIS data samples

    That is, use this class when the AIS attributes should be represented
    as either a four hot encoded vector or with the original input dimensions

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

    discrete : bool
        When True, discrete four hot encoded inputs will be used.
        When False the actual real valued inputs will be used

    Methods
    -------
    __len__()
        Returns the number of dataset samples

    __getitem__(idx)
        Gets a trajectory sample from the data set at the given index idx

    compute_mean()
        Computs the mean of how often the different bins are activated
    """

    def __init__(
        self,
        file_name,
        train_mean=None,
        train_std=None,
        validation=False,
        data_info=None,
        indicies=None,
        discrete=False,
        first_order_diff = True,
    ):
        """
        Parameters
        ----------
        file_name : str
            Name of the main part of the file where the results are saved

        train_mean : Tensor (Defaults to None)
            Mean of the times a certain bin was activated. When train_mean is not
            None, either a validation or test Dataset is created

        train_mean : Tensor (Defaults to None)
            Input dimension standard deviation (used for continuous inputs)

        validation : bool (Defaults to False)
            When validation is True, a validation Dataset is created, but when it
            is False a test set is created (assuming train_mean is None)

        data_info : dict (Defaults to None)
            Information about the data. When None, it is read from the data_info_file

        indicies : list (Defaults to None)
            Train, val, or test indicies where the tracks begin in the data file.
            When None, the indicies are read from the requested data set

        discrete : bool (Defaults to True)
            When True, the AIS attributes should be represented as a four hot encoded vector.
            When False, the original input dimensions should be used

        """
        logger = logging.getLogger(__name__)  # For logging information
        self.discrete = discrete
        self.first_order_diff = first_order_diff

        # Setup the correct foldure structure
        project_dir = Path(__file__).resolve().parents[2]
        processed_data_dir = project_dir / "data" / "processed"

        # Read the data info pickle file into memory
        if data_info is None:
            data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
            logger.info("Processing data from the info file: " + str(data_info_file))
            with open(data_info_file, "rb") as f:
                self.data_info = pickle.load(f)
        else:
            self.data_info = data_info
        # self.data_path = self.data_info["dataFileName"]
        self.data_path = str(processed_data_dir / ("data_" + file_name + ".pkl"))

        # Get the requested data set (one of train/val/test)
        if indicies is None:
            if train_mean == None:
                self.indicies = self.data_info["trainIndicies"]
            else:
                if validation:
                    self.indicies = self.data_info["valIndicies"]
                else:
                    self.indicies = self.data_info["testIndicies"]
        else:
            self.indicies = indicies

        # Get the dimension of the four hot encoded vector (#Bins = #edges - 1)
        lat_edges, lon_edges, speed_edges, course_edges = self.data_info["binedges"]
        if self.discrete:
            self.data_dim = (
                len(lat_edges)
                + len(lon_edges)
                + len(speed_edges)
                + len(course_edges)
                - 4
            )
        else:
            if self.first_order_diff:
                self.data_dim = 4
            else:
                self.data_dim = 4

        # Get the ship types, lengths (andmax length) and define a temporal mask
        self.ship_types, self.lengths = self.get_labels()
        self.max_length = torch.max(self.lengths)
        self.temporal_mask = (
            torch.arange(self.max_length, device="cpu")[:, None] < self.lengths[None, :]
        )  # Dimensions: max_seq_len X len(data_set)

        # Compute the mean and standard deviation (when appropriate) from the training set
        # or otherwise use the once computed from the training set
        if train_mean == None:
            logger.info(
                "AISDataset: Computing training mean values using self.compute_mean()"
            )
            self.mean = self.compute_mean()

            if not self.discrete:
                logger.info(
                    "AISDataset: Computing training standard deviation values using self.compute_std()"
                )
                self.std = self.compute_std()
        else:
            self.mean = train_mean
            self.std = train_std

        print(self.mean)
        print(self.std)

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
        tuple
            Set of trajectory information as well as inputs and targets
        """

        # Load the data file for the requested index
        with open(self.data_path, "rb") as file:
            # Read the track from the corresponding point in memory
            file.seek(self.indicies[idx])
            track = pickle.load(file)
        df = pd.DataFrame(track)

        # Return the shipType as a label that could be useful later (e.g. for plotting)
        ship_type_label = dataset_utils.convertShipTypeToLabel(track["shiptype"])

        if self.discrete:
            # Four hot encode the current trajectory
            encodedTrack = dataset_utils.FourHotEncode(df, self.data_info["binedges"])

            # The targets are just to reconstruct the input sequence
            targets = torch.tensor(
                encodedTrack, dtype=torch.float
            )  # seq_len X data_dim
            inputs = targets - self.mean  # Center the inputs
        elif self.first_order_diff:
            #targets = torch.tensor(
            #    np.array(df[["lat", "lon", "speed", "course"]].values.tolist()),
            #    dtype=torch.float,
            #)
            #targets = torch.diff(targets, dim=0)
            positions = torch.tensor(
                np.array(df[["lat", "lon"]].values.tolist()),
                dtype=torch.float,
            )
            kinematics = torch.tensor(
                np.array(df[["speed", "course"]].values.tolist()),
                dtype=torch.float,
            )
            targets = torch.cat([torch.diff(positions, dim=0),kinematics[1:]],dim=-1)
            inputs = (targets - self.mean) / self.std
        else:
            # The original input dimensions should be used
            targets = torch.tensor(
                np.array(df[["lat", "lon", "speed", "course"]].values.tolist()),
                dtype=torch.float,
            )
            inputs = (targets - self.mean) / self.std  # Standardize the input data

        # Return the trajectory with other (potentially) useful information
        return (
            torch.tensor(idx),
            torch.tensor(self.indicies[idx]),
            torch.tensor(track["mmsi"]),
            np.array(track["timestamp"]),
            torch.tensor(ship_type_label),
            torch.tensor(track["track_length"]-1., dtype=torch.int),
            inputs,
            targets,
        )

    def compute_mean(self):
        """Computs the input space mean values

        Returns
        -------
        Tensor
            The input space mean values
        """
        sum_all = np.zeros((self.data_dim))
        total_updates = 0

        # Loop over every observation, read into memory, and compute the sum over each dimension
        for index in self.indicies:
            with open(self.data_path, "rb") as file:
                file.seek(index)
                track = pickle.load(file)
            df = pd.DataFrame(track)

            if self.discrete:
                inputs = dataset_utils.FourHotEncode(
                    df, self.data_info["binedges"]
                )  # seq_len X data_dim
            elif self.first_order_diff:
                #inputs = np.array(df[["lat", "lon", "speed", "course"]].values.tolist())
                #inputs = np.diff(positions, axis=0)
                positions = np.array(df[["lat", "lon"]].values.tolist())
                kinematics = np.array(df[["speed", "course"]].values.tolist())
                inputs = np.concatenate([np.diff(positions, axis=0),kinematics[1:]],axis=-1)
            else:
                inputs = np.array(df[["lat", "lon", "speed", "course"]].values.tolist())
            sum_all += np.sum(inputs, axis=0)  # Sum over all time points
            total_updates += (track["track_length"]-1)
        self.total_training_updates = total_updates

        # Mean of all the times a certain bin was activated. Used as a normalization factor (Centering in getimtem)
        return torch.tensor(sum_all / total_updates, dtype=torch.float)

    def compute_std(self):
        """Computs the input standard deviation
            Note: This assumes a continuous input space

        Returns
        -------
        Tensor
            The input dimensions standard deviation
        """
        sum_diff_squared = np.zeros((self.data_dim))
        total_updates = 0

        # Loop over every observation, read into memory, and compute the sum of square differences over each dimension
        for index in self.indicies:
            with open(self.data_path, "rb") as file:
                file.seek(index)
                track = pickle.load(file)
            df = pd.DataFrame(track)

            # Compute the sum of square differences on a AIS update basis
            #for idx, row in df[["lat", "lon", "speed", "course"]].diff()[1:].iterrows():
            for (i, pos), (j, kin) in zip(df[["lat", "lon"]].diff()[1:].iterrows(),df[["speed", "course"]][1:].iterrows()):
                sum_diff_squared[0] += (pos["lat"] - self.mean[0].item()) ** 2
                sum_diff_squared[1] += (pos["lon"] - self.mean[1].item()) ** 2
                sum_diff_squared[2] += (kin["speed"] - self.mean[2].item()) ** 2
                sum_diff_squared[3] += (kin["course"] - self.mean[3].item()) ** 2
                total_updates += 1
        # Return the standard deviation
        std = (sum_diff_squared / (total_updates - 1)) ** (1 / 2)
        return torch.tensor(std, dtype=torch.float)

    def get_all_input_points_df(self):
        """Read in all the input data points and store in a single data frame

        Returns
        -------
        pandas.DataFrame
            Data frame with all the input data points
        """
        # Loop over every observation and store in one large data frame
        observations = []
        for index in self.indicies:
            with open(self.data_path, "rb") as file:
                file.seek(index)
                track = pickle.load(file)
            df = pd.DataFrame(track)

            for _, row in df[["lat", "lon", "speed", "course"]].iterrows():
                observations.append(row.to_numpy())
        return pd.DataFrame(
            observations,
            columns=["lat", "lon", "speed", "course"],
        )

    def get_labels(self):
        shiptypes = []
        lengths = []
        with torch.no_grad():
            for index in self.indicies:
                with open(self.data_path, "rb") as file:
                    file.seek(index)
                    track = pickle.load(file)
                    shiptypes.append(
                        dataset_utils.convertShipTypeToLabel(track["shiptype"])
                    )
                    lengths.append(track["track_length"])

        return torch.tensor(shiptypes), torch.tensor(lengths)
        
    def get_startpos(self,idx):
        # Load the data file for the requested index
        with open(self.data_path, "rb") as file:
            # Read the track from the corresponding point in memory
            file.seek(self.indicies[idx])
            track = pickle.load(file)
        df = pd.DataFrame(track)
        
        targets = torch.tensor(
            np.array(df[["lat", "lon", "speed", "course"]].values.tolist()),
            dtype=torch.float,
        )
        
        return targets[0,:]
        
        


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

    def __init__(
        self,
        file_name,
        train_mean=None,
        validation=False,
        data_info=None,
        indicies=None,
    ):
        """
        Parameters
        ----------
        file_name : str
            Name of the main part of the file where the results are saved

        train_mean : Tensor (Defaults to None)
            Mean of the times a certain bin was activated. When train_mean is not
            None, either a validation or test Dataset is created

        validation : bool (Defaults to False)
            When validation is True, a validation Dataset is created, but when it
            is False a test set is created (assuming train_mean is None)

        data_info : dict (Defaults to None)
            Information about the data. When None, it is read from the data_info_file

        indicies : list (Defaults to None)
            Train, val, or test indicies where the tracks begin in the data file.
            When None, the indicies are read from the requested data set
        """
        logger = logging.getLogger(__name__)  # For logging information

        # Setup the correct foldure structure
        project_dir = Path(__file__).resolve().parents[2]
        processed_data_dir = project_dir / "data" / "processed"

        # Read the data info pickle file into memory
        if data_info is None:
            data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
            logger.info("Processing data from the info file: " + str(data_info_file))
            with open(data_info_file, "rb") as f:
                self.data_info = pickle.load(f)
        else:
            self.data_info = data_info
        self.data_path = self.data_info["dataFileName"]

        # Get the requested data set (one of train/val/test)
        if indicies is None:
            if train_mean == None:
                self.indicies = self.data_info["trainIndicies"]
            else:
                if validation:
                    self.indicies = self.data_info["valIndicies"]
                else:
                    self.indicies = self.data_info["testIndicies"]
        else:
            self.indicies = indicies

        # Get the dimension of the four hot encoded vector (#Bins = #edges - 1)
        lat_edges, lon_edges, speed_edges, course_edges = self.data_info["binedges"]
        self.data_dim = (
            len(lat_edges) + len(lon_edges) + len(speed_edges) + len(course_edges) - 4
        )

        # Compute the mean from the training set but otherwise use the training mean
        if train_mean == None:
            logger.info(
                "AISDiscreteRepresentation: Computing training mean values using self.compute_mean()"
            )
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
        tuple
            Set of trajectory information as well as inputs and targets
        """

        # Load the data file for the requested index
        with open(self.data_path, "rb") as file:
            # Read the track from the corresponding point in memory
            file.seek(self.indicies[idx])
            track = pickle.load(file)
        df = pd.DataFrame(track)

        # Four hot encode the current trajectory
        encodedTrack = dataset_utils.FourHotEncode(df, self.data_info["binedges"])

        # Return the shipType as a label that could be useful later (e.g. for plotting)
        ship_type_label = dataset_utils.convertShipTypeToLabel(track["shiptype"])

        # The targets are just to reconstruct the input sequence
        targets = torch.tensor(encodedTrack, dtype=torch.float)  # seq_len X data_dim
        inputs = targets - self.mean  # Center the inputs

        # Return the trajectory with other (potentially) useful information
        return (
            torch.tensor(idx),
            torch.tensor(self.indicies[idx]),
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
        self.total_training_updates = total_updates

        # Mean of all the times a certain bin was activated. Used as a normalization factor (Centering in getimtem)
        return torch.tensor(sum_all / total_updates, dtype=torch.float)
