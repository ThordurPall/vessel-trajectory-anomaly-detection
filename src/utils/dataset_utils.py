import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing


def classNames():
    # Defines the ship types of inetest
    names = [
        "Cargo",
        "Tanker",
        "Fishing",
        "Passenger",
        "Sailing",
        "Pleasure",
        "Military",
        "HSC",  # High Speed Vessel
        "Other",
    ]

    return np.array(names), len(names)


def convertShipTypeToLabel(shipType):
    # Convert ship type to a numerical label - Mainly for classification and plotting
    choices = {
        "Cargo": 0,
        "Tanker": 1,
        "Fishing": 2,
        "Passenger": 3,
        "Sailing": 4,
        "Pleasure": 5,
        "Military": 6,
        "HSC": 7,
        "Other": 8,
    }
    # There are many other types for instance Dive Vessel, Dredgers ect. These all are added to others.
    # See here for full list of types https://help.marinetraffic.com/hc/en-us/articles/205579997-What-is-the-significance-of-the-AIS-Shiptype-number-
    return choices.get(shipType, 8)


def createDenseVector(update, lat_edges, lon_edges, speed_edges, course_edges):
    # On an update level (for a single update) and not an entire trajectory
    # Four hot encoding on indivudual update attributes. Looks at the edges of
    # the bins defined and sets a bin to one if the value falls into that bin
    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    sog_dim = len(speed_edges) - 1
    cog_dim = len(course_edges) - 1
    data_dim = lat_dim + lon_dim + sog_dim + cog_dim

    # Find out in which bin the current updates belong
    # Take max to fix case when value = lowestEdge
    lat_idx = np.max([1, np.digitize(update["lat"], lat_edges, right=True)])
    lon_idx = np.max([1, np.digitize(update["lon"], lon_edges, right=True)])
    sog_idx = np.max([1, np.digitize(update["speed"], speed_edges, right=True)])
    cog_idx = np.max([1, np.digitize(update["course"], course_edges, right=True)])

    # All but four values will be zero - Set the four ones at correct indices
    dense_vect = np.zeros(data_dim)
    dense_vect[lat_idx - 1] = 1
    dense_vect[lat_dim + lon_idx - 1] = 1
    dense_vect[lat_dim + lon_dim + sog_idx - 1] = 1
    dense_vect[lat_dim + lon_dim + sog_dim + cog_idx - 1] = 1
    return dense_vect


def FourHotEncode(track, edges):
    # For hot encode an entire trajectory
    lat_edges, lon_edges, speed_edges, course_edges = edges

    # For each update in the trajectory apply createDenseVector
    EncodedTrack = track.apply(
        createDenseVector,
        axis=1,
        args=(lat_edges, lon_edges, speed_edges, course_edges),
    )
    return np.array(EncodedTrack.to_list())  # Return numpy array of ones and zeros


class AISDataset(torch.utils.data.Dataset):
    def __init__(self, infoPath, train_mean=None):
        self.Infopath = infoPath

        with open(self.Infopath, "rb") as f:
            self.params = pickle.load(f)

        if train_mean == None:
            self.indicies = self.params["trainIndicies"]
        else:
            self.indicies = self.params["testIndicies"]

        self.datapath = self.params["dataFileName"]
        self.datasetN = len(self.indicies)

        lat_edges, lon_edges, speed_edges, course_edges = self.params["binedges"]
        self.datadim = (
            len(lat_edges) + len(lon_edges) + len(speed_edges) + len(course_edges) - 4
        )

        self.shiptypes, self.lengths = self.getLabels()
        self.maxLength = torch.max(self.lengths)
        self.temporalMask = (
            torch.arange(self.maxLength, device="cpu")[:, None] < self.lengths[None, :]
        )  # max_seq_len X len(dataset)
        self.samples_pr_class = torch.bincount(self.shiptypes)
        self.nClasses = len(classNames()[0])

        if train_mean == None:
            self.mean = self.computeMean()
        else:
            self.mean = train_mean

    def __len__(self):
        return self.datasetN

    def __getitem__(self, idx):

        index = self.indicies[idx]

        with open(self.datapath, "rb") as file:
            file.seek(index)
            track = pickle.load(file)

        tmpdf = pd.DataFrame(track)
        encodedTrack = FourHotEncode(tmpdf, self.params["binedges"])
        label = convertShipTypeToLabel(track["shiptype"])
        targets = torch.tensor(encodedTrack, dtype=torch.float)  # seq_len X data_dim
        inputs = targets - self.mean

        return (
            torch.tensor(track["mmsi"]),
            np.array(track["timestamp"]),
            torch.tensor(label),
            torch.tensor(track["track_length"], dtype=torch.float),
            inputs,
            targets,
        )

    def computeMean(self):

        sum_all = np.zeros((self.datadim))
        total_updates = 0

        for index in self.indicies:
            with open(self.datapath, "rb") as file:
                file.seek(index)
                track = pickle.load(file)
                tmpdf = pd.DataFrame(track)
                encodedTrack = FourHotEncode(tmpdf, self.params["binedges"])
                sum_all += np.sum(encodedTrack, axis=0)  # Sum over all t
                total_updates += track["track_length"]

        mean = sum_all / total_updates

        return torch.tensor(mean, dtype=torch.float)

    def getLabels(self):

        shiptypes = []
        lengths = []
        with torch.no_grad():
            for index in self.indicies:
                with open(self.datapath, "rb") as file:
                    file.seek(index)
                    track = pickle.load(file)
                    shiptypes.append(convertShipTypeToLabel(track["shiptype"]))
                    lengths.append(track["track_length"])

        return torch.tensor(shiptypes), torch.tensor(lengths)
