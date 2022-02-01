import datetime
import json
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import progressbar

from src.Config import Config


def jsonToDict(update, statushist):

    keys = [int(key) for key in statushist.keys()]
    key = (
        str(keys[np.where(np.array(keys) <= update[0])[0][-1]])
        if len(np.where(np.array(keys) <= update[0])[0]) > 0
        else False
    )

    # The ship status for the current update is the most recent status if it exists, but Moored otherwise
    status = (
        statushist[key] if key else "Moored"
    )  # If we dont know the status of the first updates we just want to discard them

    return {
        "timestamp": update[0],  # time in seconds since 2019-01-01 00:00 UTC
        "lat": update[1] / 600000,
        "lon": update[2] / 600000,
        "speed": update[3] / 10,
        "course": update[4] / 10 % 360,  # Make modulo 360 to turn -1 into 359
        "heading": update[5],
        "navstatus": status,
    }


def getMMSIs(
    ROI, maxSpeed, timePeriod, allowedShipTypes, navTypes, raw_data_dir, min_speed=None
):

    # The MMSI needs to be in the RIO within this time period - Get months in timeperiod
    s, e = timePeriod
    s = s.strftime("%y%m")
    e = e.strftime("%y%m")
    months = range(int(s), int(e) + 1)
    months = [str(m) for m in months if (m % 100 > 0 and m % 100 < 13)]

    allfolders = next(os.walk(raw_data_dir))[1]
    folders = []
    for folder in allfolders:
        month = re.search("_(.*)XX", folder).group(1)  # get month of the folder
        if month in months:  # if folder month is in timeperiod months
            folders.append(folder)

    paths = []  # Paths should contain a list of every file we should look into
    for folder in folders:
        files = os.listdir(raw_data_dir + "/" + folder)
        for file in files:
            shiptype = re.search("[a-zA-Z]*", file).group()
            if shiptype in allowedShipTypes:
                paths.append(raw_data_dir + "/" + folder + "/" + file)

    # Keep track of the total number of AIS points (after filtering all but min speed)
    # and keep track of the number of points after also filtering out by min speed
    n_points = 0
    n_points_after_min_speed = 0

    dataframe = pd.DataFrame(columns=["MMSI", "File"])
    for path in progressbar.progressbar(paths):
        f = open(path)
        track = json.load(f)[0]

        # Read data into dataframe
        # There is sometimes a JOSN field included called statushist. When it is
        # included it must be handled, but otherwise the status is just laststatus
        status = (
            track["statushist"]
            if "statushist" in track.keys()  # Take the entire statushist when available
            else {
                str(track["path"][0][0]): track["lastStatus"]
            }  # First timestap in path lastStatus similar to how it would be in statushist
        )

        # Goes through every path in the trajectory and get the AIS attributes and current ship status
        df = pd.DataFrame.from_records(
            [
                jsonToDict(update, status) for update in track["path"]
            ]  # One path update at a time
        )

        # Filter for all params x = x[x[:,LAT]>=LAT_MIN] ect.
        lat_min, lat_max, lon_min, lon_max = ROI
        t_min, t_max = timePeriod

        # Path times are in seconds since 2019-01-01 00:00 UTC
        t_min = round((t_min - datetime.datetime(2019, 1, 1)).total_seconds())
        t_max = round((t_max - datetime.datetime(2019, 1, 1)).total_seconds())
        df = df.loc[
            (df["timestamp"] >= t_min)
            & (df["timestamp"] < t_max)
            & (df["lat"] >= lat_min)
            & (df["lat"] <= lat_max)
            & (df["lon"] >= lon_min)
            & (df["lon"] <= lon_max)
            & (df["speed"] <= maxSpeed)
            & (df["navstatus"].isin(navTypes))
        ]
        n_points += len(df)

        if min_speed is not None:
            # Add filtering on the min speed as well as an experiment
            df = df.loc[df["speed"] >= min_speed]
            n_points_after_min_speed += len(df)

        # Add mmsi and path to dataframe (if there are any path updates left
        #  after filtering out what we are not interested in)
        if len(df.index) > 0:
            new_row = {"MMSI": track["mmsi"], "File": path}
            dataframe = dataframe.append(new_row, ignore_index=True)

    # Keep track of the total number of AIS points (after filtering all but min speed)
    # and keep track of the number of points after also filtering out by min speed
    print(
        "Total number of AIS points (after every filter expect min speed): "
        + str(n_points)
    )
    print(
        "Total number of AIS points (after also filtering out by min speed): "
        + str(n_points_after_min_speed)
    )
    dataframe.sort_values(["MMSI"], inplace=True)
    return dataframe


def ReadAndJoinTracks(paths):
    # Go through each file path the MMSI has data in
    dataframes = []
    for path in paths:
        f = open(path)
        track = json.load(f)[0]

        # Read data into dataframe
        stype = track["shiptype"]

        # There is sometimes a JOSN field included called statushist. When it is
        # included it must be handled, but otherwise the status is just laststatus
        # That is, if there is no statushist, every path AIS attribute update
        # will have the same status (lastStatus)
        status = (
            track["statushist"]  # Take the entire statushist when available
            if "statushist" in track.keys()
            else {str(track["path"][0][0]): track["lastStatus"]}
        )  # First timestap in path lastStatus similar to how it would be in statushist

        # Goes through every path in the trajectory and get the AIS attributes and current ship status
        dataframes.append(
            pd.DataFrame.from_records(
                [jsonToDict(update, status) for update in track["path"]]
            )
        )

    df = pd.concat(dataframes)
    df.sort_values(["timestamp"], inplace=True)

    return df, stype


def FilterDataFrame(df, ROI, maxSpeed, timePeriod, min_speed=None):
    lat_min, lat_max, lon_min, lon_max = ROI
    t_min, t_max = timePeriod
    t_min = round((t_min - datetime.datetime(2019, 1, 1)).total_seconds())
    t_max = round((t_max - datetime.datetime(2019, 1, 1)).total_seconds())
    df = df.loc[
        (df["timestamp"] >= t_min)
        & (df["timestamp"] <= t_max)
        & (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
        & (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
        & (df["speed"] <= maxSpeed)
    ]

    if min_speed is not None:
        # Add filtering on the min speed as well as an experiment
        df = df.loc[df["speed"] >= min_speed]
    return df


def FilterOutStationaryNavStatus(df):
    config = Config()  # General configuration
    # df = df.loc[(~df["navstatus"].isin(config.STAT_NAV_STATUSES))]
    df = df.loc[(~df["navstatus"].isin(config.get_property("STAT_NAV_STATUSES")))]
    return df


def SplitIntoTracks(df, timediff):

    # Split when time difference greater than timediff seconds
    group_ids = (df["timestamp"] > (df["timestamp"].shift() + timediff)).cumsum()
    df["TrackNumber"] = group_ids

    return df


def RemoveShortTracks(df, min_time, min_updates):

    tracks = df.groupby("TrackNumber")
    trackNums = []

    for tracknum, track in tracks:
        if (len(track) > min_updates) & (
            (track["timestamp"].iloc[-1] - track["timestamp"].iloc[0]) > min_time
        ):
            trackNums.append(tracknum)

    df = df.loc[(df["TrackNumber"].isin(trackNums))]

    return df


def InterpolateTrackAndResample(df, frequency):

    df["timestamp"] = df["timestamp"] + 1546300800  # Add time between 1970 and 2019
    df["timestamp"] = df["timestamp"].apply(datetime.datetime.utcfromtimestamp)

    # Resample time-series data to _frequency_ time between samples in seconds. The AIS message
    # attribute values at those time points will be the mean if there are values in that interval
    df = df.set_index("timestamp").resample(str(frequency) + "S", origin="start").mean()

    # After the resampling, there could be some missing values tat we need to interpolate
    df = df.interpolate(method="linear", axis=0)
    df = df.reset_index(level=0, inplace=False)

    return df


def dumpTrackToPickle(mmsi, shiptype, track, f):

    savedTrack = {
        "mmsi": mmsi,
        "shiptype": shiptype,
        "track_length": len(track.index),
        "lat": track["lat"].to_list(),
        "lon": track["lon"].to_list(),
        "speed": track["speed"].to_list(),
        "course": track["course"].to_list(),
        "heading": track["heading"].to_list(),
        "timestamp": track["timestamp"].to_list(),
    }
    index = f.tell()
    pickle.dump(savedTrack, f)

    return index


def createDataset(
    params,
    raw_data_dir,
    processed_data_dir,
    dataset_filename,
    inject_cargo_proportion=0.0,
):
    # Determine the max and min updates. The max update should not be overly large since
    # when sequences get too long it can hinder learning (derivatives can get small).
    if params["resampleFrequency"] == 0:  # No resampling (take the actual AIS updates)
        maxUpdates = 9 ** 9
        minUpdates = 0
    else:
        maxUpdates = params["maxTrackLength"] / params["resampleFrequency"]
        minUpdates = params["minTrackLength"] / params["resampleFrequency"]

    # Check if the max and min lengths are the same (so, we have fixed length trajectories)
    fixedLength = params["maxTrackLength"] == params["minTrackLength"]

    print("Finding relevant MMSIs")
    mmsis = getMMSIs(
        params["ROI"],
        params["maxspeed"],
        params["timeperiod"],
        params["shiptypes"],
        params["navstatuses"],
        raw_data_dir,
        params["minspeed"],
    )  # Returns a data frame with relevant "MMSI" and its "File" name

    if inject_cargo_proportion != 0.0:
        # Inject cargo ships in proportion to the MMSIS already defined
        cargo_mmsis = getMMSIs(
            params["ROI"],
            params["maxspeed"],
            params["timeperiod"],
            ["Carg"],
            params["navstatuses"],
            raw_data_dir,
            params["minspeed"],
        )
        cargo_mmsis = cargo_mmsis.sample(int(mmsis.shape[0] * inject_cargo_proportion))
        mmsis = pd.concat([mmsis, cargo_mmsis], ignore_index=True)
    dataFileName = processed_data_dir + "/data_" + dataset_filename + ".pkl"
    with open(dataFileName, "wb") as dataFile:
        print("Processing MMSIs")
        indicies = []
        for mmsi in progressbar.progressbar(pd.unique(mmsis["MMSI"])):
            tmp = mmsis.loc[mmsis["MMSI"] == mmsi, :]

            # Implement function that reads all data for given mmsi. Would be best if it saved it to a dataframe
            # ReadAndJoinTracks for the list of all filepaths this MMSI has data in
            # Returns a data frame of AIS message attributes with the current nav status, as well as shipType
            data, shipType = ReadAndJoinTracks(tmp["File"].tolist())
            # Take only data in this ROI and timeperiod, with min_speed < speed < maxspeed
            data = FilterDataFrame(
                data,
                params["ROI"],
                params["maxspeed"],
                params["timeperiod"],
                params["minspeed"],
            )
            data = FilterOutStationaryNavStatus(
                data
            )  # Take out stationary updates (like "At anchor")
            data = SplitIntoTracks(data, params["splitTrackTimeDiff"])

            # Remove short tracks since they might be really hard to learn
            data = RemoveShortTracks(data, params["minTrackLength"], minUpdates)

            # Split based on tracknumber, so we are now looking at each trajectgory
            mmsiTracks = data.groupby("TrackNumber")
            for tracknum, track in mmsiTracks:
                track["previous_timestamp"] = track["timestamp"].shift()
                track["timestamp_difference"] = (
                    track["timestamp"] - track["previous_timestamp"]
                )

                # Resample time-series data to resampleFrequency time between samples in seconds.
                # Then linearly interpolate missing values in the track
                if params["resampleFrequency"] != 0:
                    track = InterpolateTrackAndResample(
                        track, params["resampleFrequency"]
                    )
                else:
                    track["timestamp"] = (
                        track["timestamp"] + 1546300800
                    )  # Add time between 1970 and 2019
                    track["timestamp"] = track["timestamp"].apply(
                        datetime.datetime.utcfromtimestamp
                    )
                    track = track.drop(columns="navstatus")

                    # Calculate the maxUpdates from params["maxTrackLength"] and
                    # mean time difference between observations
                    maxUpdates = (
                        params["maxTrackLength"] // track["timestamp_difference"].mean()
                    )

                if fixedLength == True:
                    groups = track.groupby(
                        np.arange(len(track.index)) // maxUpdates
                    )  # Split ensure max length
                    for _, trackSegment in groups:
                        if len(trackSegment.index) == maxUpdates:
                            # Save tracksegment
                            index = dumpTrackToPickle(
                                mmsi, shipType, trackSegment, dataFile
                            )
                            indicies.append(index)

                else:
                    # Separate into pieces less than maxTrackLength
                    # Check how many tracks we need to split the enitre trajectory into
                    num_tracks = math.ceil(
                        len(track.index) / maxUpdates
                    )  # Split into num_tracks = Ceil(Duration/Maxlength) pieces

                    # Do the actual track splitting, where the traks are split into
                    # num_tracks many sub-tracks of equal size
                    for trackSegment in np.array_split(track, num_tracks):
                        # Save each track segment to a pickle file
                        index = dumpTrackToPickle(
                            mmsi, shipType, trackSegment, dataFile
                        )
                        indicies.append(index)

    track_indcies = {
        "indicies": indicies,
        "dataFileName": dataFileName,
        "ROI": params["ROI"],
        "timeperiod": params["timeperiod"],
        "maxspeed": params["maxspeed"],
        "minspeed": params["minspeed"],
        "navstatuses": params["navstatuses"],
        "shiptypes": params["shiptypes"],
        "binedges": params["binedges"],
        "minTrackLength": params["minTrackLength"],
        "maxTrackLength": params["maxTrackLength"],
        "resampleFrequency": params["resampleFrequency"],
    }

    with open(
        processed_data_dir + "/datasetInfo_" + dataset_filename + ".pkl", "wb"
    ) as file:
        pickle.dump(track_indcies, file)

    return track_indcies
