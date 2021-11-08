import pickle

import matplotlib.pyplot as plt
import pandas as pd

import src.utils.plotting as plotting


# Utils file for useful functions used throughout the code base
def add_plot_extras(
    ax,
    save_figure,
    plot_figure,
    file_name_path=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    title=None,
):
    """Add the requested plot extras

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Pre-existing axes for the plot

    save_figure : bool
        Whether or not to save all created figures by default

    plot_figure : bool
        Whether or not to plot all created figures by default

    file_name_path : pathlib.WindowsPath (Defaults to None)
        File path where the figure will be saved

    xlabel : str (Defaults to None)
        x label text to put on the plot

    ylabel : str (Defaults to None)
        y label text to put on the plot

    xlim : list (Defaults to None)
        Limit of the x-axis

    ylim : list (Defaults to None)
        Limit of the y-axis

    title : str (Defaults to None)
        The figure title
    """
    # Add labels when requested
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Change axis limits when requested
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # Add title when requested
    if title is not None:
        ax.set_title(title)

    # Check whether to save or plot the figure
    if save_figure & (file_name_path is not None):
        plt.savefig(file_name_path, bbox_inches="tight")
    if plot_figure:
        plt.show()


# Read the info file to know how to read the data file
def read_data_info_file(data_info_file):
    """Get the requested track - Read the data file from the current index

    Parameters
    ----------
    data_info_file : pathlib.WindowsPath
        Location of the file to read

    Returns
    ----------
    dict
        Dictionary of data file information
    """
    with open(data_info_file, "rb") as f:
        data_info = pickle.load(f)
    return data_info


def get_track_by_index(
    path=None,
    idx=None,
    keep_cols=None,
    col_names=None,
    data_set=None,
    continuous_representation=True,
):
    """Get the requested track - Read the data file from the current index

    Parameters
    ----------
    path : pathlib.WindowsPath (Defaults to None)
        Location of the file to read

    idx : int (Defaults to None)
        Where to start reading the data file

    keep_cols : list (Defaults to None)
        The columns to keep

    col_names : list (Defaults to None)
        The column names to use for the returned data frame

    data_set : src.data.Datasets.AISDiscreteRepresentation (Defaults to None)
        The data set to use to read in the track

    continuous_representation : bool (Defaults to True)
            Either continuous or discrete AIS trajectory representation

    Returns
    ----------
    pandas.DataFrame
        Data frame with the requested trajectory
    """
    if continuous_representation and path is not None and idx is not None:
        with open(path, "rb") as f:
            f.seek(idx)
            track = pickle.load(f)
        df = pd.DataFrame(track)

        if keep_cols is not None:
            df = df[keep_cols]
        if col_names is not None:
            df.columns = col_names
    elif not continuous_representation and data_set is not None:
        Longitude, Latitude = [], []
        for i in range(0, len(data_set)):
            mmsi, time, ship_type_label, track_length, inputs, target = data_set[i]

            # The targets are the actual tracks (not centered)
            lon, lat = plotting.PlotDatasetTrack(target, data_set.data_info["binedges"])
            Longitude.extend(lon)
            Latitude.extend(lat)

        df = pd.DataFrame({"Longitude": Longitude, "Latitude": Latitude})

    else:
        print(
            "Either use continuous representation with path and idx or discrete with data_set"
        )
        df = -1
    return df
