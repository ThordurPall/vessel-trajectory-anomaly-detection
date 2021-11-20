# -*- coding: utf-8 -*-
import datetime
import logging
from pathlib import Path

import click
import pandas as pd
import seaborn as sns

import src.utils.utils as utils
from src.visualisation.VisualiseTrajectories import VisualiseTrajectories


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization to use for
    explanations in the report
    """
    show_binning()


def show_binning():
    """Outputs a figure that shows how the binning works"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # Use the VisualiseTrajectories class
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region, save_figures=True, plot_figures=True, fig_size=fig_size, zoom=zoom
    )

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()
    file_name = "RegionBornholm_01062019_30092019_CargTank_14400_86400_600"
    processed_data_dir = visualise_trajectories.processed_data_dir
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(
        data_info_file
    )  # Read the info file to know how to read the data file
    print(data_info["indicies"][:10])

    # Plot two complete vessel trajectory examples on the static map
    fig, ax = visualise_trajectories.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories.plot_figures
    visualise_trajectories.plot_figures = False
    test = visualise_trajectories.plot_multiple_tracks(
        ax, indicies=[15450, 31414], data_path=data_file, use_cmap=True
    )
    visualise_trajectories.plot_figures = plot_figures_tmp

    # Get the lat/lon bin edges
    lat_edges = visualise_trajectories.binedges[0]
    lon_edges = visualise_trajectories.binedges[1]
    min_lat, max_lat = min(lat_edges), max(lat_edges)
    min_lon, max_lon = min(lon_edges), max(lon_edges)
    linewidth = 1  # Bin line chart width
    step = 5  # Step size to take while plotting the bins (set to 1 to show all bins)

    # Plot the latitude bins
    for i in range(0, len(lat_edges), step):
        data_tmp = pd.DataFrame(
            {"Latitude": [lat_edges[i], lat_edges[i]], "Longitude": [min_lon, max_lon]}
        )
        sns.lineplot(
            x="Longitude",
            y="Latitude",
            color="black",
            linewidth=linewidth,
            ax=ax,
            data=data_tmp,
        )

    # Plot the longitude bins
    for i in range(0, len(lon_edges), step):
        data_tmp = pd.DataFrame(
            {"Latitude": [min_lat, max_lat], "Longitude": [lon_edges[i], lon_edges[i]]}
        )
        sns.lineplot(
            x="Longitude",
            y="Latitude",
            color="black",
            linewidth=linewidth,
            ax=ax,
            data=data_tmp,
        )
    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories.project_dir
        / "figures"
        / "report"
        / "Explain_Binning_Bornholm.pdf",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
