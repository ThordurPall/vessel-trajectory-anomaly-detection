# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualisation.SummaryTrajectories import SummaryTrajectories
from src.visualisation.VisualiseTrajectories import VisualiseTrajectories


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization to use for
    data exploration in the report
    """
    # show_region_based_heatmaps()
    show_data_summary_plots_Bornholm()
    # show_data_summary_plots_Skagen()


def show_region_based_heatmaps():
    """Outputs region based heatmaps of AIS messages"""
    # Define the regions to look into
    fig_size = (10, 10)
    Denmark = VisualiseTrajectories(
        region="Denmark",
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=6,
    )
    Skagen = VisualiseTrajectories(
        region="Skagen",
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=7,
    )
    Bornholm = VisualiseTrajectories(
        region="Bornholm",
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=8,
    )
    project_dir = Denmark.project_dir
    processed_data_dir = project_dir / "data" / "processed"

    Denmark.trajectories_fig_dir = (
        Denmark.project_dir / "figures" / "report" / "regions" / Denmark.region
    )
    Skagen.trajectories_fig_dir = (
        Skagen.project_dir / "figures" / "report" / "regions" / Skagen.region
    )
    Bornholm.trajectories_fig_dir = (
        Bornholm.project_dir / "figures" / "report" / "regions" / Bornholm.region
    )

    # Read in the latitude and longitude values from a pickle file - Fishing vessels
    file_name = "RegionAll_01042019_31032020_Fish_600_315569220_60"
    with open(processed_data_dir / (file_name + "_lats_lons.pickle"), "rb") as f:
        data = pickle.load(f)
    df_fishing = pd.DataFrame(
        {"Longitude": data["Longitudes"], "Latitude": data["Latitudes"]}
    )

    # Read in the latitude and longitude values from a pickle file - Cargo and tanker vessels
    file_name = "RegionAll_01042019_31032020_CargTank_600_315569220_60"
    with open(processed_data_dir / (file_name + "_lats_lons.pickle"), "rb") as f:
        data = pickle.load(f)
    df_carg_tank = pd.DataFrame(
        {"Longitude": data["Longitudes"], "Latitude": data["Latitudes"]}
    )

    # Fishing vessels
    print(np.shape(df_fishing))
    lat_lon_names = ["Latitude", "Longitude"]

    # Take out points outside the min/max lon/lat interval - Denmark
    df_fishing_Denmark = Denmark.remove_points_outside_ROI(df_fishing, lat_lon_names)
    print(np.shape(df_fishing_Denmark))

    # Take out points outside the min/max lon/lat interval - Skagen
    df_fishing_Skagen = Skagen.remove_points_outside_ROI(df_fishing, lat_lon_names)
    print(np.shape(df_fishing_Skagen))

    # Take out points outside the min/max lon/lat interval - Bornholm
    df_fishing_Bornholm = Bornholm.remove_points_outside_ROI(df_fishing, lat_lon_names)
    print(np.shape(df_fishing_Bornholm))
    print()

    # Cargo and tanker vessels
    print(np.shape(df_carg_tank))
    # Take out points outside the min/max lon/lat interval - Denmark
    df_carg_tank_Denmark = Denmark.remove_points_outside_ROI(
        df_carg_tank, lat_lon_names
    )
    print(np.shape(df_carg_tank_Denmark))

    # Take out points outside the min/max lon/lat interval - Skagen
    df_carg_tank_Skagen = Skagen.remove_points_outside_ROI(df_carg_tank, lat_lon_names)
    print(np.shape(df_carg_tank_Skagen))

    # Take out points outside the min/max lon/lat interval - Bornholm
    df_carg_tank_Bornholm = Bornholm.remove_points_outside_ROI(
        df_carg_tank, lat_lon_names
    )
    print(np.shape(df_carg_tank_Bornholm))

    # Plot a heatmap of the AIS update geographic points on the static map - Fishing vessels
    type = "Hex"
    cb_label = "Number of AIS messages"
    x = "Longitude"
    y = "Latitude"
    Denmark.plot_points(
        df_fishing_Denmark,
        type=type,
        x=x,
        y=y,
        bins=300,
        vmax=10000,
        cmin=2000,
        cb_label=cb_label,
        file_name="AIS_heatmap_Fishing",
    )
    Skagen.plot_points(
        df_fishing_Skagen,
        type=type,
        x=x,
        y=y,
        bins=250,  # 400,
        vmax=6000,
        cmin=1000,
        cb_label=cb_label,
        file_name="AIS_heatmap_Fishing",
    )
    Bornholm.plot_points(
        df_fishing_Bornholm,
        type=type,
        x=x,
        y=y,
        bins=150,
        vmax=2000,
        cmin=500,
        cb_label=cb_label,
        file_name="AIS_heatmap_Fishing",
    )

    # Plot a heatmap of the AIS update geographic points on the static map - Cargo and tanker vessels
    Denmark.plot_points(
        df_carg_tank_Denmark,
        type=type,
        x=x,
        y=y,
        bins=300,
        vmax=10000,
        cmin=2000,
        file_name="AIS_heatmap_Cargo",
        cb_label=cb_label,
    )
    Skagen.plot_points(
        df_carg_tank_Skagen,
        type=type,
        x=x,
        y=y,
        bins=250,  # 400,
        vmax=6000,
        cmin=1000,
        file_name="AIS_heatmap_Cargo",
        cb_label=cb_label,
    )
    Bornholm.plot_points(
        df_carg_tank_Bornholm,
        type=type,
        x=x,
        y=y,
        bins=150,
        vmax=2000,
        cmin=500,
        file_name="AIS_heatmap_Cargo",
        cb_label=cb_label,
    )


def show_data_summary_plots_Bornholm():
    """Outputs some data summary plots for the Bornholm ROI"""
    # Use the SummaryTrajectories class
    fig_size = (10, 10)
    summary_file = "RegionBornholm_01062019_30092019_FishCargTank_1_315569220_0_trajectories_summary.csv"
    summary_trajectories = SummaryTrajectories(
        summary_file,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        date="DateTimeStart",
    )
    summary_trajectories.explore_fig_dir = (
        summary_trajectories.project_dir / "figures" / "report" / "regions" / "Bornholm"
    )
    df = summary_trajectories.df
    df["Track length (sec)"] = pd.to_datetime(df["DateTimeEnd"]) - pd.to_datetime(
        df["DateTimeStart"]
    )
    df["Track length (sec)"] = df["Track length (sec)"].dt.total_seconds().astype(int)
    df["Ship type"] = df["ShipType"]

    # Plot the trajectory count for each ship type
    col_order = [1, 2, 0]
    summary_trajectories.hist_bar_plot(
        df["ShipType"].value_counts().reset_index(name="counts"),
        "Bar",
        "counts",
        "index",
        file_name="Bornholm_Summer_trajectory_count_by_shipType",
        xlabel="Trajectory count",
        ylabel="Ship type",
        col_order=col_order,
    )

    # Plot the mean speed histogram for each ship type (zoomed in a bit)
    plt.clf()
    df_hist = df.reset_index()
    hue_order = ["Fishing", "Cargo", "Tanker"]
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "MeanSpeed",
        file_name="Bornholm_mean_speed_histogram_by_shipType",
        xlabel="Mean speed (knots)",
        hue="Ship type",
        hue_order=hue_order,
        xlim=[-1, 22],
    )

    # Plot trajectory length (in seconds) for each ship type (zoomed in a bit)
    plt.clf()
    df_line = pd.DataFrame(
        {
            "x": [86400, 86400],
            "y": [0, 700],
        }
    )
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "Track length (sec)",
        file_name="Bornholm_trajectory_length_histogram_by_shipType",
        hue="Ship type",
        hue_order=hue_order,
        xlim=[0, 100000],
        ylim=[0, 700],
        df_line=df_line,
    )

    # After Preprocessing
    plt.clf()
    summary_file = "RegionBornholm_01062019_30092019_FishCargTank_14400_86400_600_trajectories_summary.csv"
    summary_trajectories = SummaryTrajectories(
        summary_file,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        date="DateTimeStart",
    )
    summary_trajectories.explore_fig_dir = (
        summary_trajectories.project_dir / "figures" / "report" / "regions" / "Bornholm"
    )
    df = summary_trajectories.df
    df["Track length (sec)"] = pd.to_datetime(df["DateTimeEnd"]) - pd.to_datetime(
        df["DateTimeStart"]
    )
    df["Track length (sec)"] = df["Track length (sec)"].dt.total_seconds().astype(int)
    df["Ship type"] = df["ShipType"]

    # Plot the trajectory count for each ship type
    summary_trajectories.hist_bar_plot(
        df["ShipType"].value_counts().reset_index(name="counts"),
        "Bar",
        "counts",
        "index",
        file_name="Bornholm_Summer_trajectory_count_by_shipType_After_Preprocessing",
        xlabel="Trajectory Count",
        ylabel="Ship type",
        col_order=col_order,
    )


def show_data_summary_plots_Skagen():
    """Outputs some data summary plots for the Skagen ROI"""
    # Use the SummaryTrajectories class
    fig_size = (10, 10)
    summary_file = "RegionSkagen_01062019_30092019_FishCargTank_1_315569220_0_trajectories_summary.csv"
    summary_trajectories = SummaryTrajectories(
        summary_file,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        date="DateTimeStart",
    )
    summary_trajectories.explore_fig_dir = (
        summary_trajectories.project_dir / "figures" / "report" / "regions" / "Skagen"
    )
    df = summary_trajectories.df
    df["Track length (sec)"] = pd.to_datetime(df["DateTimeEnd"]) - pd.to_datetime(
        df["DateTimeStart"]
    )
    df["Track length (sec)"] = df["Track length (sec)"].dt.total_seconds().astype(int)
    df["Ship type"] = df["ShipType"]

    # Plot the trajectory count for each ship type
    col_order = [1, 0, 2]
    summary_trajectories.hist_bar_plot(
        df["ShipType"].value_counts().reset_index(name="counts"),
        "Bar",
        "counts",
        "index",
        file_name="Skagen_Summer_trajectory_count_by_shipType",
        xlabel="Trajectory count",
        ylabel="Ship type",
        col_order=col_order,
    )

    # Plot the mean speed histogram for each ship type (zoomed in a bit)
    plt.clf()
    df_hist = df.reset_index()
    hue_order = ["Fishing", "Cargo", "Tanker"]
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "MeanSpeed",
        file_name="Skagen_mean_speed_histogram_by_shipType",
        xlabel="Mean speed (knots)",
        hue="Ship type",
        hue_order=hue_order,
        xlim=[-1, 22],
    )

    # Plot trajectory length (in seconds) for each ship type (zoomed in a bit)
    plt.clf()
    df_line = pd.DataFrame(
        {
            "x": [86400, 86400],
            "y": [0, 2250],
        }
    )
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "Track length (sec)",
        file_name="Skagen_trajectory_length_histogram_by_shipType",
        hue="Ship type",
        hue_order=hue_order,
        xlim=[0, 200000],
        ylim=[0, 2250],
        df_line=df_line,
    )

    # After Preprocessing
    plt.clf()
    summary_file = "RegionSkagen_01062019_30092019_FishCargTank_14400_86400_600_trajectories_summary.csv"
    summary_trajectories = SummaryTrajectories(
        summary_file,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        date="DateTimeStart",
    )
    summary_trajectories.explore_fig_dir = (
        summary_trajectories.project_dir / "figures" / "report" / "regions" / "Skagen"
    )
    df = summary_trajectories.df
    df["Track length (sec)"] = pd.to_datetime(df["DateTimeEnd"]) - pd.to_datetime(
        df["DateTimeStart"]
    )
    df["Track length (sec)"] = df["Track length (sec)"].dt.total_seconds().astype(int)
    df["Ship type"] = df["ShipType"]

    # Plot the trajectory count for each ship type
    summary_trajectories.hist_bar_plot(
        df["ShipType"].value_counts().reset_index(name="counts"),
        "Bar",
        "counts",
        "index",
        file_name="Skagen_Summer_trajectory_count_by_shipType_After_Preprocessing",
        xlabel="Trajectory Count",
        ylabel="Ship type",
        col_order=col_order,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
