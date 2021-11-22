# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.visualisation.VisualiseTrajectories import VisualiseTrajectories


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization to use for
    data exploration in the report
    """
    show_region_based_heatmaps()


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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
