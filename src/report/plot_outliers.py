# -*- coding: utf-8 -*-
from src.visualisation.SummaryModels import SummaryModels
import src.utils.utils as utils
from src.visualisation.VisualiseTrajectories import VisualiseTrajectories
from src.data.Datasets import AISDiscreteRepresentation
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.style as style
import matplotlib.pyplot as plt
import numpy as np

style.use("seaborn-colorblind")


def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization to use for
    showing detected outliers
    """
    contrario_epsilon = 1e-9
    # Bornholm_plot_detected_outlier(contrario_epsilon)
    # Bornholm_plot_detected_outlier_cargo_injected(contrario_epsilon)
    # Skagen_plot_detected_outlier(contrario_epsilon)
    # Skagen_plot_detected_outlier_06(contrario_epsilon)
    # Bornholm_plot_detected_outlier_differences(contrario_epsilon)
    # Bornholm_plot_detected_outlier_continuous(contrario_epsilon)
    # Skagen_plot_detected_outlier_continuous(contrario_epsilon)
    # Skagen_plot_detected_outlier_continuous_06(contrario_epsilon)
    # Skagen_plot_detected_outlier_differences(contrario_epsilon)
    Skagen_plot_detected_outlier_differences_Find(contrario_epsilon)


def Bornholm_plot_detected_outlier(contrario_epsilon=1e-9):
    """Plot the detected Bornholm outliers"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # Use the SummaryModels class
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    intermediate_epoch = 250
    summary_models = SummaryModels(
        file_name,
        intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Number of training AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_train,
        type=type,
        x=x,
        y=y,
        bins=60,
        vmax=100,
        cmin=5,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    # Plot normal test trajectories
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=normal_indicies,
        data_path=data_file,
        color=sns.color_palette()[1],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(
        loc="upper left",
        labels=["Test data: Normal", "Test data: Anomaly"],
        fontsize=22,
    )  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[1])
    leg.legendHandles[1].set_color(sns.color_palette()[2])
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Bornholm_Discrete_Fishing_Vessel_Outliers_EPS{contrario_epsilon}.pdf",
    )


def Bornholm_plot_detected_outlier_cargo_injected(contrario_epsilon=1e-9):
    """Plot the detected Bornholm outliers"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # Use the SummaryModels class
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    inject_cargo_proportion = 2.0
    intermediate_epoch = 250
    summary_models = SummaryModels(
        file_name,
        intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        inject_cargo_proportion=inject_cargo_proportion,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Number of training AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_train,
        type=type,
        x=x,
        y=y,
        bins=60,
        vmax=100,
        cmin=5,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    # Plot normal test trajectories
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=normal_indicies,
        data_path=data_file,
        color=sns.color_palette()[1],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(
        loc="upper left",
        labels=["Test data: Normal", "Test data: Anomaly"],
        fontsize=22,
    )  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[1])
    leg.legendHandles[1].set_color(sns.color_palette()[2])
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Bornholm_Discrete_Fishing_Vessel_Cargo_Injected_Outliers_EPS{contrario_epsilon}.pdf",
    )


def Bornholm_plot_detected_outlier_continuous(contrario_epsilon=1e-9):
    """Plot the detected Bornholm outliers when using a continious model"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # Use the SummaryModels class
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True
    contrario_epsilon = 1e-9
    summary_models = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories.trajectories_fig_dir = (
        visualise_trajectories.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories.plot_figures
    visualise_trajectories.plot_figures = False
    type = "Hex"
    cb_label = "Number of training AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories.plot_points(
        df_fishing_train,
        type=type,
        x=x,
        y=y,
        bins=60,
        vmax=100,
        cmin=5,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    # Plot normal test trajectories
    tracks = visualise_trajectories.plot_multiple_tracks(
        ax,
        indicies=normal_indicies,
        data_path=data_file,
        color=sns.color_palette()[1],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(
        loc="upper left",
        labels=["Test data: Normal", "Test data: Anomaly"],
        fontsize=22,
    )  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[1])
    leg.legendHandles[1].set_color(sns.color_palette()[2])
    tracks = visualise_trajectories.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )
    visualise_trajectories.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories.trajectories_fig_dir
        / f"Bornholm_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}.pdf",
    )


def Skagen_plot_detected_outlier(contrario_epsilon=1e-9):
    """Plot the detected Skagen outliers"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # Use the SummaryModels class
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    # intermediate_epoch = 250
    summary_models = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the normal test set
    lats, lons = [], []
    for idx in normal_indicies:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_normal = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal = visualise_trajectories_discrete.remove_points_outside_ROI(
        df_fishing_normal
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Number of normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_normal,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=22)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Skagen_Discrete_Fishing_Vessel_Outliers_EPS{contrario_epsilon}.pdf",
    )


def Skagen_plot_detected_outlier_06(contrario_epsilon=1e-9):
    """Plot the detected Skagen outliers"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # Use the SummaryModels class
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600_06"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    # intermediate_epoch = 250
    summary_models = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the normal test set
    lats, lons = [], []
    for idx in normal_indicies:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_normal = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal = visualise_trajectories_discrete.remove_points_outside_ROI(
        df_fishing_normal
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Number of normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_normal,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=22)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Skagen_Discrete_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_06.pdf",
    )


def Skagen_plot_detected_outlier_continuous(contrario_epsilon=1e-9):
    """Plot the detected Skagen outliers"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # Use the SummaryModels class
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True
    contrario_epsilon = 1e-9
    summary_models = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the normal test set
    lats, lons = [], []
    for idx in normal_indicies:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_normal = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal = visualise_trajectories.remove_points_outside_ROI(
        df_fishing_normal
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories.trajectories_fig_dir = (
        visualise_trajectories.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories.plot_figures
    visualise_trajectories.plot_figures = False
    type = "Hex"
    cb_label = "Number of normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories.plot_points(
        df_fishing_normal,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=22)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories.trajectories_fig_dir
        / f"Skagen_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}.pdf",
    )


def Skagen_plot_detected_outlier_continuous_06(contrario_epsilon=1e-9):
    """Plot the detected Skagen outliers"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # Use the SummaryModels class
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600_06"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True
    contrario_epsilon = 1e-9
    summary_models = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models.model_name
            + ".pkl"
        )
    )
    outlier_indicies = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the normal test set
    lats, lons = [], []
    for idx in normal_indicies:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_normal = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal = visualise_trajectories.remove_points_outside_ROI(
        df_fishing_normal
    )

    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories.trajectories_fig_dir = (
        visualise_trajectories.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories.plot_figures
    visualise_trajectories.plot_figures = False
    type = "Hex"
    cb_label = "Number of normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories.plot_points(
        df_fishing_normal,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=22)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories.trajectories_fig_dir
        / f"Skagen_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_06.pdf",
    )


def Bornholm_plot_detected_outlier_differences(contrario_epsilon=1e-9):
    """Plot the detected Bornholm outlier differences (discrete vs. continuous)"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # First setup the discrete case
    # Use the SummaryModels class
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    intermediate_epoch = 250
    summary_models_discrete = SummaryModels(
        file_name,
        intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 2.5
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers_discrete = utils.read_data_info_file(
        summary_models_discrete.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models_discrete.model_name
            + ".pkl"
        )
    )
    outlier_indicies_discrete = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers_discrete["test_outliers"])
        if v
    ]
    normal_indicies_discrete = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers_discrete["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train_discrete = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )

    # Setup the continuous case
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True
    contrario_epsilon = 1e-9
    summary_models_continuous = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 3.8
    visualise_trajectories_continuous = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_continuous.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_continuous.read_static_map()

    # Get the test set outliers
    outliers = utils.read_data_info_file(
        summary_models_continuous.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models_continuous.model_name
            + ".pkl"
        )
    )
    outlier_indicies_continuous = [
        i for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"]) if v
    ]
    normal_indicies_continuous = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train_continuous = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )

    # Tracks in one of the data representations but not in the other
    outlier_indicies_discrete_only = list(
        set(outlier_indicies_discrete) - set(outlier_indicies_continuous)
    )
    outlier_indicies_continuous_only = list(
        set(outlier_indicies_continuous) - set(outlier_indicies_discrete)
    )
    normal_indicies_discrete_only = list(
        set(normal_indicies_discrete) - set(normal_indicies_continuous)
    )
    normal_indicies_continuous_only = list(
        set(normal_indicies_continuous) - set(normal_indicies_discrete)
    )

    # Plot discrete only
    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Training AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_train_discrete,
        type=type,
        x=x,
        y=y,
        bins=60,
        vmax=100,
        cmin=5,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    # Plot abnormal test trajectories

    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_discrete_only,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp
    ax.legend(
        loc="upper left",
        labels=["Test data: Anomaly"],
        fontsize=28,
    )  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Bornholm_Discrete_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Comparison.pdf",
    )

    # Plot the same as above but continuous only
    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_continuous.trajectories_fig_dir = (
        visualise_trajectories_continuous.project_dir
        / "figures"
        / "report"
        / "outliers"
    )
    fig, ax = visualise_trajectories_continuous.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_continuous.plot_figures
    visualise_trajectories_continuous.plot_figures = False
    type = "Hex"
    cb_label = "Training AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_continuous.plot_points(
        df_fishing_train_continuous,
        type=type,
        x=x,
        y=y,
        bins=60,
        vmax=100,
        cmin=5,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    # Plot abnormal test trajectories
    tracks = visualise_trajectories_continuous.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_continuous_only,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )
    visualise_trajectories_continuous.plot_figures = plot_figures_tmp
    ax.legend(
        loc="upper left",
        labels=["Test data: Anomaly"],
        fontsize=28,
    )  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_continuous.trajectories_fig_dir
        / f"Bornholm_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Comparison.pdf",
    )


def Skagen_plot_detected_outlier_differences(contrario_epsilon=1e-9):
    """Plot the detected Skagen outlier differences (discrete vs. continuous)"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # First setup the discrete case
    # Use the SummaryModels class
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    # intermediate_epoch = 250
    summary_models_discrete = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 3
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers_discrete = utils.read_data_info_file(
        summary_models_discrete.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models_discrete.model_name
            + ".pkl"
        )
    )
    outlier_indicies_discrete = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers_discrete["test_outliers"])
        if v
    ]
    normal_indicies_discrete = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers_discrete["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the normal test set
    lats, lons = [], []
    for idx in normal_indicies_discrete:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_normal_discrete = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal_discrete = (
        visualise_trajectories_discrete.remove_points_outside_ROI(
            df_fishing_normal_discrete
        )
    )

    # Setup the continuous case
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True
    contrario_epsilon = 1e-9
    summary_models_continuous = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 4
    visualise_trajectories_continuous = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_continuous.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_continuous.read_static_map()

    # Get the test set outliers
    outliers_continuous = utils.read_data_info_file(
        summary_models_continuous.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models_continuous.model_name
            + ".pkl"
        )
    )

    outlier_indicies_continuous = [
        i
        for (i, v) in zip(
            data_info["testIndicies"], outliers_continuous["test_outliers"]
        )
        if v
    ]
    normal_indicies_continuous = [
        i
        for (i, v) in zip(
            data_info["testIndicies"], outliers_continuous["test_outliers"]
        )
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train_continuous = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal_continuous = (
        visualise_trajectories_discrete.remove_points_outside_ROI(
            df_fishing_train_continuous
        )
    )

    # Tracks in one of the data representations but not in the other
    outlier_indicies_discrete_only = list(
        set(outlier_indicies_discrete) - set(outlier_indicies_continuous)
    )
    outlier_indicies_continuous_only = list(
        set(outlier_indicies_continuous) - set(outlier_indicies_discrete)
    )

    # Plot discrete only
    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_normal_discrete,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_discrete_only[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=32)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_discrete_only,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Skagen_Discrete_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Comparison.pdf",
    )

    # Plot the same as above but continuous only
    visualise_trajectories_continuous.trajectories_fig_dir = (
        visualise_trajectories_continuous.project_dir
        / "figures"
        / "report"
        / "outliers"
    )
    fig, ax = visualise_trajectories_continuous.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_continuous.plot_figures
    visualise_trajectories_continuous.plot_figures = False
    type = "Hex"
    cb_label = "Normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_continuous.plot_points(
        df_fishing_normal_continuous,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories_continuous.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_continuous_only[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=32)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories_continuous.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories_continuous.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_continuous_only,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories_continuous.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_continuous.trajectories_fig_dir
        / f"Skagen_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Comparison.pdf",
    )


def Skagen_plot_detected_outlier_differences_Find(contrario_epsilon=1e-9):
    """Plot the detected Skagen outlier differences (discrete vs. continuous)"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # First setup the discrete case
    # Use the SummaryModels class
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Bernoulli"
    learning_rate = 0.001
    # intermediate_epoch = 250
    summary_models_discrete = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 3
    visualise_trajectories_discrete = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_discrete.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_discrete.read_static_map()

    # Get the test set outliers
    outliers_discrete = utils.read_data_info_file(
        summary_models_discrete.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models_discrete.model_name
            + ".pkl"
        )
    )
    outlier_indicies_discrete = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers_discrete["test_outliers"])
        if v
    ]
    normal_indicies_discrete = [
        i
        for (i, v) in zip(data_info["testIndicies"], outliers_discrete["test_outliers"])
        if not v
    ]

    # Get the lat and lon positions for the normal test set
    lats, lons = [], []
    for idx in normal_indicies_discrete:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_normal_discrete = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal_discrete = (
        visualise_trajectories_discrete.remove_points_outside_ROI(
            df_fishing_normal_discrete
        )
    )

    # Setup the continuous case
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True
    contrario_epsilon = 1e-9
    summary_models_continuous = SummaryModels(
        file_name,
        # intermediate_epoch=intermediate_epoch,
        learning_rate=learning_rate,
        generative_dist=generative_dist,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
    )

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    # fig_size = (4, 4)
    fig_size = (12, 8)
    # font_scale = 1.5
    font_scale = 4
    visualise_trajectories_continuous = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        continuous_representation=False,
        font_scale=font_scale,
    )
    processed_data_dir = visualise_trajectories_continuous.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories_continuous.read_static_map()

    # Get the test set outliers
    outliers_continuous = utils.read_data_info_file(
        summary_models_continuous.project_dir
        / "outliers"
        / (
            "outliers_eps"
            + str(contrario_epsilon)
            + "_"
            + summary_models_continuous.model_name
            + ".pkl"
        )
    )

    outlier_indicies_continuous = [
        i
        for (i, v) in zip(
            data_info["testIndicies"], outliers_continuous["test_outliers"]
        )
        if v
    ]
    normal_indicies_continuous = [
        i
        for (i, v) in zip(
            data_info["testIndicies"], outliers_continuous["test_outliers"]
        )
        if not v
    ]

    # Get the lat and lon positions for the training set
    lats, lons = [], []
    for idx in data_info["trainIndicies"]:
        df_lon_lat = utils.get_track_by_index(
            data_file, idx, keep_cols=["Longitude", "Latitude"]
        )
        lats += list(df_lon_lat["Latitude"])
        lons += list(df_lon_lat["Longitude"])
    df_fishing_train_continuous = pd.DataFrame(
        {
            "Latitude": lats,
            "Longitude": lons,
        }
    )
    df_fishing_normal_continuous = (
        visualise_trajectories_discrete.remove_points_outside_ROI(
            df_fishing_train_continuous
        )
    )

    # Tracks in one of the data representations but not in the other
    outlier_indicies_discrete_only = list(
        set(outlier_indicies_discrete) - set(outlier_indicies_continuous)
    )
    outlier_indicies_continuous_only = list(
        set(outlier_indicies_continuous) - set(outlier_indicies_discrete)
    )

    # Plot discrete only
    # Plot a heatmap of the AIS training update geographic points on the static map
    visualise_trajectories_discrete.trajectories_fig_dir = (
        visualise_trajectories_discrete.project_dir / "figures" / "report" / "outliers"
    )
    fig, ax = visualise_trajectories_discrete.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_discrete.plot_figures
    visualise_trajectories_discrete.plot_figures = False
    type = "Hex"
    cb_label = "Normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_discrete.plot_points(
        df_fishing_normal_discrete,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_discrete_only[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=32)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories_discrete.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_discrete_only,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories_discrete.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_discrete.trajectories_fig_dir
        / f"Skagen_Discrete_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Comparison_Find.pdf",
    )

    # Plot the same as above but continuous only
    visualise_trajectories_continuous.trajectories_fig_dir = (
        visualise_trajectories_continuous.project_dir
        / "figures"
        / "report"
        / "outliers"
    )
    fig, ax = visualise_trajectories_continuous.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_continuous.plot_figures
    visualise_trajectories_continuous.plot_figures = False
    type = "Hex"
    cb_label = "Normal test AIS messages"
    x = "Longitude"
    y = "Latitude"

    for i in range(len(outlier_indicies_continuous_only)):
        outlier_indicies_continuous_only_i = [outlier_indicies_continuous_only[i]]
        outlier_indicies_continuous_only_i.append(outlier_indicies_continuous_only[i])
        # outlier_indicies_continuous_only = outlier_indicies_continuous_only + [
        #    temp[i] for i in [1, 14, 18, 28, 34, 35, 46, 53, 59, 72, 89]
        # ]

        fig, ax = visualise_trajectories_continuous.visualise_static_map(img)
        plot_figures_tmp = visualise_trajectories_continuous.plot_figures
        visualise_trajectories_continuous.plot_figures = False

        visualise_trajectories_continuous.plot_points(
            df_fishing_normal_continuous,
            type=type,
            x=x,
            y=y,
            bins=100,
            vmax=200,
            cmin=10,
            cb_label=cb_label,
            # file_name="AIS_heatmap_Fishing",
            ax=ax,
            fig=fig,
        )

        tracks = visualise_trajectories_continuous.plot_multiple_tracks(
            ax,
            indicies=outlier_indicies_continuous_only_i,
            data_path=data_file,
            color=sns.color_palette()[2],
            plot_start=False,
            plot_end=False,
            s=100,
            fig=fig,
            plot_axis=True,
            ylim=[55.63, 59.4],
        )
        visualise_trajectories_continuous.plot_figures = plot_figures_tmp

        utils.add_plot_extras(
            ax,
            True,
            False,
            visualise_trajectories_continuous.trajectories_fig_dir
            / f"Skagen_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Comparison_Find_{i}.pdf",
        )
        plt.clf()

    # Plot the same as above but continuous only
    outlier_indicies_continuous_only = [
        outlier_indicies_continuous_only[i]
        for i in [1, 14, 18, 28, 34, 35, 46, 53, 59, 72, 89]
    ]
    breakpoint()
    visualise_trajectories_continuous.trajectories_fig_dir = (
        visualise_trajectories_continuous.project_dir
        / "figures"
        / "report"
        / "outliers"
    )
    fig, ax = visualise_trajectories_continuous.visualise_static_map(img)
    plot_figures_tmp = visualise_trajectories_continuous.plot_figures
    visualise_trajectories_continuous.plot_figures = False
    type = "Hex"
    cb_label = "Normal test AIS messages"
    x = "Longitude"
    y = "Latitude"
    visualise_trajectories_continuous.plot_points(
        df_fishing_normal_continuous,
        type=type,
        x=x,
        y=y,
        bins=100,
        vmax=200,
        cmin=10,
        cb_label=cb_label,
        # file_name="AIS_heatmap_Fishing",
        ax=ax,
        fig=fig,
    )

    tracks = visualise_trajectories_continuous.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_continuous_only[:3],
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
    )

    # Plot abnormal test trajectories
    ax.legend(loc="upper left", labels=["Test data: Anomaly"], fontsize=32)  # 37)
    leg = ax.get_legend()
    leg.legendHandles[0].set_color(sns.color_palette()[2])
    visualise_trajectories_continuous.plot_figures = plot_figures_tmp
    tracks = visualise_trajectories_continuous.plot_multiple_tracks(
        ax,
        indicies=outlier_indicies_continuous_only,
        data_path=data_file,
        color=sns.color_palette()[2],
        plot_start=False,
        plot_end=False,
        s=100,
        fig=fig,
        plot_axis=True,
        ylim=[55.63, 59.4],
    )
    visualise_trajectories_continuous.plot_figures = plot_figures_tmp

    utils.add_plot_extras(
        ax,
        True,
        False,
        visualise_trajectories_continuous.trajectories_fig_dir
        / f"Skagen_Diagonal_Fishing_Vessel_Outliers_EPS{contrario_epsilon}_Find.pdf",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    main()
