# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import math

from src.visualisation.SummaryTrajectories import SummaryTrajectories
from src.visualisation.VisualiseTrajectories import VisualiseTrajectories


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization to use for
    data exploration in the report
    """
    # explore_Gaussian_Bornholm()
    explore_Gaussian_Bornholm_with_bias()


def explore_Gaussian_Bornholm():
    """Look into the Bornholm Gaussian distribution found on some test examples"""
    # Use the SummaryTrajectories class
    fig_size = (4, 4)
    summary_file = "RegionBornholm_01062019_30092019_FishCargTank_1_315569220_0_trajectories_summary.csv"
    summary_trajectories = SummaryTrajectories(
        summary_file,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        date="DateTimeStart",
        font_scale=1.5,
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

    # Plot the mean speed distribution
    plt.clf()
    df_hist = df.reset_index()
    df_hist = df_hist.loc[df_hist["Ship type"] == "Fishing"]
    df_hist = df_hist.reset_index()
    mu = 2.5
    sigma = 3
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "MeanSpeed",
        file_name="Bornholm_mean_speed_distribution",
        xlabel="Mean speed (knots)",
        xlim=[0, 15],
        dist_x=x,
        dist_y=stats.norm.pdf(x, mu, sigma),
        stat="density",
    )

    # Plot the mean course histogram for fishing vessels
    plt.clf()
    mu = 187
    sigma = 100
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "MeanCourse",
        file_name="Bornholm_mean_course_distribution",
        xlabel="Mean course (degrees)",
        # xlim=[0, 360],
        dist_x=x,
        dist_y=stats.norm.pdf(x, mu, sigma),
        stat="density",
    )


def explore_Gaussian_Bornholm_with_bias():
    """Look into the Bornholm Gaussian distribution found on some test examples"""
    # Use the SummaryTrajectories class
    fig_size = (4, 4)
    summary_file = "RegionBornholm_01062019_30092019_FishCargTank_1_315569220_0_trajectories_summary.csv"
    summary_trajectories = SummaryTrajectories(
        summary_file,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        date="DateTimeStart",
        font_scale=1.5,
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

    # Plot the mean speed distribution
    plt.clf()
    df_hist = df.reset_index()
    df_hist = df_hist.loc[df_hist["Ship type"] == "Fishing"]
    df_hist = df_hist.reset_index()
    mu = 12.55
    sigma = 0.58
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "MeanSpeed",
        file_name="Bornholm_mean_speed_distribution_with_bias",
        xlabel="Mean speed (knots)",
        xlim=[0, 15],
        dist_x=x,
        dist_y=stats.norm.pdf(x, mu, sigma),
        stat="density",
    )

    # Plot the mean course histogram for fishing vessels
    plt.clf()
    mu = 184.8
    sigma = 100.7
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    summary_trajectories.hist_bar_plot(
        df_hist,
        "Histogram",
        "MeanCourse",
        file_name="Bornholm_mean_course_distribution_with_bias",
        xlabel="Mean course (degrees)",
        # xlim=[0, 360],
        dist_x=x,
        dist_y=stats.norm.pdf(x, mu, sigma),
        stat="density",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
