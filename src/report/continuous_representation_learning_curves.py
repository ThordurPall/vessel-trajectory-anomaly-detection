# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from src.visualisation.SummaryModels import SummaryModels


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization related to
    discrete representation learning curves
    """
    # learning_curves_Bornholm()
    Bornholm_test_set()
    # learning_curves_Skagen()


def learning_curves_Bornholm():
    """Constructs learning curves for cargo injected models in Bornholm"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(4, 30), (0, 1), (-30, -4)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the diagonal Gaussian
    setup_type = "Diagonal Gaussian"
    generative_dist = "Diagonal"
    learning_rate = 0.00005
    scheduler_gamma = [0.5, 0.5, 0.7, 0.6]
    scheduler_milestones = [500, 700, 1000, 1300]

    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_Diagonal = summary_models.load_curves_df(setup_type, level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(4, 40), (0, 10), (-40, -4)]
    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Loss_Learning_Curves",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
    )

    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[2]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
    )

    # Plot GMM models along with the diagonal Gaussian
    plt.clf()
    generative_dist = "GMM"
    setup_type = "GMM: 3 components"
    learning_rate = 0.00003
    GMM_equally_weighted = False
    scheduler_gamma = [0.7, 0.5, 0.6, 0.6]
    scheduler_milestones = [600, 800, 1000, 1300]
    GMM_components = 3
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        GMM_equally_weighted=GMM_equally_weighted,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        GMM_components=GMM_components,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_GMM_3 = summary_models.load_curves_df(setup_type)

    setup_type = "GMM: 4 components"
    GMM_components = 4
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        GMM_equally_weighted=GMM_equally_weighted,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        GMM_components=GMM_components,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_GMM_4 = summary_models.load_curves_df(setup_type, level=level)

    # Add the number of optimiser steps instead of using epoch (old way)
    df_GMM_4 = df_GMM_4[:3160]
    df_GMM_3["Epoch"] = df_GMM_4["Number of optimiser steps"]
    df_GMM_3.columns = df_GMM_4.columns

    # Concat the different models
    df_Diagonal = df_Diagonal[:3160]
    df = pd.concat([df_Diagonal, df_GMM_3, df_GMM_4])
    df.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    hue_order = ["Diagonal Gaussian", "GMM: 3 components", "GMM: 4 components"]
    x = "Number of optimiser steps"
    ylims = [(4, 40), (0, 10), (-40, -4)]
    xlims = [(0, 45000), (0, 45000), (0, 45000)]

    # Setup the correct foldure structure and do the plotting
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    summary_models.plot_curves(
        df[df["Data set type"] == "Validation"],
        hue,
        hue_order,
        x=x,
        ylims=[ylims[0]],
        xlims=[xlims[0]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Loss_Learning_Curves_Comparison",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
    )

    summary_models.plot_curves(
        df[df["Data set type"] == "Validation"],
        hue,
        hue_order,
        x=x,
        ylims=[ylims[2]],
        xlims=[xlims[2]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_Comparison",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
    )


def Bornholm_test_set():
    """Constructs figures using the Bornholm test set for the chocen model"""
    # Set variables to use for constructing the plot
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the diagonal Gaussian
    generative_dist = "Diagonal"
    learning_rate = 0.00005
    scheduler_gamma = [0.5, 0.5, 0.7, 0.6]
    scheduler_milestones = [500, 700, 1000, 1300]

    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=True,
    )

    # Get data on the test set
    data = summary_models.run_evaluation(validation=False)["TrajectoryLevelData"]

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    # Do the actual plotting
    x = "Equally weighted reconstruction log probability"
    summary_models.hist_stacked_plot(
        data,
        type="Histogram",
        x=x,
        print_summary_stats=True,
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Histogram",
        xlabel="Reconstruction log probability",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
