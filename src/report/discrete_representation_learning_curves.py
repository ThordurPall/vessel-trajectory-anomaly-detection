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
    # learning_curves_injected_cargo_Bornholm()
    # Bornholm_test_set()
    learning_curves_Skagen()


def learning_curves_Bornholm():
    """Constructs learning curves for cargo injected models in Bornholm"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(8, 25), (0, 0.1), (-25, -8)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the different models
    summary_models = SummaryModels(file_name)
    df_0_001 = summary_models.load_curves_df("LR: 0.001", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.0001)
    df_default_0_0001 = summary_models.load_curves_df("LR: 0.0001", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.0002)
    df_default_0_0002 = summary_models.load_curves_df("LR: 0.0002", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.00025)
    df_default_0_00025 = summary_models.load_curves_df("LR: 0.00025", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.0003)
    df_default_0_0003 = summary_models.load_curves_df("LR: 0.0003", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.0005)
    df_default_0_0005 = summary_models.load_curves_df("LR: 0.0005", level=level)

    summary_models = SummaryModels(
        file_name,
        learning_rate=0.0007,
        fig_size=fig_size,
        font_scale=font_scale,
        plot_figures=False,
        save_figures=True,
    )
    df_default_0_0007 = summary_models.load_curves_df("LR: 0.0007", level=level)

    # Condat the learning curve data frames
    df = pd.concat(
        [
            df_default_0_0001,
            df_default_0_0002,
            df_default_0_00025,
            df_default_0_0003,
            df_default_0_0005,
            # df_default_0_0007,
            df_0_001,
        ]
    )
    df.reset_index(drop=True, inplace=True)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    # Do the actual plotting
    hue = "Setup type"
    hue_order = [
        "LR: 0.0001",
        "LR: 0.0002",
        "LR: 0.00025",
        "LR: 0.0003",
        "LR: 0.0005",
        # "LR: 0.0007",
        "LR: 0.001",
    ]
    x = "Number of optimiser steps"
    summary_models.plot_curves(
        df[df["Data set type"] == "Validation"],
        hue,
        hue_order,
        # title="Validation data set",
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Discrete_Fishing_Vessel_Only_Validation_Loss_Learning_Curve_Comparison",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        remove_label_title=True,
    )

    summary_models.plot_curves(
        df[df["Data set type"] == "Validation"],
        hue,
        hue_order,
        # title="Validation data set",
        x=x,
        ylims=[ylims[2]],
        file_name="Bornholm_Discrete_Fishing_Vessel_Only_Validation_Reconstruction_Learning_Curve_Comparison",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        remove_label_title=True,
    )


def learning_curves_injected_cargo_Bornholm():
    """Constructs learning curves for cargo injected models in Bornholm"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(5, 25), (0, 1), (-25, -5)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the different models
    setup_type = "Fishing"
    summary_models = SummaryModels(
        file_name, save_figures=True, plot_figures=False, fig_size=fig_size
    )
    df_default_step = summary_models.load_curves_df(setup_type, level=level)

    setup_type = "Fishing + 5%"
    inject_cargo_proportion = 0.05
    summary_models_005 = SummaryModels(
        file_name,
        inject_cargo_proportion=inject_cargo_proportion,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
    )
    df_default_005_step = summary_models_005.load_curves_df(setup_type, level=level)

    setup_type = "Fishing + 10%"
    inject_cargo_proportion = 0.1
    summary_models_01 = SummaryModels(
        file_name,
        inject_cargo_proportion=inject_cargo_proportion,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
    )
    df_default_01_step = summary_models_01.load_curves_df(setup_type, level=level)

    setup_type = "Fishing + 20%"
    inject_cargo_proportion = 0.2
    summary_models_02 = SummaryModels(
        file_name,
        inject_cargo_proportion=inject_cargo_proportion,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
    )
    df_default_02_step = summary_models_02.load_curves_df(setup_type, level=level)

    setup_type = "Fishing + 50%"
    inject_cargo_proportion = 0.5
    summary_models_05 = SummaryModels(
        file_name,
        inject_cargo_proportion=inject_cargo_proportion,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
    )
    df_default_05_step = summary_models_05.load_curves_df(setup_type, level=level)

    setup_type = "Fishing + 100%"
    inject_cargo_proportion = 1.0
    summary_models_10 = SummaryModels(
        file_name,
        inject_cargo_proportion=inject_cargo_proportion,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
    )
    df_default_10_step = summary_models_10.load_curves_df(setup_type, level=level)

    setup_type = "Fishing + 200%"
    inject_cargo_proportion = 2.0
    summary_models_20 = SummaryModels(
        file_name,
        inject_cargo_proportion=inject_cargo_proportion,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
    )
    df_default_20_step = summary_models_20.load_curves_df(setup_type, level=level)

    setup_type = "Cargo/Tankers"
    model_prefix = "Fishing_vessels_only_"
    summary_models_carg_tank = SummaryModels(
        "RegionBornholm_01062019_30092019_CargTank_14400_86400_600",
        model_prefix=model_prefix,
    )
    df_carg_tank_step = summary_models_carg_tank.load_curves_df(
        setup_type, validation_only=True, level=level
    )

    setup_type = "Fishing/Cargo/Tankers"
    summary_models_fish_carg_tank = SummaryModels(
        "RegionBornholm_01062019_30092019_FishCargTank_14400_86400_600",
        model_prefix=model_prefix,
    )
    df_fish_carg_tank_step = summary_models_fish_carg_tank.load_curves_df(
        setup_type, validation_only=True, level=level
    )

    # Make the learning curve comparison plot
    df_step = pd.concat(
        [
            df_default_step,
            df_default_005_step,
            # df_default_01_step,
            df_default_02_step,
            df_default_05_step,
            # df_default_10_step,
            df_default_20_step,
            df_carg_tank_step,
            df_fish_carg_tank_step,
        ]
    )
    df_step.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    x = "Number of optimiser steps"
    hue_order = [
        "Fishing",
        "Fishing + 5%",
        #    "Fishing + 10%",
        "Fishing + 20%",
        "Fishing + 50%",
        # "Fishing + 100%",
        "Fishing + 200%",
        "Cargo/Tankers",
        "Fishing/Cargo/Tankers",
    ]

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    summary_models.plot_curves(
        df_step[df_step["Data set type"] == "Training"],
        hue,
        hue_order,
        title="Training data set",
        x=x,
    )
    summary_models.plot_curves(
        df_step[df_step["Data set type"] == "Validation"],
        hue,
        hue_order,
        title="Validation data set",
        x=x,
    )
    summary_models.plot_curves(
        df_step[df_step["Data set type"] == "Training"],
        hue,
        hue_order,
        title="Training data set",
        x=x,
        ylims=ylims,
    )
    summary_models.plot_curves(
        df_step[df_step["Data set type"] == "Validation"],
        hue,
        hue_order,
        # title="Validation data set",
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Discrete_Fishing_Vessel_Validation_Loss_Learning_Curve_Comparison",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
    )

    summary_models.plot_curves(
        df_step[df_step["Data set type"] == "Validation"],
        hue,
        hue_order,
        # title="Validation data set",
        x=x,
        ylims=[ylims[2]],
        file_name="Bornholm_Discrete_Fishing_Vessel_Validation_Reconstruction_Learning_Curve_Comparison",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
    )

    # Plot stacked reconstruction histograms for fishing/cargo/tanker vessels
    plt.clf()
    # fig_size = (14, 10)
    file_name = "RegionBornholm_01062019_30092019_FishCargTank_14400_86400_600"
    summary_models = SummaryModels(
        file_name,
        fig_size=fig_size,
        font_scale=font_scale,
        plot_figures=False,
        save_figures=True,
    )
    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    x = "Equally weighted reconstruction log probability"
    hue = "Ship type"
    hue_order = ["Fishing", "Cargo", "Tanker"]
    data = summary_models.run_evaluation()["TrajectoryLevelData"]
    summary_models.hist_stacked_plot(
        data, type="Histogram", x=x, hue=hue, hue_order=hue_order
    )
    summary_models.hist_stacked_plot(
        data, type="Stacked", x=x, hue=hue, hue_order=hue_order
    )
    plt.clf()
    summary_models.hist_stacked_plot(
        data,
        type="Stacked",
        x=x,
        hue=hue,
        hue_order=hue_order,
        stat="normalized_each_bin",
        ylabel="Stacked bin percentages",
        file_name="Bornholm_Discrete_Stacked_Histogram_Comparison",
        bins=30,
        xlabel="Reconstruction log probability",
    )


def Bornholm_test_set():
    """Constructs figures using the Bornholm test set for the chosen model"""
    # Set variables to use for constructing the plot
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the diagonal Gaussian
    generative_dist = "Bernoulli"
    learning_rate = 0.0003
    intermediate_epoch = 600

    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        intermediate_epoch=intermediate_epoch,
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
        file_name="Bornholm_Bernoulli_Fishing_Vessel_Test_Set_Reconstruction_Histogram",
        xlabel="Reconstruction log probability",
    )


def learning_curves_Skagen():
    """Constructs learning curves for Skagen ROI"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(5, 20), (0, 0.125), (-20, -5)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the different models
    summary_models = SummaryModels(file_name)
    df_default_0_001 = summary_models.load_curves_df("LR: 0.001", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.005)
    df_default_0_005 = summary_models.load_curves_df("LR: 0.005", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.0003)
    df_default_0_0003 = summary_models.load_curves_df("LR: 0.0003", level=level)

    summary_models = SummaryModels(file_name, learning_rate=0.0005)
    df_default_0_0005 = summary_models.load_curves_df("LR: 0.0005", level=level)

    # Add cargo and tankers for comparison
    setup_type = "Cargo/Tankers"
    model_prefix = "Fishing_vessels_only_"
    summary_models_carg_tank = SummaryModels(
        "RegionSkagen_01062019_30092019_CargTank_14400_86400_600",
        model_prefix=model_prefix,
    )
    df_carg_tank_step = summary_models_carg_tank.load_curves_df(
        setup_type, validation_only=True, level=level
    )

    setup_type = "Fish/Carg/Tank"
    summary_models_fish_carg_tank = SummaryModels(
        "RegionSkagen_01062019_30092019_FishCargTank_14400_86400_600",
        model_prefix=model_prefix,
    )
    df_fish_carg_tank_step = summary_models_fish_carg_tank.load_curves_df(
        setup_type, validation_only=True, level=level
    )

    summary_models = SummaryModels(
        file_name,
        learning_rate=0.0007,
        fig_size=fig_size,
        font_scale=font_scale,
        plot_figures=False,
        save_figures=True,
    )
    df_default_0_0007 = summary_models.load_curves_df("LR: 0.0007", level=level)

    # Condat the learning curve data frames
    df = pd.concat(
        [
            df_carg_tank_step,
            df_fish_carg_tank_step,
            df_default_0_0003,
            df_default_0_0005,
            df_default_0_0007,
            df_default_0_001,
            df_default_0_005,
        ]
    )
    df.reset_index(drop=True, inplace=True)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    # Do the actual plotting
    hue = "Setup type"
    hue_order = [
        # "LR: 0.0001",
        "Cargo/Tankers",
        "Fish/Carg/Tank",
        "LR: 0.0003",
        "LR: 0.0005",
        "LR: 0.0007",
        "LR: 0.001",
        "LR: 0.005",
    ]
    x = "Number of optimiser steps"
    summary_models.plot_curves(
        df[df["Data set type"] == "Validation"],
        hue,
        hue_order,
        # title="Validation data set",
        x=x,
        ylims=[ylims[0]],
        file_name="Skagen_Discrete_Fishing_Vessel_Only_Validation_Loss_Learning_Curve_Comparison",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        remove_label_title=True,
    )

    summary_models.plot_curves(
        df[df["Data set type"] == "Validation"],
        hue,
        hue_order,
        # title="Validation data set",
        x=x,
        ylims=[ylims[2]],
        file_name="Skagen_Discrete_Fishing_Vessel_Only_Validation_Reconstruction_Learning_Curve_Comparison",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        remove_label_title=True,
    )

    # Plot stacked reconstruction histograms for fishing/cargo/tanker vessels
    plt.clf()
    # fig_size = (14, 10)
    file_name = "RegionSkagen_01062019_30092019_FishCargTank_14400_86400_600"
    summary_models = SummaryModels(
        file_name,
        fig_size=fig_size,
        font_scale=font_scale,
        plot_figures=False,
        save_figures=True,
    )
    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    x = "Equally weighted reconstruction log probability"
    hue = "Ship type"
    hue_order = ["Fishing", "Cargo", "Tanker"]
    data = summary_models.run_evaluation()["TrajectoryLevelData"]
    summary_models.hist_stacked_plot(
        data, type="Histogram", x=x, hue=hue, hue_order=hue_order
    )
    summary_models.hist_stacked_plot(
        data, type="Stacked", x=x, hue=hue, hue_order=hue_order
    )
    plt.clf()
    summary_models.hist_stacked_plot(
        data,
        type="Stacked",
        x=x,
        hue=hue,
        hue_order=hue_order,
        stat="normalized_each_bin",
        ylabel="Stacked bin percentages",
        file_name="Skagen_Discrete_Stacked_Histogram_Comparison",
        bins=30,
        xlabel="Reconstruction log probability",
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
