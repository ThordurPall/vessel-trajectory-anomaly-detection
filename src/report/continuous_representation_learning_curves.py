# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualisation.SummaryModels import SummaryModels
import src.utils.utils as utils


def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization related to
    discrete representation learning curves
    """
    # learning_curves_Bornholm()
    # learning_curves_Bornholm_trials()
    # learning_curves_with_bias_Bornholm_trials()
    # learning_curves_Bornholm_with_Bias()
    # Bornholm_test_set()
    # learning_curves_Skagen()
    learning_curves_Skagen_with_Bias()
    # learning_curves_with_bias_Skagen_trials()


def learning_curves_Bornholm():
    """Constructs learning curves for cargo injected models in Bornholm"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(4, 30), (0, 1), (-30, -4)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    opt_steps_per_epoch = 29

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
    vertical_locations = [opt_steps_per_epoch * x for x in scheduler_milestones]
    vertical_heights = ([5] * len(scheduler_milestones),)
    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Loss_Learning_Curves",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        vertical_locations=vertical_locations,
        vertical_heights=vertical_heights,
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

    setup_type = "GMM: 2 components"
    GMM_components = 2
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
    df_GMM_2 = summary_models.load_curves_df(setup_type, level=level)

    # Add the number of optimiser steps instead of using epoch (old way)
    df_GMM_4 = df_GMM_4[:3160]
    df_GMM_2 = df_GMM_2[:3160]
    df_GMM_3["Epoch"] = df_GMM_4["Number of optimiser steps"]
    df_GMM_3.columns = df_GMM_4.columns

    # Concat the different models
    df_Diagonal = df_Diagonal[:3160]
    df = pd.concat([df_Diagonal, df_GMM_2, df_GMM_3, df_GMM_4])
    df.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    hue_order = [
        "Diagonal Gaussian",
        "GMM: 2 components",
        "GMM: 3 components",
        "GMM: 4 components",
    ]
    x = "Number of optimiser steps"
    ylims = [(4, 40), (0, 10), (-40, -4)]
    xlims = [(0, 45000), (0, 45000), (0, 45000)]

    # Setup the correct foldure structure and do the plotting
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    vertical_locations = [opt_steps_per_epoch * x for x in scheduler_milestones]
    vertical_heights = ([5] * len(scheduler_milestones),)
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
        vertical_locations=vertical_locations,
        vertical_heights=vertical_heights,
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


def learning_curves_Bornholm_with_Bias():
    """Constructs learning curves for diagonal Guassian model in Bornholm (with Bias)"""
    # Set variables to use for constructing the plot
    level = "Step"
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    opt_steps_per_epoch = 29

    # Get the learning curves for the diagonal Gaussian
    setup_type = "Diagonal Gaussian"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True

    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        use_generative_bias=use_generative_bias,
    )
    df_Diagonal = summary_models.load_curves_df(setup_type, level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(-5, 40), (0, 1), (-40, 5)]
    vertical_locations = [opt_steps_per_epoch * x for x in scheduler_milestones]
    vertical_heights = [-4] * len(scheduler_milestones)
    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Loss_Learning_Curves_with_bias",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        vertical_locations=vertical_locations,
        vertical_heights=vertical_heights,
        vertical_heights_min=-5,
    )

    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[2]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_with_bias",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
    )


def Bornholm_test_set():
    """Constructs figures using the Bornholm test set for the chosen model"""
    # Set variables to use for constructing the plot
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the diagonal Gaussian
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True

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
        use_generative_bias=use_generative_bias,
    )

    # Get data on the test set
    data = summary_models.run_evaluation(validation=False)["TrajectoryLevelData"]

    # Get outliers and normal indicies
    processed_data_dir = summary_models.project_dir / "data" / "processed"
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)
    contrario_epsilon = 1e-9

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
    track_type = []
    for i in data["Index"]:
        track_type.append("Anomalous" if i in outlier_indicies else "Normal")
    data["Trajectory type"] = track_type

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
        xlabel="Reconstruction log likelihood",
        hue="Trajectory type",
        hue_order=["Normal", "Anomalous"],
        palette=True,
    )


def learning_curves_Bornholm_trials():
    """Show different learning rate trials in Bornholm"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(4, 30), (0, 1), (-30, -4)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Use the SummaryModels class
    generative_dist = "GMM"
    GMM_equally_weighted = False
    learning_rate = 0.0001
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        GMM_equally_weighted=GMM_equally_weighted,
    )
    df_0_00003 = summary_models.load_curves_df("GMM, LR: 0.0001", level=level)

    generative_dist = "Isotropic_Gaussian"
    learning_rate = 0.00005
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_0_00005 = summary_models.load_curves_df("Gaussian, LR: 0.00005", level=level)

    learning_rate = 0.0001
    summary_models = SummaryModels(
        "RegionBornholm_01062019_30092019_FishCargTank_14400_86400_600",
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_0_0001 = summary_models.load_curves_df(
        "Fishing/Cargo/Tankers", level=level, validation_only=True
    )

    generative_dist = "GMM"
    learning_rate = 0.0003
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        GMM_equally_weighted=GMM_equally_weighted,
    )
    df_0_0003 = summary_models.load_curves_df("GMM, LR: 0.0003", level=level)

    generative_dist = "Isotropic_Gaussian"
    learning_rate = 0.001
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_0_001 = summary_models.load_curves_df("Gaussian, LR: 0.001", level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    df = pd.concat([df_0_00003, df_0_00005, df_0_0001, df_0_0003, df_0_001])
    df.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    hue_order = [
        "GMM, LR: 0.0001",
        "GMM, LR: 0.0003",
        "Gaussian, LR: 0.00005",
        "Gaussian, LR: 0.001",
        "Fishing/Cargo/Tankers",
    ]

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(4, 40), (0, 10), (-40, -4)]
    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Loss_Learning_Curves_Trials",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )

    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[2]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_Trials",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )

    # Show trials with scheduler that decreases by a factor gamma after each epoch
    plt.clf()
    learning_rate = 0.001

    scheduler_gamma = 0.999
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        scheduler_gamma=scheduler_gamma,
    )
    df_0_999 = summary_models.load_curves_df("LR: 0.001, γ = 0.999", level=level)

    scheduler_gamma = 0.995
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        scheduler_gamma=scheduler_gamma,
    )
    df_0_995 = summary_models.load_curves_df("LR: 0.001, γ = 0.995", level=level)

    scheduler_gamma = 0.993
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        scheduler_gamma=scheduler_gamma,
    )
    df_0_993 = summary_models.load_curves_df("LR: 0.001, γ = 0.993", level=level)

    scheduler_gamma = 0.99
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        scheduler_gamma=scheduler_gamma,
    )
    df_0_99 = summary_models.load_curves_df("LR: 0.001, γ = 0.99", level=level)

    scheduler_gamma = 0.985
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        scheduler_gamma=scheduler_gamma,
    )
    df_0_985 = summary_models.load_curves_df("LR: 0.001, γ = 0.985", level=level)

    generative_dist = "Diagonal"
    learning_rate = 0.0001
    scheduler_gamma = 0.5
    scheduler_milestones = [350, 500, 800, 1200, 1500]
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
    )
    df_0_5 = summary_models.load_curves_df("LR: 0.0001, γ = 0.5", level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    df = pd.concat([df_0_999, df_0_995, df_0_993, df_0_99, df_0_985, df_0_5])
    df.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    hue_order = [
        "LR: 0.001, γ = 0.999",
        "LR: 0.001, γ = 0.995",
        "LR: 0.001, γ = 0.993",
        "LR: 0.001, γ = 0.99",
        "LR: 0.001, γ = 0.985",
        "LR: 0.0001, γ = 0.5",
    ]

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(4, 40), (0, 10), (-40, -4)]
    xlims = [(0, 14000), (0, 14000), (0, 14000)]
    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[0]],
        xlims=[xlims[0]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Loss_Learning_Curves_Trials_2",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )

    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[2]],
        xlims=[xlims[2]],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_Trials_2",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )


def learning_curves_with_bias_Bornholm_trials():
    """Show different learning rate trials in Bornholm for models that have bias"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(4, 30), (0, 1), (-30, -4)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"

    # Use the SummaryModels class
    generative_dist = "Diagonal"
    learning_rate = 0.00005
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_00005 = summary_models.load_curves_df("LR: 0.00005", level=level)

    learning_rate = 0.0001
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_0001 = summary_models.load_curves_df("LR: 0.0001", level=level)

    learning_rate = 0.0005
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_0005 = summary_models.load_curves_df("LR: 0.0005", level=level)

    learning_rate = 0.001
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_001 = summary_models.load_curves_df("LR: 0.001", level=level)

    learning_rate = 0.003
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_003 = summary_models.load_curves_df("LR: 0.003", level=level)

    learning_rate = 0.005
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_005 = summary_models.load_curves_df("LR: 0.005", level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    df = pd.concat(
        [
            df_default_0_00005,
            df_default_0_0001,
            df_default_0_0005,
            df_default_0_001,
            df_default_0_003,
            df_default_0_005,
        ]
    )
    df.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    hue_order = [
        "LR: 0.00005",
        "LR: 0.0001",
        "LR: 0.0005",
        "LR: 0.003",
        "LR: 0.005",
        "LR: 0.001",
    ]

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(-5, 40), (0, 10), (-40, 5)]
    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[0]],
        file_name="Bornholm_Bias_Diagonal_Fishing_Vessel_Loss_Learning_Curves_Trials",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )

    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[2]],
        file_name="Bornholm_Bias_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_Trials",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )


def learning_curves_Skagen():
    """Constructs learning curves for Skagen ROI"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(5, 40), (0, 10), (-40, -5)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the different models

    # Get the learning curves for the diagonal Gaussians
    generative_dist = "Diagonal"
    summary_models = SummaryModels(
        file_name, learning_rate=0.005, generative_dist=generative_dist
    )
    df_default_0_005 = summary_models.load_curves_df("LR: 0.005", level=level)

    summary_models = SummaryModels(
        file_name, learning_rate=0.003, generative_dist=generative_dist
    )
    df_default_0_003 = summary_models.load_curves_df("LR: 0.003", level=level)

    summary_models = SummaryModels(file_name, generative_dist=generative_dist)
    df_default_0_001 = summary_models.load_curves_df("LR: 0.001", level=level)

    summary_models = SummaryModels(
        file_name, learning_rate=0.0005, generative_dist=generative_dist
    )
    df_default_0_0005 = summary_models.load_curves_df("LR: 0.0005", level=level)

    summary_models = SummaryModels(
        file_name, learning_rate=0.0001, generative_dist=generative_dist
    )
    df_default_0_0001 = summary_models.load_curves_df("LR: 0.0001", level=level)

    summary_models = SummaryModels(
        file_name, learning_rate=0.00005, generative_dist=generative_dist
    )
    df_default_0_00005 = summary_models.load_curves_df("LR: 0.00005", level=level)

    # summary_models = SummaryModels(file_name, learning_rate=0.00001)
    # df_default_0_00001 = summary_models.load_curves_df("LR: 0.00001", level=level)

    summary_models = SummaryModels(
        file_name,
        learning_rate=0.00003,
        fig_size=fig_size,
        font_scale=font_scale,
        plot_figures=False,
        save_figures=True,
        generative_dist=generative_dist,
    )
    df_default_0_00003 = summary_models.load_curves_df("LR: 0.00003", level=level)

    # Condat the learning curve data frames
    df = pd.concat(
        [
            # df_default_0_00001
            df_default_0_00003,
            df_default_0_00005,
            df_default_0_0001,
            df_default_0_0005,
            df_default_0_001,
            df_default_0_003,
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
        # "LR: 0.00001",
        "LR: 0.00003",
        "LR: 0.00005",
        "LR: 0.0001",
        "LR: 0.0005",
        "LR: 0.001",
        "LR: 0.003",
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
        file_name="Skagen_Continuous_Fishing_Vessel_Only_Validation_Loss_Learning_Curve_Comparison",
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
        file_name="Skagen_Continuous_Fishing_Vessel_Only_Validation_Reconstruction_Learning_Curve_Comparison",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        remove_label_title=True,
    )


def learning_curves_Skagen_with_Bias():
    """Constructs learning curves for diagonal Guassian model in Skagen (with Bias)"""
    # Set variables to use for constructing the plot
    level = "Step"
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    opt_steps_per_epoch = 813

    # Get the learning curves for the diagonal Gaussian
    setup_type = "Diagonal Gaussian"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True

    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
        use_generative_bias=use_generative_bias,
    )
    df_Diagonal = summary_models.load_curves_df(setup_type, level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(-5, 40), (0, 1), (-40, 5)]
    vertical_locations = [opt_steps_per_epoch * x for x in scheduler_milestones]
    vertical_heights = [-4] * len(scheduler_milestones)
    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[0]],
        file_name="Skagen_Diagonal_Fishing_Vessel_Loss_Learning_Curves_with_bias",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        vertical_locations=vertical_locations,
        vertical_heights=vertical_heights,
        vertical_heights_min=-5,
    )

    summary_models.plot_curves(
        df_Diagonal,
        x=x,
        ylims=[ylims[2]],
        file_name="Skagen_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_with_bias",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
    )


def learning_curves_with_bias_Skagen_trials():
    """Show different learning rate trials in Skagen for models that have bias"""
    # Set variables to use for constructing the plot
    level = "Step"
    ylims = [(4, 30), (0, 1), (-30, -4)]
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"

    # Use the SummaryModels class
    generative_dist = "Diagonal"
    learning_rate = 0.00001
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_00001 = summary_models.load_curves_df("LR: 0.00001", level=level)

    learning_rate = 0.0007
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_0007 = summary_models.load_curves_df("LR: 0.0007", level=level)

    learning_rate = 0.001
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_001 = summary_models.load_curves_df("LR: 0.001", level=level)

    learning_rate = 0.003
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_003 = summary_models.load_curves_df("LR: 0.003", level=level)

    learning_rate = 0.005
    use_generative_bias = True
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        save_figures=True,
        plot_figures=False,
    )
    df_default_0_005 = summary_models.load_curves_df("LR: 0.005", level=level)

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "learning-curves"

    df = pd.concat(
        [
            df_default_0_00001,
            df_default_0_0007,
            df_default_0_001,
            df_default_0_003,
            df_default_0_005,
        ]
    )
    df.reset_index(drop=True, inplace=True)
    hue = "Setup type"
    hue_order = [
        "LR: 0.00001",
        "LR: 0.0007",
        "LR: 0.001",
        "LR: 0.003",
        "LR: 0.005",
    ]

    # Do the actual plotting
    x = "Number of optimiser steps"
    ylims = [(-5, 40), (0, 10), (-40, 5)]
    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[0]],
        file_name="Skagen_Bias_Diagonal_Fishing_Vessel_Loss_Learning_Curves_Trials",
        plot_kl=False,
        plot_recon=False,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )

    summary_models.plot_curves(
        df,
        x=x,
        ylims=[ylims[2]],
        file_name="Skagen_Bias_Diagonal_Fishing_Vessel_Reconstruction_Learning_Curves_Trials",
        plot_loss=False,
        plot_kl=False,
        plot_recon=True,
        fig_size=fig_size,
        hue=hue,
        hue_order=hue_order,
    )


def Skagen_test_set():
    """Constructs figures using the Skagen test set for the chosen model"""
    # Set variables to use for constructing the plot
    fig_size = (4, 4)
    font_scale = 1.5
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"

    # Get the learning curves for the diagonal Gaussian
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True

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
        use_generative_bias=use_generative_bias,
    )

    # Get data on the test set
    data = summary_models.run_evaluation(validation=False)["TrajectoryLevelData"]

    # Get outliers and normal indicies
    processed_data_dir = summary_models.project_dir / "data" / "processed"
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)
    contrario_epsilon = 1e-9

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
    track_type = []
    for i in data["Index"]:
        track_type.append("Anomalous" if i in outlier_indicies else "Normal")
    data["Trajectory type"] = track_type

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
        file_name="Skagen_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Histogram",
        xlabel="Reconstruction log likelihood",
        hue="Trajectory type",
        hue_order=["Normal", "Anomalous"],
        palette=True,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
