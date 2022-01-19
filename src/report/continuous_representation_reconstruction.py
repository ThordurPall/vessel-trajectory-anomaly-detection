# Loading the required packages to run this script
from src.visualisation.SummaryModels import SummaryModels
import src.utils.utils as utils
from src.visualisation.VisualiseTrajectories import VisualiseTrajectories
from src.data.Datasets import AISDataset
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():  # main(input_filepath, output_filepath):
    """Runs code to generate report ready visualization related to
    discrete representation learning curves
    """
    # validation_reconstruction_Bornholm()
    # validation_reconstruction_Bornholm_with_bias()
    # validation_reconstruction_Bornholm_with_bias_best()
    # test_reconstruction_Bornholm()
    test_reconstruction_Bornholm_for_comparison()
    # validation_reconstruction_Skagen_with_bias()
    # test_reconstruction_Skagen()


def validation_reconstruction_Bornholm():
    """Constructs reconstruction plots for models in Bornholm"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8
    n = 10

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.00005
    scheduler_gamma = [0.5, 0.5, 0.7, 0.6]
    scheduler_milestones = [500, 700, 1000, 1300]
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    data = summary_models.run_evaluation_get_n(n=100, worst=True)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the worst reconstruction

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Worst_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Worst_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )

    # Do the same as above for GMM
    plt.clf()
    generative_dist = "GMM"
    GMM_equally_weighted = False
    GMM_components = 4
    learning_rate = 0.00003
    scheduler_gamma = [0.7, 0.5, 0.6, 0.6]
    scheduler_milestones = [600, 800, 1000, 1300]

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        GMM_equally_weighted=GMM_equally_weighted,
        GMM_components=GMM_components,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    data = summary_models.run_evaluation_get_n(n=100, worst=True)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the worst reconstruction

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_GMM_Fishing_Vessel_Reconstruction_Worst_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_GMM_Fishing_Vessel_Reconstruction_Worst_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )


def validation_reconstruction_Bornholm_with_bias():
    """Constructs reconstruction plots for models (with bias) in Bornholm"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    data = summary_models.run_evaluation_get_n(n=100, worst=True)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the worst reconstruction

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Worst_1_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Worst_2_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Actual trajectory": df_actual,
            "Reconstructed trajectory": df_recon,
        }
    )

    # Do the same things as above for the second worst reconstruction
    idx = 1

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Worst_3_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Worst_4_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Actual trajectory": df_actual,
            "Reconstructed trajectory": df_recon,
        }
    )
    print(df_recon.sort_values("Speed sigma"))
    print(df_recon.sort_values("Course sigma"))


def validation_reconstruction_Bornholm_with_bias_best():
    """Constructs best reconstruction plots for models (with bias) in Bornholm"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    data = summary_models.run_evaluation_get_n(n=100, worst=False)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the best reconstruction

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Best_1_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Best_2_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Actual trajectory": df_actual,
            "Reconstructed trajectory": df_recon,
        }
    )

    # Do the same things as above for the second best reconstruction
    idx = 1

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Best_3_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Reconstruction_Best_4_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Actual trajectory": df_actual,
            "Reconstructed trajectory": df_recon,
        }
    )


def test_reconstruction_Bornholm():
    """Constructs reconstruction plots for models in Bornholm"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8
    n = 10

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
        use_generative_bias=use_generative_bias,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    validation = False
    data = summary_models.run_evaluation_get_n(n=100, worst=True, validation=validation)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the worst reconstruction (also the worst in Bernoulli)

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.test_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_1_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_1_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )
    plt.clf()

    # Show the fifth-worst reconstruction (the second-worst in Bernoulli)
    idx = 4

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.test_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_5_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_5_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )


def test_reconstruction_Bornholm_for_comparison():
    """Constructs reconstruction plots for models in Bornholm to copare with discrete representation"""
    # Define the region to look into
    region = "Bornholm"
    zoom = 8
    n = 10

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionBornholm_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.003
    scheduler_gamma = [0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.5]
    scheduler_milestones = [25, 50, 100, 150, 200, 250, 400]
    use_generative_bias = True
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
        use_generative_bias=use_generative_bias,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    validation = False
    data = summary_models.run_evaluation_get_n(n=100, worst=True, validation=validation)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = 381297
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = int(df_n.loc[df_n["Index"] == index]["Data set Index"])
    data_set = train_evaluate.test_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Comparison_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Comparison_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)
    index = 2777964
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = int(df_n.loc[df_n["Index"] == index]["Data set Index"])
    data_set = train_evaluate.test_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Comparison_3",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Bornholm_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Comparison_4",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )


def validation_reconstruction_Skagen_with_bias():
    """Constructs reconstruction plots for models (with bias) in Skagen"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        use_generative_bias=use_generative_bias,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    data = summary_models.run_evaluation_get_n(n=100, worst=True)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the worst reconstruction

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.validation_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Skagen_Diagonal_Fishing_Vessel_Reconstruction_Worst_1_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Skagen_Diagonal_Fishing_Vessel_Reconstruction_Worst_2_with_bias",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Actual trajectory": df_actual,
            "Reconstructed trajectory": df_recon,
        }
    )


def test_reconstruction_Skagen():
    """Constructs reconstruction plots for models in Skagen"""
    # Define the region to look into
    region = "Skagen"
    zoom = 7

    # Use the SummaryModels class for everything related to the reconstructions
    file_name = "RegionSkagen_01062019_30092019_Fish_14400_86400_600"
    generative_dist = "Diagonal"
    learning_rate = 0.001
    scheduler_gamma = [0.8, 0.7]
    scheduler_milestones = [20, 40]
    use_generative_bias = True
    fig_size = (4, 4)
    font_scale = 1.5

    # Use the SummaryModels class
    summary_models = SummaryModels(
        file_name,
        generative_dist=generative_dist,
        learning_rate=learning_rate,
        scheduler_gamma=scheduler_gamma,
        scheduler_milestones=scheduler_milestones,
        font_scale=font_scale,
        fig_size=fig_size,
        save_figures=True,
        plot_figures=False,
        use_generative_bias=use_generative_bias,
    )

    # Setup the correct foldure structure
    summary_models.model_fig_dir = (
        summary_models.project_dir / "figures" / "report" / "models"
    )
    summary_models.learning_curve_dir = summary_models.model_fig_dir / "reconstruction"

    # Use the VisualiseTrajectories class for geographically visualising trajectories
    fig_size = (10, 10)
    visualise_trajectories = VisualiseTrajectories(
        region,
        save_figures=True,
        plot_figures=False,
        fig_size=fig_size,
        zoom=zoom,
        font_scale=1.5 * font_scale,
    )
    processed_data_dir = visualise_trajectories.processed_data_dir

    # Read the info file to know how to read the data file
    data_file = processed_data_dir / ("data_" + file_name + ".pkl")
    data_info_file = processed_data_dir / ("datasetInfo_" + file_name + ".pkl")
    data_info = utils.read_data_info_file(data_info_file)

    # Load the static Google Map image for the RIO
    img = visualise_trajectories.read_static_map()

    # Get some trajectories to show an example from
    validation = False
    data = summary_models.run_evaluation_get_n(n=100, worst=True, validation=validation)
    df_n = data["TrajectoryLevelData"]
    train_evaluate = data["TrainEvaluateObject"]
    idx = 0  # Show the worst reconstruction

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.test_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.125, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Skagen_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_1_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.125, 0.65, 0.3, 0.23],
        file_name="Skagen_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_1_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Actual trajectory": df_actual,
            "Reconstructed trajectory": df_recon,
        }
    )
    print(df_recon.sort_values("Speed sigma"))
    print(df_recon.sort_values("Course sigma"))
    plt.clf()

    # Show the fifth-worst reconstruction
    idx = 4

    # Setup for the plotting
    fig, ax = visualise_trajectories.visualise_static_map(img)

    # For plotting a single complete vessel trajectory on the static map
    index = df_n.iloc[idx]["Index"]
    data_set = AISDataset(
        file_name, data_info=data_info, indicies=[index], discrete=False
    )
    df_actual = utils.get_tracks_from_dataset(data_set, continuous_representation=True)

    # For plotting the corresponding model reconstruction
    index = df_n.iloc[idx]["Data set Index"]
    data_set = train_evaluate.test_dataloader.dataset
    reconstruction = train_evaluate.track_reconstructions(data_set, index)
    df_recon = reconstruction["Reconstruction"][2:]  # Warmup

    # Setup actual and reconstruction data frames
    df_speed = utils.concat_actual_recon(df_actual, df_recon, "Speed")
    df_course = utils.concat_actual_recon(df_actual, df_recon, "Course")
    df_speed["Type"] = df_speed["Type"].str.replace("Actual", "Actual speed")
    df_course["Type"] = df_course["Type"].str.replace("Actual", "Actual course")

    # Do the actual plotting
    tmp = visualise_trajectories.trajectories_fig_dir
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_actual,
        ax,
        use_cmap=True,
        df_speed=df_speed,
        # df_course=df_course,
        fig=fig,
        rect=[0.200, 0.65, 0.3, 0.23],
        # rect2=[0.575, 0.72, 0.12, 0.12],
        file_name="Skagen_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_5_1",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    plt.clf()

    fig, ax = visualise_trajectories.visualise_static_map(img)
    visualise_trajectories.trajectories_fig_dir = summary_models.learning_curve_dir
    visualise_trajectories.plot_single_track(
        df_recon,
        ax,
        use_cmap=True,
        df_course=df_course,
        fig=fig,
        # rect=[0.125, 0.65, 0.3, 0.23],
        rect2=[0.200, 0.65, 0.3, 0.23],
        file_name="Skagen_Diagonal_Fishing_Vessel_Test_Set_Reconstruction_Worst_5_2",
        font_size=37,
    )
    visualise_trajectories.trajectories_fig_dir = tmp
    print(
        {
            "Reconstructed trajectory": df_recon,
        }
    )


if __name__ == "__main__":
    main()
