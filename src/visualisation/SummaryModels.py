from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import seaborn as sns

import src.utils.utils as utils


class SummaryModels:
    """
    A class used to handle everything related to visualising and providing
    a summary of different model setups

    ...

    Attributes
    ----------
    model_dir : pathlib.WindowsPath
        Directory where the model training results are located

    model_fig_dir : pathlib.WindowsPath
        Directory where model figures should be stored

    learning_curve_dir : pathlib.WindowsPath
        Directory where the learning curve model figures should be stored

    reconstruction_dir : pathlib.WindowsPath
        Directory where model trajectory reconstructions should be stored

    model_name : str
        String that indentifies the current model setup

    save_figures : bool
        Whether or not to save all created figures by default

    plot_figures : bool
        Whether or not to plot all created figures by default

    fig_size : tuple
        Default figure size to use for visualisations

    Methods
    -------
    xyz(resample_frequency)
        xyz


    """

    def __init__(
        self,
        file_name,
        save_figures=False,
        plot_figures=True,
        fig_size=(12, 8),
        model="VRNN",
        latent_dim="100",
        recurrent_dim="100",
        batch_norm=False,
        scheduler=False,
        kl_annealing=False,
    ):
        """
        Parameters
        ----------
        file_name : str
            Name of the main part of the file where the results are saved

        save_figures : bool (Defaults to False)
            Whether or not to save all created figures by default

        plot_figures : bool (Defaults to True)
            Whether or not to plot all created figures by default

        fig_size : tuple (Defaults to (12,8))
            Default figure size to use for visualizations

        model : str (Defaults to 'VRNN')
            String that describes the model used

        latent_dim : int (Defaults to '100')
            Latent space size in the networks

        recurrent_dim : int (Defaults to '100')
            Recurrent latent space size in the networks

        batch_norm : bool (Defaults to False)
            When set to True, batch normalization was included in the networks

        scheduler : bool (Defaults to False)
            When set to true a Scheduler was used

        kl_annealing : bool (Defaults to False)
            When set to true a Kullback–Leibler annealing was done
        """

        super().__init__()
        self.save_figures = save_figures
        self.plot_figures = plot_figures
        self.fig_size = fig_size

        # Setup the correct foldure structure
        project_dir = Path(__file__).resolve().parents[2]
        self.model_dir = project_dir / "models" / "saved-models"
        self.model_fig_dir = project_dir / "figures" / "models"
        self.learning_curve_dir = self.model_fig_dir / "learning-curves"
        self.reconstruction_dir = self.model_fig_dir / "reconstruction"

        # Make sure that the model figure paths exists
        self.learning_curve_dir.mkdir(parents=True, exist_ok=True)
        self.reconstruction_dir.mkdir(parents=True, exist_ok=True)

        # Construct the entire model name string for the current model setup
        BatchNorm = "_batchNormTrue" if batch_norm else "_batchNormFalse"
        Scheduler = "_SchedulerTrue" if scheduler else ""
        KLAnneal = "_KLTrue" if kl_annealing else ""
        self.model_name = (
            model
            + "_"
            + file_name
            + "_latent"
            + latent_dim
            + "_recurrent"
            + recurrent_dim
            + BatchNorm
            + Scheduler
            + KLAnneal
        )

        # Use seaborn style defaults and set the default figure size
        sns.set(rc={"figure.figsize": self.fig_size})
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", rc={"lines.linewidth": 3.0})
        style.use("seaborn-colorblind")

    # Define a function to read in the training/validation curves for a specified setup


def load_curves_df(self, setup_type, validation_only=False):
    """Read learning curves for the current setup

    Parameters
    ----------
    setup_type : str
        Model setup used. This will be the value in the 'Setup type' column

    validation_only : bool (Defaults to False)
        When True, only the validation curves are returned
    """
    curves_file = self.model_name + "_curves.csv"
    df = pd.read_csv(self.model_dir / curves_file)
    df_val = df.iloc[:, 3:6].copy()
    df_val.columns = ["Loss", "KL divergence", "Reconstruction log probabilities"]
    df_val["Data set type"] = "Validation"
    df_val["Epoch"] = df_val.index + 1

    if not validation_only:
        df_train = df.iloc[:, 0:3].copy()
        df_train.columns = ["Loss", "KL divergence", "Reconstruction log probabilities"]
        df_train["Data set type"] = "Training"
        df_train["Epoch"] = df_val.index + 1
        df = pd.concat([df_train, df_val])
    else:
        df = df_val
    df["Setup type"] = setup_type
    df.reset_index(drop=True, inplace=True)
    return df
