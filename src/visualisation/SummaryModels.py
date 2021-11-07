from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import seaborn as sns

import src.utils.utils as utils
from src.models.TrainEvaluate import TrainEvaluate


class SummaryModels:
    """
    A class used to handle everything related to visualising and providing
    a summary of different model setups

    ...

    Attributes
    ----------
    file_name : str
            Name of the main part of the file where the results are saved

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
    load_curves_df(setup_type, validation_only)
        Read in the training/validation curves for the current setup

    plot_curves_large(df, hue, hue_order, title)
        Plot the loss, KL divergence, and reconstruction log probabilities one at a time (large)

    plot_curves(df, hue, hue_order, title, file_name, xlims, ylims, fig_size)
        Plots the loss, KL divergence, and reconstruction log probabilities side by side

    run_evaluation(validation=True)
        Run evaluation loop and return the data

    hist_stacked_plot(data, type, x, y, file_name, xlabel, ylabel, hue, hue_order, xlim, ylim, print_summary_stats)
        Creates a histogram or stacked histogram plot
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
        model_prefix="",
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

        model_prefix : str (Defaults to empty string '')
            Model name prefix (e.g. 'Fishing_vessels_only_')
        """

        super().__init__()
        self.file_name = file_name
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
            model_prefix
            + model
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
        sns.set_style("ticks")

    def load_curves_df(self, setup_type, validation_only=False):
        """Read in the training/validation curves for the current setup

        Parameters
        ----------
        setup_type : str
            Model setup used. This will be the value in the 'Setup type' column

        validation_only : bool (Defaults to False)
            When True, only the validation curves are returned

        Returns
        -------
        pandas.DataFrame
            Data frame containing the learning curves
        """
        curves_file = self.model_name + "_curves.csv"
        df = pd.read_csv(self.model_dir / curves_file)
        if validation_only:
            df_val = df.iloc[:, 0:3].copy()
        else:
            df_val = df.iloc[:, 3:6].copy()
        df_val.columns = ["Loss", "KL divergence", "Reconstruction log probabilities"]
        df_val["Data set type"] = "Validation"
        df_val["Epoch"] = df_val.index + 1

        if not validation_only:
            df_train = df.iloc[:, 0:3].copy()
            df_train.columns = [
                "Loss",
                "KL divergence",
                "Reconstruction log probabilities",
            ]
            df_train["Data set type"] = "Training"
            df_train["Epoch"] = df_val.index + 1
            df = pd.concat([df_train, df_val])
        else:
            df = df_val
        df["Setup type"] = setup_type
        df.reset_index(drop=True, inplace=True)
        return df

    def plot_curves(
        self,
        df,
        hue="Data set type",
        hue_order=["Training", "Validation"],
        title=None,
        file_name=None,
        xlims=None,
        ylims=None,
        fig_size=(18, 5),
    ):
        """Plots the loss, KL divergence, and reconstruction log probabilities side by side

        Parameters
        ----------
        df : pandas.DataFrame
            Data to use when plotting the learning curves

        hue : str (Defaults to None)
            Variable that determines the color of plot elements

        hue_order : list (Defaults to None)
            Specify the order of processing and plotting for categorical levels of hue

        title : str
            Overall figure title

        file_name : str (Defaults to None)
            File name where the figure will be saved

        xlims : list (Defaults to None)
            Limit of the x-axis

        ylims : list (Defaults to None)
            Limit of the y-axis

        fig_size : tuple
            The overall figure size
        """
        _, ax = plt.subplots(1, 3, figsize=fig_size)
        x = "Epoch"
        sns.lineplot(x=x, y="Loss", hue=hue, hue_order=hue_order, data=df, ax=ax[0])
        sns.lineplot(
            x=x, y="KL divergence", hue=hue, hue_order=hue_order, data=df, ax=ax[1]
        )
        sns.lineplot(
            x=x,
            y="Reconstruction log probabilities",
            hue=hue,
            hue_order=hue_order,
            data=df,
            ax=ax[2],
        )
        sns.despine()

        if xlims is not None:
            ax[0].set(xlim=xlims[0])
            ax[1].set(xlim=xlims[1])
            ax[2].set(xlim=xlims[2])

        if ylims is not None:
            ax[0].set(ylim=ylims[0])
            ax[1].set(ylim=ylims[1])
            ax[2].set(ylim=ylims[2])

        if title is not None:
            plt.suptitle(title)

        file_path = None
        if file_name is not None:
            file_path = self.learning_curve_dir / (file_name + ".pdf")

        # Check whether to save or plot the figure
        if self.save_figures & (file_path is not None):
            plt.savefig(file_path, bbox_inches="tight")
        if self.plot_figures:
            plt.show()

    def plot_curves_large(
        self, df, hue="Data set type", hue_order=["Training", "Validation"], title=None
    ):
        """Plot the loss, KL divergence, and reconstruction log probabilities one at a time (large)

        Parameters
        ----------
        df : pandas.DataFrame
            Data to use when plotting the learning curves

        hue : str (Defaults to None)
            Variable that determines the color of plot elements

        hue_order : list (Defaults to None)
            Specify the order of processing and plotting for categorical levels of hue

        title : str
            Overall figure title
        """
        # Training and validation loss
        x = "Epoch"
        plt.figure()
        ax = sns.lineplot(x=x, y="Loss", hue=hue, hue_order=hue_order, data=df)
        sns.despine()
        if title is not None:
            ax.set_title(title)

        # Training and validation Kullback–Leibler divergence
        plt.figure()
        ax = sns.lineplot(x=x, y="KL divergence", hue=hue, hue_order=hue_order, data=df)
        sns.despine()
        if title is not None:
            ax.set_title(title)

        # Training and validation Reconstruction log probabilities
        plt.figure()
        ax = sns.lineplot(
            x=x,
            y="Reconstruction log probabilities",
            hue=hue,
            hue_order=hue_order,
            data=df,
        )
        sns.despine()
        if title is not None:
            ax.set_title(title)

        if self.plot_figures:
            plt.show()

    def run_evaluation(
        self,
        validation=True,
    ):
        """Run evaluation loop and return the data

        Parameters
        ----------
        validation : bool (Defaults to True)
            When True, the validation DataLoader is used, but otherwise the test loader is used

        Returns
        -------
        dict
            Dictionary containing the evaluation information
        """
        train_evaluate = TrainEvaluate(self.file_name, is_trained=True)
        if validation:
            eval_results = train_evaluate.evaluate_loop(
                train_evaluate.validation_dataloader, train_evaluate.validation_n
            )
        else:
            eval_results = train_evaluate.evaluate_loop(
                train_evaluate.test_dataloader, train_evaluate.test_n
            )
        data = pd.DataFrame(
            {
                "Reconstruction log probability": eval_results[3],
                "Length": eval_results[4],
                "Ship type": eval_results[5],
            }
        )
        data["Equally weighted reconstruction log probability"] = (
            data["Reconstruction log probability"] / data["Length"]
        )
        return {
            "EquallyWeightedMeanLoss": eval_results[0],
            "EquallyWeightedMeanKLDivergence": eval_results[1],
            "EquallyWeightedMeanReconstructionLogProbability": eval_results[2],
            "TrajectoryLevelData": data,
        }

    def hist_stacked_plot(
        self,
        data,
        type,
        x,
        y=None,
        file_name=None,
        xlabel=None,
        ylabel=None,
        hue=None,
        hue_order=None,
        xlim=None,
        ylim=None,
        print_summary_stats=False,
    ):
        """Creates a histogram or stacked histogram plot

        Parameters
        ----------
        data : int
            Data set to plot

        type : str
            Type of figure to plot (either "Histogram" or "Stacked")

        x : str
            Variable to plot on the x-axis

        y : str (Defaults to None)
            Variable to plot on the y-axis

        file_name : str (Defaults to None)
            File name where the figure will be saved

        xlabel : str (Defaults to None)
            x label text to put on the plot

        ylabel : str (Defaults to None)
            y label text to put on the plot

        hue : str (Defaults to None)
            Variable that determines the color of plot elements

        hue_order : list (Defaults to None)
            Specify the order of processing and plotting for categorical levels of hue

        xlim : list (Defaults to None)
            Limit of the x-axis

        ylim : list (Defaults to None)
            Limit of the y-axis

        print_summary_stats : bool
            When True, summary statistics will also be printed
        """
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", rc={"lines.linewidth": 3.0})
        style.use("seaborn-colorblind")

        if type == "Histogram":
            ax = sns.histplot(x=x, hue=hue, hue_order=hue_order, data=data)

        elif type == "Stacked":
            ax = sns.histplot(
                x=x, hue=hue, multiple="stack", hue_order=hue_order, data=data
            )
        else:
            print("Currently only implmented for 'Histogram' and 'Stacked'")

        file_path = None
        if file_name is not None:
            file_path = self.explore_fig_dir / (file_name + ".pdf")
        utils.add_plot_extras(
            ax,
            self.save_figures,
            self.plot_figures,
            file_path,
            xlabel,
            ylabel,
            xlim,
            ylim,
        )
        if print_summary_stats:
            print(
                f"Mean {x}: {data[x].mean()}",
                f"Median {x}: {data[x].median()}",
                f"{x} statistics by ship type: ",
                data.groupby("Ship type")[x].describe(),
                sep="\n",
            )
