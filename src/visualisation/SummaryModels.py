from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns

import src.utils.utils as utils
from src.models.TrainEvaluate import TrainEvaluate


class SummaryModels:
    """
    A class used to handle everything related to visualising and providing
    a summary of different model setups and their reconstructions

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

    GMM_components : int (Defaults to 4)
          The number of components to use as part of the GMM

    GMM_equally_weighted : bool (Defaults to True)
        When True, all GMM components are equally weighted

    Methods
    -------
    load_curves_df(setup_type, validation_only)
        Read in the training/validation curves for the current setup

    plot_curves_large(df, hue, hue_order, title)
        Plot the loss, KL divergence, and reconstruction log probabilities one at a time (large)

    plot_curves(df, hue, hue_order, title, file_name, xlims, ylims, fig_size)
        Plots the loss, KL divergence, and reconstruction log probabilities side by side

    run_evaluation(validation, setup_type, fishing_file, fishing_new_file)
        Run evaluation loop and return the data

    hist_stacked_plot(data, type, x, y, file_name, xlabel, ylabel, hue, hue_order, xlim, ylim, bins, print_summary_stats, title)
        Creates a histogram or stacked histogram plot

    intermediate_epoch : int (Defaults to None)
            When not None, the intermediate model saved at epoch intermediate_epoch will be loaded

    generative_dist : str (Defaults to 'Bernoulli')
        The observation model used
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
        kl_annealing=False,
        model_prefix="",
        inject_cargo_proportion=0.0,
        intermediate_epoch=None,
        generative_dist="Bernoulli",
        learning_rate=0.001,
        font_scale=1,
        GMM_components=4,
        GMM_equally_weighted=True,
        scheduler_gamma=None,
        scheduler_step_size=1,
        scheduler_milestones=None,
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

        kl_annealing : bool (Defaults to False)
            When set to true a Kullback–Leibler annealing was done

        model_prefix : str (Defaults to empty string '')
            Model name prefix (e.g. 'Fishing_vessels_only_')

        inject_cargo_proportion : float (Defaults to 0.0)
            Inject additional cargo/tanker vessel trajectories in inject_cargo_proportion proportion to the training trajectories

        intermediate_epoch : int (Defaults to None)
            When not None, the intermediate model saved at epoch intermediate_epoch will be loaded

        generative_dist : str (Defaults to 'Bernoulli')
            The observation model to use

        learning_rate : float (Defaults to 0.001)
            How much the models parameters were updated at each batch

        font_scale : int (Defaults to 1)
            The font size to use while plotting

        GMM_components : int (Defaults to 4)
          The number of components to use as part of the GMM

        GMM_equally_weighted : bool (Defaults to True)
            When True, all GMM components are equally weighted

        scheduler_gamma : float (Defaults to None)
            Multiplicative factor of learning rate decay. When None,
            no learning rate decay was applied

        scheduler_step_size : int (Defaults to 1)
            Decays the learning rate of each parameter group by gamma every step_size epochs

        scheduler_milestones : list (Defaults to None)
            The epochs where the learning rate is decreased by a factor of scheduler_gamma
        """

        super().__init__()
        self.file_name = file_name
        self.save_figures = save_figures
        self.plot_figures = plot_figures
        self.fig_size = fig_size
        self.model_prefix = model_prefix
        self.inject_cargo_proportion = inject_cargo_proportion
        self.intermediate_epoch = intermediate_epoch
        self.generative_dist = generative_dist
        self.discrete = True if self.generative_dist == "Bernoulli" else False
        self.font_scale = font_scale
        self.GMM_components = GMM_components
        self.GMM_equally_weighted = GMM_equally_weighted

        # Setup the correct foldure structure
        self.project_dir = Path(__file__).resolve().parents[2]
        self.model_dir = self.project_dir / "models" / "saved-models"
        self.model_fig_dir = self.project_dir / "figures" / "models"
        self.learning_curve_dir = self.model_fig_dir / "learning-curves"
        self.reconstruction_dir = self.model_fig_dir / "reconstruction"

        # Make sure that the model figure paths exists
        self.learning_curve_dir.mkdir(parents=True, exist_ok=True)
        self.reconstruction_dir.mkdir(parents=True, exist_ok=True)

        # Construct the entire model name string for the current model setup
        BatchNorm = "_batchNormTrue" if batch_norm else "_batchNormFalse"
        Scheduler = ""
        if scheduler_gamma is not None:
            if scheduler_milestones is not None:
                Scheduler = (
                    "_S"
                    + "".join([str(i) for i in scheduler_milestones])
                    + "_"
                    + str(scheduler_gamma).replace(".", "")
                )
            else:
                Scheduler = (
                    "_S"
                    + str(scheduler_step_size)
                    + "_"
                    + str(scheduler_gamma).replace(".", "")
                )

        KLAnneal = "_KLTrue" if kl_annealing else ""
        LearningRate = (
            "_lr" + str(learning_rate).replace(".", "")
            if learning_rate != 0.001
            else ""
        )
        cargo_injected = (
            "_Injected" + str(inject_cargo_proportion).replace(".", "")
            if inject_cargo_proportion != 0.0
            else ""
        )
        GenerativeDist = "_" + self.generative_dist if not self.discrete else ""
        GMMComponents = ""
        GMMEquallyWeighted = ""
        if self.generative_dist == "GMM":
            GMMComponents = str(self.GMM_components)
            if self.GMM_equally_weighted:
                GMMEquallyWeighted = "EW"
            else:
                GMMEquallyWeighted = "NEW"

        self.model_name = (
            model_prefix
            + model
            + "_"
            + file_name
            + cargo_injected
            + "_latent"
            + latent_dim
            + "_recurrent"
            + recurrent_dim
            + BatchNorm
            + GenerativeDist
            + GMMComponents
            + GMMEquallyWeighted
            + LearningRate
            + Scheduler
            + KLAnneal
        )

        # Use seaborn style defaults and set the default figure size
        sns.set(rc={"figure.figsize": self.fig_size})
        sns.set_theme(style="whitegrid")
        sns.set_context(
            "paper", rc={"lines.linewidth": 3.0}, font_scale=self.font_scale
        )
        style.use("seaborn-colorblind")
        sns.set_style("ticks")

    def load_curves_df(self, setup_type, validation_only=False, level="Epoch"):
        """Read in the training/validation curves for the current setup

        Parameters
        ----------
        setup_type : str
            Model setup used. This will be the value in the 'Setup type' column

        validation_only : bool (Defaults to False)
            When True, only the validation curves are returned

        level : str (Defaults to 'Epoch')
            The level of the learning curve values ('Epoch'/'Sample'/'Step')

        Returns
        -------
        pandas.DataFrame
            Data frame containing the learning curves
        """
        # Figure out the correct .csv file to read
        if level == "Step":
            level = "Number of optimiser steps"
            if self.model_prefix == "":
                curves_file = "opt_step_lvl_" + self.model_name
            else:
                curves_file = self.model_name.replace(
                    self.model_prefix, self.model_prefix + "opt_step_lvl_"
                )
        elif level == "Sample":
            level = "Number of training samples processed"
            if self.model_prefix == "":
                curves_file = "sample_lvl_" + self.model_name
            else:
                curves_file = self.model_name.replace(
                    self.model_prefix, self.model_prefix + "sample_lvl_"
                )
        else:
            curves_file = self.model_name
        curves_file += "_curves.csv"

        # Read the file and setup the data frame
        df = pd.read_csv(self.model_dir / curves_file)
        if validation_only:
            df_val = df.iloc[:, 0:3].copy()
        else:
            df_val = df.iloc[:, 3:6].copy()
        df_val.columns = ["Loss", "KL divergence", "Reconstruction log probabilities"]
        df_val["Data set type"] = "Validation"

        # Add the level to the data frame
        if level == "Epoch":
            df_val["Epoch"] = df_val.index + 1
        else:
            df_val[level] = df[level].copy()

        # Add the training curves when requested
        if not validation_only:
            df_train = df.iloc[:, 0:3].copy()
            df_train.columns = [
                "Loss",
                "KL divergence",
                "Reconstruction log probabilities",
            ]
            df_train["Data set type"] = "Training"
            # Add the level to the data frame
            if level == "Epoch":
                df_train["Epoch"] = df_val.index + 1
            else:
                df_train[level] = df[level].copy()
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
        x="Epoch",
        plot_loss=True,
        plot_kl=True,
        plot_recon=True,
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

        x : str (Defaults to "Epoch")
            The x-axis variable

        plot_loss : bool (Defailts to True)
            When True the loss learning curve is plotted

        plot_kl : bool (Defailts to True)
            When True the KL learning curve is plotted

        plot_recon : bool (Defailts to True)
            When True the reconstruction learning curve is plotted
        """
        n = plot_loss + plot_kl + plot_recon
        i = 0
        _, ax = plt.subplots(1, n, figsize=fig_size)
        ax = [ax] if n == 1 else ax

        # Plot the requested figures
        if plot_loss:
            sns.lineplot(x=x, y="Loss", hue=hue, hue_order=hue_order, data=df, ax=ax[i])
            i += 1
        if plot_kl:
            sns.lineplot(
                x=x, y="KL divergence", hue=hue, hue_order=hue_order, data=df, ax=ax[i]
            )
            i += 1
        if plot_recon:
            sns.lineplot(
                x=x,
                y="Reconstruction log probabilities",
                hue=hue,
                hue_order=hue_order,
                data=df,
                ax=ax[i],
            )
            i += 1

        sns.despine()

        if xlims is not None:
            for i in range(n):
                ax[i].set(xlim=xlims[i])

        if ylims is not None:
            for i in range(n):
                ax[i].set(ylim=ylims[i])

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
        setup_type=None,
        fishing_file=None,
        fishing_new_file=None,
    ):
        """Run evaluation loop and return the data

        Parameters
        ----------
        validation : bool (Defaults to True)
            When True, the validation DataLoader is used, but otherwise the test loader is used

        setup_type : str (Defaults to None)
            Model setup used. This will be the value in the 'Setup type' column

        fishing_file : str (Defaults to None)
            Name of the main part of the fishing vessel only data file for the same ROI and dates as in file_name.
            When not None, the evaluation will be done using this data set

        fishing_new_file : str (Defaults to None)
            Name of the main part of a new fishing vessel only data file with a different ROI or dates (or both).
            When not None, the evaluation will be done using this data set. In case fishing_file and fishing_new_file
            are both not None evaluation will be done on the fishing_file

        Returns
        -------
        dict
            Dictionary containing the evaluation information
        """
        # Initialize the TrainEvaluate with the current setup
        train_evaluate = TrainEvaluate(
            self.file_name,
            is_trained=True,
            fishing_file=fishing_file,
            fishing_new_file=fishing_new_file,
            inject_cargo_proportion=self.inject_cargo_proportion,
            intermediate_epoch=self.intermediate_epoch,
            generative_dist=self.generative_dist,
            trained_model_name=self.model_name,
            GMM_components=self.GMM_components,
            GMM_equally_weighted=self.GMM_equally_weighted,
        )

        # Check which data set to actually use when running the evaluation loop
        if fishing_file is not None:
            if validation:
                eval_results = train_evaluate.evaluate_loop(
                    train_evaluate.fishing_validation_dataloader,
                    train_evaluate.fishing_validation_n,
                )
            else:
                eval_results = train_evaluate.evaluate_loop(
                    train_evaluate.fishing_test_dataloader,
                    train_evaluate.fishing_test_n,
                )
        elif fishing_new_file is not None:
            if validation:
                eval_results = train_evaluate.evaluate_loop(
                    train_evaluate.fishing_new_validation_dataloader,
                    train_evaluate.fishing_new_validation_n,
                )
            else:
                eval_results = train_evaluate.evaluate_loop(
                    train_evaluate.fishing_new_test_dataloader,
                    train_evaluate.fishing_new_test_n,
                )
        else:
            if validation:
                eval_results = train_evaluate.evaluate_loop(
                    train_evaluate.validation_dataloader, train_evaluate.validation_n
                )
            else:
                eval_results = train_evaluate.evaluate_loop(
                    train_evaluate.test_dataloader, train_evaluate.test_n
                )

        # Setup a data frame for the trajectory level data
        data = pd.DataFrame(
            {
                "Data set Index": eval_results[8],
                "Index": eval_results[6],
                "MMSI": eval_results[7],
                "Ship type": eval_results[5],
                "Length": eval_results[4],
                "Reconstruction log probability": eval_results[3],
            }
        )
        data["Equally weighted reconstruction log probability"] = (
            data["Reconstruction log probability"] / data["Length"]
        )
        if setup_type is not None:
            data["Setup type"] = setup_type
        return {
            "EquallyWeightedMeanLoss": eval_results[0],
            "EquallyWeightedMeanKLDivergence": eval_results[1],
            "EquallyWeightedMeanReconstructionLogProbability": eval_results[2],
            "TrajectoryLevelData": data,
            "TrainEvaluateObject": train_evaluate,
        }

    def run_evaluation_get_n(
        self,
        n=6,
        worst=True,
        vessel_type="Any",
        equally_weighted=True,
        validation=True,
        setup_type=None,
        fishing_file=None,
        fishing_new_file=None,
    ):
        """Run evaluation loop and return the best/worst ones

        Parameters
        ----------
        n : int (Defaults to 6)
            How many trajectories to return

        worst : bool (Defaults to None)
            When True, the worst reconstructions are returned.
            When False, the best reconstructions are returned

        vessel_type : str (Defaults to 'Any')
            Possibility to list which vessel type to focus on (one of 'Any'/'Fishing'/'Cargo'/'Tanker')

        equally_weighted : bool (Defaults to True)
            When True, the equally weighted reconstruction log probability are used.
            When False, the reconstruction log probability are used.

        validation : bool (Defaults to True)
            When True, the validation DataLoader is used, but otherwise the test loader is used

        setup_type : str (Defaults to None)
            Model setup used. This will be the value in the 'Setup type' column

        fishing_file : str (Defaults to None)
            Name of the main part of the fishing vessel only data file for the same ROI and dates as in file_name.
            When not None, the evaluation will be done using this data set

        fishing_new_file : str (Defaults to None)
            Name of the main part of a new fishing vessel only data file with a different ROI or dates (or both).
            When not None, the evaluation will be done using this data set. In case fishing_file and fishing_new_file
            are both not None evaluation will be done on the fishing_file

        Returns
        -------
        dict
            Dictionary containing the evaluation information
        """
        # Run the evaluation on the entire data set
        data = self.run_evaluation(
            validation=validation,
            setup_type=setup_type,
            fishing_file=fishing_file,
            fishing_new_file=fishing_new_file,
        )
        df = data["TrajectoryLevelData"]
        if vessel_type != "Any":
            df = df.loc[df["Ship type"] == vessel_type]

        if not equally_weighted:
            df = df.sort_values("Reconstruction log probability", ascending=worst)
        else:
            df = df.sort_values(
                "Equally weighted reconstruction log probability", ascending=worst
            )
        return {
            "TrajectoryLevelData": df.head(n),
            "TrainEvaluateObject": data["TrainEvaluateObject"],
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
        bins="auto",
        title=None,
        print_summary_stats=False,
        stat=None,
        weights=None,
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

        bins : str, number, vector, or a pair of such values
            decides on the numbe of histogram bins

        title : str (Defaults to None)
            The figure title

        print_summary_stats : bool (Defaults to False)
            When True, summary statistics will also be printed

        stat : str (Defaults to None)
            Aggregate statistic to compute in each bin

        weights : list (Defaults to None)
            If provided, weight the contribution of the corresponding data points towards the count in each bin by these factors
        """
        sns.set_theme(style="whitegrid")
        sns.set_context(
            "paper", rc={"lines.linewidth": 3.0}, font_scale=self.font_scale
        )
        style.use("seaborn-colorblind")

        if type == "Histogram":
            if stat is None:
                stat = "count"
            ax = sns.histplot(
                x=x, bins=bins, hue=hue, hue_order=hue_order, data=data, stat=stat
            )

        elif type == "Stacked":
            if stat is None:
                stat = "proportion"
            elif stat == "normalized_largest":
                # Get the number of bins and the max count (to normalize by the largest count)
                hist_info = np.histogram(data[x], bins=bins)
                weights = [1 / max(hist_info[0])] * data.shape[0]
                bins = len(hist_info[0])
                stat = "count"
            elif stat == "normalized_each_bin":
                # Get the weights to use to normalize each bin
                hist_info = np.histogram(data[x], bins=bins)
                bins = hist_info[1]
                bin_weights = [
                    100 / weight if weight != 0 else weight for weight in hist_info[0]
                ]
                weights = []
                for index, row in data.iterrows():
                    hist_info_i = np.histogram(row[x], bins=bins)
                    weights.append(bin_weights[np.where(hist_info_i[0] == 1)[0][0]])
                stat = "count"
                bins = len(hist_info[0])

            ax = sns.histplot(
                x=x,
                bins=bins,
                hue=hue,
                multiple="stack",
                hue_order=hue_order,
                data=data,
                stat=stat,
                weights=weights,
            )
        else:
            print("Currently only implmented for 'Histogram' and 'Stacked'")

        file_path = None
        if file_name is not None:
            file_path = self.learning_curve_dir / (file_name + ".pdf")
        utils.add_plot_extras(
            ax,
            self.save_figures,
            self.plot_figures,
            file_path,
            xlabel,
            ylabel,
            xlim,
            ylim,
            title,
        )
        if print_summary_stats:
            print(
                f"Mean {x}: {data[x].mean()}",
                f"Median {x}: {data[x].median()}",
                f"{x} statistics by ship type: ",
                data.groupby("Ship type")[x].describe(),
                sep="\n",
            )

    def reconstruct(
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
        bins="auto",
        title=None,
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

        bins : str, number, vector, or a pair of such values
            decides on the numbe of histogram bins

        title : str (Defaults to None)
            The figure title

        print_summary_stats : bool
            When True, summary statistics will also be printed
        """
        sns.set_theme(style="whitegrid")
        sns.set_context(
            "paper", rc={"lines.linewidth": 3.0}, font_scale=self.font_scale
        )
        style.use("seaborn-colorblind")

        if type == "Histogram":
            ax = sns.histplot(x=x, bins=bins, hue=hue, hue_order=hue_order, data=data)

        elif type == "Stacked":
            ax = sns.histplot(
                x=x,
                bins=bins,
                hue=hue,
                multiple="stack",
                hue_order=hue_order,
                data=data,
            )
        else:
            print("Currently only implmented for 'Histogram' and 'Stacked'")

        file_path = None
        if file_name is not None:
            file_path = self.learning_curve_dir / (file_name + ".pdf")
        utils.add_plot_extras(
            ax,
            self.save_figures,
            self.plot_figures,
            file_path,
            xlabel,
            ylabel,
            xlim,
            ylim,
            title,
        )
        if print_summary_stats:
            print(
                f"Mean {x}: {data[x].mean()}",
                f"Median {x}: {data[x].median()}",
                f"{x} statistics by ship type: ",
                data.groupby("Ship type")[x].describe(),
                sep="\n",
            )
