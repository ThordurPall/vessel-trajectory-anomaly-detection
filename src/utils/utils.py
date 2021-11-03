import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Utils file for useful functions used throughout the code base
def add_plot_extras(
    ax,
    save_figure,
    plot_figure,
    file_name_path=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
):
    """Add the requested plot extras

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Pre-existing axes for the plot

    save_figure : bool
        Whether or not to save all created figures by default

    plot_figure : bool
        Whether or not to plot all created figures by default

    file_name_path : pathlib.WindowsPath (Defaults to None)
        File path where the figure will be saved

    xlabel : str (Defaults to None)
        x label text to put on the plot

    ylabel : str (Defaults to None)
        y label text to put on the plot

    xlim : list (Defaults to None)
        Limit of the x-axis

    ylim : list (Defaults to None)
        Limit of the y-axis
    """
    # Add labels when requested
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Change axis limits when requested
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # Check whether to save or plot the figure
    if save_figure & (file_name_path is not None):
        plt.savefig(file_name_path, bbox_inches="tight")
    if plot_figure:
        plt.show()


# Define a function to read in the training/validation curves for a specified setup
def curves_df(
    setup_type,
    data_processed,
    model_dir,
    model="VRNN",
    latent_dim="100",
    recurrent_dim="100",
    batch_norm=False,
    scheduler=False,
    kl_annealing=False,
):
    # Figure out the .csv file name for the current model setup
    BatchNorm = "_batchNormTrue" if batch_norm else "_batchNormFalse"
    Scheduler = "_SchedulerTrue" if scheduler else ""
    KLAnneal = "_KLTrue" if kl_annealing else ""
    curves_file = (
        model
        + "_"
        + data_processed
        + "_latent"
        + latent_dim
        + "_recurrent"
        + recurrent_dim
        + BatchNorm
        + Scheduler
        + KLAnneal
        + "_curves.csv"
    )

    # Read in the training/validation learning curves for the current setup
    df = pd.read_csv(model_dir / curves_file)
    df_train = df.iloc[:, 0:3].copy()
    df_train.columns = ["Loss", "KL divergence", "Reconstruction log probabilities"]
    df_train["Data set type"] = "Training"
    df_train["Epoch"] = df_train.index + 1

    df_val = df.iloc[:, 3:6].copy()
    df_val.columns = ["Loss", "KL divergence", "Reconstruction log probabilities"]
    df_val["Data set type"] = "Validation"
    df_val["Epoch"] = df_train.index + 1
    df = pd.concat([df_train, df_val])
    df["Setup type"] = setup_type
    df.reset_index(drop=True, inplace=True)
    return df


# Define a function to plot the loss, KL divergence, and Reconstruction log probabilities side by side
def plot_curves(
    df,
    hue="Data set type",
    hue_order=["Training", "Validation"],
    title=None,
    xlims=None,
    ylims=None,
):
    _, ax = plt.subplots(1, 3, figsize=(18, 5))
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
    plt.show()


# Define a function to plot the loss, KL divergence, and Reconstruction log probabilities one at a time (large)
def plot_curves_large(
    df, hue="Data set type", hue_order=["Training", "Validation"], title=None
):
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
        x=x, y="Reconstruction log probabilities", hue=hue, hue_order=hue_order, data=df
    )
    sns.despine()
    if title is not None:
        ax.set_title(title)
