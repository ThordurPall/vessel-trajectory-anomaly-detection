import matplotlib.pyplot as plt


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
