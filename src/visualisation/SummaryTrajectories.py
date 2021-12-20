from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import seaborn as sns

import src.utils.utils as utils


class SummaryTrajectories:
    """
    A class used to handle everything related to visualising and providing
    a summary of a trajectory data set

    ...

    Attributes
    ----------
    df : pandas.DataFrame
        Processed trajectories for exploratory analysis

    df_date_type : pandas.DataFrame
        Daily trajectory counts grouped by ShipType

    processed_data_dir : pathlib.WindowsPath
        Directory where the processed data is be located

    explore_fig_dir : pathlib.WindowsPath
        Directory where the summary explorartory figures should be stored

    save_figures : bool
        Whether or not to save all created figures by default

    plot_figures : bool
        Whether or not to plot all created figures by default

    fig_size : tuple
        Default figure size to use for visualisations


    Methods
    -------
    basic_statistics(resample_frequency):
        Prints some basic summary statistics

    get_longest_trajectories(self, n, keep_cols):
        Returns the longest n trajectories

    hist_bar_plot(data, type, x, y, file_name, xlabel, ylabel, hue, hue_order, xlim, ylim)
        Creates a histogram or bar plot

    time_series_line_plot(df, type, x, y, file_name, xlabel, ylabel, xlim, ylim, include_points)
        Creates a time series line plot

    box_plot(df, x, y, file_name, xlabel, ylabel, xlim, ylim, vessel_types)
        Creates a box plot
    """

    def __init__(
        self,
        summary_file,
        save_figures,
        plot_figures,
        fig_size,
        date="DateTime",
        font_scale=1,
    ):
        """
        Parameters
        ----------
        summary_file: str
            File path to the processed summary results

        save_figures : bool
            Whether or not to save all created figures by default

        plot_figures : bool
            Whether or not to plot all created figures by default

        fig_size : tuple
            Default figure size to use for visualizations

        date : str
            Which date should be the main (default) date for plotting

        font_scale : int (Defaults to 1)
            The font size to use while plotting
        """

        super().__init__()
        self.save_figures = save_figures
        self.plot_figures = plot_figures
        self.fig_size = fig_size
        self.date = date

        # Setup the correct foldure structure
        self.project_dir = Path(__file__).resolve().parents[2]
        self.processed_data_dir = self.project_dir / "data" / "processed"
        self.explore_fig_dir = self.project_dir / "figures" / "exploration" / "summary"

        # Make sure that the summary exploration figures path exists
        self.explore_fig_dir.mkdir(parents=True, exist_ok=True)

        # Use seaborn style defaults and set the default figure size
        sns.set(rc={"figure.figsize": self.fig_size})
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", rc={"lines.linewidth": 3.0}, font_scale=font_scale)
        style.use("seaborn-colorblind")

        # Read in the processed trajectories for exploratory analysis
        self.df = pd.read_csv(self.processed_data_dir / summary_file)
        self.df["DateTimeMean"] = self.df["DateTime"]
        self.df["DateTime"] = pd.to_datetime(self.df[self.date])

        # For some basic time series analysis based on daily trajectory count grouped by ShipType
        self.df_date_type = (
            self.df.groupby([self.df["DateTime"].dt.date, "ShipType"])
            .size()
            .reset_index(name="counts")
        )
        self.df_date_type = pd.pivot_table(
            self.df_date_type,
            values="counts",
            index="DateTime",
            columns="ShipType",
            fill_value=0,
        )
        self.df_date_type.index = pd.to_datetime(self.df_date_type.index)
        self.df_date_type["Month"] = self.df_date_type.index.month
        self.df_date_type["Weekday Name"] = self.df_date_type.index.strftime("%A")

        # Let the time stamp be the data frame index and add useful date related columns
        self.df = self.df.set_index("DateTime")
        self.df.index = pd.to_datetime(self.df.index)
        self.df["Month"] = self.df.index.month
        self.df["Weekday Name"] = self.df.index.strftime("%A")
        self.df["Day of Month"] = self.df.index.day

    def basic_statistics(self, resample_frequency):
        """Prints some basic summary statistics

        Parameters
        ----------
        resample_frequency : int
            Time between samples in seconds

        """
        print(
            f"Number of Trajectories: {self.df.shape[0]}",
            f"Number of AIS message updates: {self.df['TrackLength'].sum()}",
            f"Number of unique MMSIs: {self.df['MMSI'].nunique()}",
            "",
            f"Number of unique MMSIs by {self.df.groupby('ShipType')['MMSI'].nunique()}",
            "",
            f"Mean speed by {self.df.groupby('ShipType')['MeanSpeed'].mean()}",
            "",
            "Number of trajectories by ShipType",
            self.df["ShipType"].value_counts(),
            "",
            "Summary statistics for track length in minutes: ",
            (self.df["TrackLength"] * resample_frequency / 60).describe(),
            "",
            sep="\n",
        )

        if self.date == "DateTime":
            max_diff = max(
                pd.to_datetime(self.df.index) - pd.to_datetime(self.df["DateTimeStart"])
            )
            print(
                f"Max difference between mean datetime and start datetime: {max_diff}"
            )

    def get_longest_trajectories(self, n, keep_cols=None):
        """Returns the longest n trajectories

        Parameters
        ----------
        n : int
            Number of trajectories to return

        keep_cols : list (Defaults to None)
            When set to None, all columns are returned. When a list is given
            only the column names specified in the list are returned

        Returns
        -------
        pandas.DataFrame
            Data frame containing the n longest trajectories
        """
        df = self.df.sort_values("TrackLength").tail(n)

        if keep_cols is not None:
            df = df[keep_cols]
        return df

    def hist_bar_plot(
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
        df_line=None,
        col_order=None,
        dist_x=None,
        dist_y=None,
        stat="count",
    ):
        """Creates a histogram or bar plot

        Parameters
        ----------
        data : int
            Data set to plot

        type : str
            Type of figure to plot (either "Histogram" or "Bar")

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

        df_line : pandas.DataFrame (Defaults to None)
            Possible x and y values for adding a line to the figure

        col_order : list (Defaults to None)
            The color order to use when plotting
        """

        if type == "Histogram":
            ax = sns.histplot(x=x, hue=hue, data=data, hue_order=hue_order, stat=stat)

        elif type == "Bar":
            palette = None
            if col_order is not None:
                palette = [sns.color_palette()[i] for i in col_order]

            ax = sns.barplot(
                x=x,
                y=y,
                data=data,
                palette=palette,
            )
            sns.despine()
        else:
            print("Currently only implmented for 'Histogram' and 'Bar'")

        if df_line is not None:
            ax.plot(df_line["x"], df_line["y"], linewidth=3, color="black")

        if dist_x is not None and dist_y is not None:
            ax.plot(dist_x, dist_y, linewidth=3, color="black")

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

    def time_series_line_plot(
        self,
        df,
        type,
        x=None,
        y=None,
        file_name=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        include_points=False,
    ):
        """Creates a time series line plot

        Parameters
        ----------
        df : pandas.DataFrame
            Time series data to plot

        type : str
            Either 'Pandas' or 'Seaborn'

        x : str (Defaults to None)
            Variable to plot on the x-axis

        y : str (Defaults to None)
            Variable to plot on the y-axis

        file name : str (Defaults to None)
            File name where the figure will be saved

        xlabel : str (Defaults to None)
            x label text to put on the plot

        ylabel : str (Defaults to None)
            y label text to put on the plot

        xlim : list (Defaults to None)
            Limit of the x-axis

        ylim : list (Defaults to None)
            Limit of the y-axis

        include_points : bool (Defaults to False)
            When true incldues weekday name points on the line plot
        """
        sns.set_style("ticks")

        # Add weekday name points on the line plot when requested
        if include_points:
            hue_order = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
            ax = sns.scatterplot(
                x=x, y=y, data=df, hue="Weekday Name", s=300, hue_order=hue_order
            )

        if type == "Pandas":
            ax = df.plot()

        elif type == "Seaborn":
            ax = sns.lineplot(x=x, y=y, data=df)
            sns.despine()
        else:
            print("Currently only implmented for 'Pandas' and 'Seaborn'")
        sns.despine()

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

    def box_plot(
        self,
        df,
        x,
        y=None,
        file_name=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        vessel_types=["Cargo", "Tanker", "Fishing"],
    ):
        """Creates a box plot

        Parameters
        ----------
        df : pandas.DataFrame
            Time series data to plot

        x : str
            Variable to plot on the x-axis

        y : str (Defaults to None)
            Variable to plot on the y-axis

        file name : str (Defaults to None)
            File name where the figure will be saved

        xlabel : str (Defaults to None)
            x label text to put on the plot

        ylabel : str (Defaults to None)
            y label text to put on the plot

        xlim : list (Defaults to None)
            Limit of the x-axis

        ylim : list (Defaults to None)
            Limit of the y-axis

        vessel_types : list (Defaults to ["Cargo", "Tanker", "Fishing"])
            Vessel types to
        """
        sns.set_style("whitegrid")
        y_included = y is not None

        n = len(vessel_types)
        _, axes = plt.subplots(n, 1, figsize=self.fig_size, sharex=True)
        for type, ax in zip(vessel_types, axes):
            data = df
            if not y_included:
                y = type
            else:
                data = df.loc[df["ShipType"] == type]

            sns.boxplot(data=data, x=x, y=y, ax=ax)
            ax.set_title(type)
            ax.set_ylabel(ylabel)
            if ax != axes[-1]:
                ax.set_xlabel("")

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
