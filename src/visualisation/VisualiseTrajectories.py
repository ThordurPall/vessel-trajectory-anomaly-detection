import logging
import pickle
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import progressbar
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.utils.plotting as plotting
import src.utils.utils as utils
from src.Config import ROISCOGConfig


class VisualiseTrajectories:
    """
    A class used to handle everything related to geographically
    visualising trajectories

    ...

    Attributes
    ----------
    processed_data_dir : pathlib.WindowsPath
        Directory where the processed data is be located

    trajectories_fig_dir : pathlib.WindowsPath
        Directory where the trajectory figures should be stored

    region : str
        One of the specified regions in the config file (like "Bornholm")

    continuous_representation : bool
        Either continuous (True) or discrete (False) AIS trajectory representation

    save_figures : bool
        Whether or not to save all created figures by default

    plot_figures : bool
        Whether or not to plot all created figures by default

    fig_size : tuple
        Default figure size to use for visualisations

    zoom : int (Defauls to None)
        The zoom to use for the static Google Maps map for the ROI.
        When no zoom is given, the zoom from the config file is used

    binedges : tuple
        The lat/lon bin edges for this region defined in the config file

    bounds : list
        The [lon_min, lon_max, lat_min, lat_max] static map boundaries

    Methods
    -------
    create_static_map(api_key)
        Creates a static Google Maps map of the ROI

    read_static_map():
        Gets an already created static Google Maps map for the ROI

    remove_points_outside_ROI(df, lat_lon_names)
        Removes points outside the min/max lon/lat interval

    visualise_static_map(img)
        Visualises a static Google Maps map

    plot_single_track(df_lon_lat, ax, use_cmap, color, plot_start, plot_end, progress_bar, fig)
        Plots a single vessel trajectory on an axis ax

    plot_multiple_tracks(ax, indicies, data_path, df_lon_lat, use_cmap, color, plot_start, plot_end, s, progress_bar, fig)
        Plots multiple vessel trajectory on an axis ax

    read_visualise_static_map()
        Gets an already created static Google Maps map for the ROI and plots it

    read_map_plot_single_track(df_lon_lat, use_cmap, color, plot_start, plot_end, s, progress_bar)
        Reads the created static map and plots a single vessel trajectory
    """

    def __init__(
        self,
        region,
        save_figures,
        plot_figures,
        fig_size,
        zoom=None,
        continuous_representation=True,
    ):
        """
        Parameters
        ----------
        region : str
            One of the predefined regions in the config file (like "Bornholm")

        save_figures : bool
            Whether or not to save all created figures by default

        plot_figures : bool
            Whether or not to plot all created figures by default

        fig_size : tuple
            Default figure size to use for visualizations

        zoom : int (Defauls to None)
            The zoom to use for the static Google Maps map for the ROI.
            When no zoom is given, the zoom from the config file is used

        continuous_representation : bool (Defaults to True)
            Either continuous or discrete AIS trajectory representation
        """

        super().__init__()
        self.region = region
        self.save_figures = save_figures
        self.plot_figures = plot_figures
        self.fig_size = fig_size
        self.continuous_representation = continuous_representation

        # Setup the correct foldure structure
        self.project_dir = Path(__file__).resolve().parents[2]
        self.processed_data_dir = self.project_dir / "data" / "processed"
        self.trajectories_fig_dir = (
            self.project_dir / "figures" / "regions" / self.region
        )

        # Make sure that the trajectory figures path exists
        self.trajectories_fig_dir.mkdir(parents=True, exist_ok=True)

        # Use seaborn style defaults and set the default figure size
        sns.set(rc={"figure.figsize": self.fig_size})
        # sns.set_theme(style="whitegrid")
        sns.set_theme(style="ticks")
        sns.set_context(
            "paper", rc={"lines.linewidth": 3.0}
        )  # One of paper, notebook, talk, and poster
        style.use("seaborn-colorblind")

        # Set values from the region config files
        ROISCOG_config = ROISCOGConfig(self.region)  # ROI, SOG, and COG configurations
        self.binedges = ROISCOG_config.binedges
        self.zoom = ROISCOG_config.ZOOM
        if zoom is not None:
            self.zoom = zoom

        # Get the [lon_min, lon_max, lat_min, lat_max] map boundaries
        lat_min, lat_max, lon_min, lon_max = plotting.getPositionalBoundaries(
            self.binedges, self.zoom
        )
        self.bounds = [lon_min, lon_max, lat_min, lat_max]

    def create_static_map(
        self,
        api_key,
    ):
        """Creates a static Google Maps map of the ROI

        Parameters
        ----------
        api_key : str
            Google Maps api key string

        Returns
        -------
        str
            Returns the static map file name
        """
        logger = logging.getLogger(__name__)  # For logging information
        logger.info("Calling plotting.createStaticMap to create a static map")
        result_file_name = self.region + "_static_map_zoom_" + str(self.zoom) + ".png"
        plotting.createStaticMap(
            self.binedges,
            api_key,
            self.trajectories_fig_dir / result_file_name,
            self.zoom,
        )
        logger.info("Main part of static map results file name: " + result_file_name)
        return result_file_name

    def read_static_map(self):
        """Gets an already created static Google Maps map for the ROI

        Returns
        ----------
        numpy.ndarray
            The mpimg static Google Maps map image of the ROI
        """
        # Read the Google Maps image of the ROI
        static_map_file = self.region + "_static_map_zoom_" + str(self.zoom) + ".png"
        return mpimg.imread(self.trajectories_fig_dir / static_map_file)

    def visualise_static_map(self, img, subplots=[1, 1]):
        """Visualises a static Google Maps map

        Returns
        ----------
        matplotlib.axes.Axes
            The axes for the plot
        """
        fig, axs = plt.subplots(
            nrows=subplots[0], ncols=subplots[1], figsize=self.fig_size
        )
        if sum(subplots) > 2:
            for ax in axs:
                ax.imshow(img, extent=self.bounds, aspect="auto")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
        else:
            axs.imshow(img, extent=self.bounds, aspect="auto")
            axs.set_xlabel("Longitude")
            axs.set_ylabel("Latitude")
        return fig, axs

    def remove_points_outside_ROI(self, df, lat_lon_names=["Latitude", "Longitude"]):
        """Remove points outside the min/max lon/lat interval

        Returns
        ----------
        df_lon_lat : pandas.DataFrame
            Data frame containing only points inside the ROI
        """
        lon_min, lon_max, lat_min, lat_max = self.bounds
        return df[
            df[lat_lon_names[0]].between(lat_min, lat_max)
            & df[lat_lon_names[1]].between(lon_min, lon_max)
        ]

    def plot_single_track(
        self,
        df_lon_lat,
        ax,
        use_cmap=False,
        color=None,
        plot_start=True,
        plot_end=True,
        s=100,
        progress_bar=False,
        return_ax=False,
        df_speed=None,
        df_course=None,
        fig=None,
        rect=None,
        rect2=None,
    ):
        """Plots a single vessel trajectory on an axis ax

        The axis object ax usually comes from using imshow on a static Google Maps map of the ROI

        Parameters
        ----------
        df_lon_lat : pandas.DataFrame
            Data frame of the longitudes and latitudes to plot

        ax : matplotlib.axes.Axes
            Pre-existing axes for the plot

        use_cmap : bool (defaults to False)
            When False, uses a single color for the entire trajectory. Uses the color
            variable if it is specified, but otherwise a default color is used.
            When True, a color map will be used, such that the trajectory is
            blue in the beginning and yellow in the end.

        color : str (Defaults to None)
            Defines the color to use when plotting the trajectory

        plot_start : bool (Defaults to True)
            When True, the start of trajectory is plotted as a point on top of the figure

        plot_end : bool (Defaults to True)
            When True, the end of trajectory is plotted as a point on top of the figure

        s: float or array-like (Defaults to 100)

        progress_bar : bool (Defaults to False)
            When True, a progressbar.progressbar will be used when plotting points

        return_ax : bool (Defaults to False)
            Return the axis plotted on when True

        df_speed : pandas.DataFrame (Defaults to None)
            Data frame of actual and reconstructed speed. When not None,
            the speed is plotted in the top left corner

        df_course : pandas.DataFrame (Defaults to None)
            Data frame of actual and reconstructed course

        fig : Figure (Defaults to None)
            The figure instance to use

        rect : list (Defaults to None)
            The dimensions [left, bottom, width, height] of the new Axes. All quantities are in fractions of figure width and height

        rect2 : list (Defaults to None)
            The dimensions [left, bottom, width, height] of the new Axes. All quantities are in fractions of figure width and height
        """
        logger = logging.getLogger(__name__)  # For logging information

        # Define the colors to use
        df_lon_lat = df_lon_lat.copy()
        n = len(df_lon_lat.index)
        cmap = plt.cm.cividis  # Blue in the beginning and yellow in the end
        colors = [cmap(float(i) / (n - 1)) for i in range(n - 1)]

        # Ensure that no longitude or latitude values are exactly the same
        eps = [0.000000001 * i for i in range(n)]
        df_lon_lat["Longitude"] = df_lon_lat["Longitude"] + eps
        df_lon_lat["Latitude"] = df_lon_lat["Latitude"] + eps

        # Plot the trajectory on a map
        if not use_cmap:
            logger.info(
                "Plotting a line with the order that observations appear in the dataset"
            )
            if color is None:
                color = colors[0]

            sns.lineplot(
                x="Longitude",
                y="Latitude",
                sort=False,
                color=color,
                ax=ax,
                data=df_lon_lat,
            )
        else:
            logger.info(
                "Plot a single complete vessel trajectory one connection at a time"
            )
            if progress_bar:
                for i in progressbar.progressbar(
                    range(n - 1),
                ):
                    # Plot a line connecting the current row to the next one
                    sns.lineplot(
                        x=df_lon_lat.loc[df_lon_lat.index[[i, i + 1]], "Longitude"],
                        y=df_lon_lat.loc[df_lon_lat.index[[i, i + 1]], "Latitude"],
                        sort=False,
                        color=colors[i],
                        ax=ax,
                    )
            else:
                for i in range(n - 1):
                    # Plot a line connecting the current row to the next one
                    sns.lineplot(
                        x=df_lon_lat.loc[df_lon_lat.index[[i, i + 1]], "Longitude"],
                        y=df_lon_lat.loc[df_lon_lat.index[[i, i + 1]], "Latitude"],
                        sort=False,
                        color=colors[i],
                        ax=ax,
                    )

        # Check whether to plot the start and end of trajectory as a point
        if plot_start & plot_end:
            ax.scatter(
                x=df_lon_lat.loc[df_lon_lat.index[[0, -1]], "Longitude"],
                y=df_lon_lat.loc[df_lon_lat.index[[0, -1]], "Latitude"],
                color=[colors[0], colors[-1]],
                s=s,
            )

        elif plot_start & ~plot_end:
            ax.scatter(
                x=df_lon_lat.loc[df_lon_lat.index[0], "Longitude"],
                y=df_lon_lat.loc[df_lon_lat.index[0], "Latitude"],
                color=colors[0],
                s=s,
            )
        elif ~plot_start & plot_end:
            ax.scatter(
                x=df_lon_lat.loc[df_lon_lat.index[-1], "Longitude"],
                y=df_lon_lat.loc[df_lon_lat.index[-1], "Latitude"],
                color=colors[-1],
                s=s,
            )

        if df_speed is not None:
            # Plot speed in the top left corner
            new_ax = fig.add_axes(rect=rect)
            sns.lineplot(
                x="index",
                y="Speed",
                color=color,
                ax=new_ax,
                hue="Type",
                hue_order=["Actual", "Reconstructed"],
                data=df_speed,
            )
            new_ax.set(
                xlabel="Update number", ylabel="Speed (knots)"
            )  # , title="title")

        if df_course is not None:
            # Plot speed in the top left corner
            new_ax2 = fig.add_axes(rect=rect2)
            sns.lineplot(
                x="index",
                y="Course",
                color=color,
                ax=new_ax2,
                hue="Type",
                hue_order=["Actual", "Reconstructed"],
                data=df_course,
            )
            new_ax2.set(
                xlabel="Update number", ylabel="Course (degrees)"
            )  # , title="title")

        # Check whether to save or plot the figure
        if self.save_figures:
            plt.savefig(
                self.trajectories_fig_dir
                / (self.region + "_single_trajectory_zoom_" + str(self.zoom) + ".pdf"),
                bbox_inches="tight",
            )
        if self.plot_figures:
            plt.show()
        if return_ax:
            return ax

    def plot_multiple_tracks(
        self,
        ax,
        indicies=None,
        data_path=None,
        df=None,
        use_cmap=False,
        color=None,
        plot_start=True,
        plot_end=True,
        s=100,
        progress_bar=False,
        fig=None,
    ):
        """Plots multiple vessel trajectory on an axis ax

        The axis object ax typically comes from using imshow on a static Google Maps map of the ROI

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Pre-existing axes for the plot

        indicies : list (Defaults to None)
            The trajectory starting index position to plot

        data_path : pathlib.WindowsPath (Defaults to None)
            Location of the file to read. Required when indicies are given

        df : pandas.DataFrame (Defaults to None)
            Data frame of the longitudes and latitudes to plot and a unique
            trajectory id. Either indicies or df must be non None

        use_cmap : bool (defaults to False)
            When False, uses a single color for the entire trajectory. Uses the color
            variable if it is specified, but otherwise a default color is used.
            When True, a color map will be used, such that the trajectory is
            blue in the beginning and yellow in the end.

        color : str (Defaults to None)
            Defines the color to use when plotting the trajectory

        plot_start : bool (Defaults to True)
            When True, the start of trajectory is plotted as a point on top of the figure

        plot_end : bool (Defaults to True)
            When True, the end of trajectory is plotted as a point on top of the figure

        s: float or array-like (Defaults to 100)

        progress_bar : bool (Defaults to False)
            When True, a progressbar.progressbar will be used when plotting points

        fig : Figure (Defaults to None)
            The figure instance to use
        """
        # Keep track of trajectory information and turn off save/plot for now
        trajectories = []
        self.save_figures = False
        self.plot_figures = False
        save_tmp = self.save_figures
        plot_tmp = self.plot_figures

        if indicies is not None:
            # When only the indicies are given
            for idx in indicies[:-1]:
                df = utils.get_track_by_index(
                    data_path, idx, keep_cols=None, col_names=None
                )
                trajectories.append(
                    [
                        df["MMSI"][0],
                        df["Track length"][0],
                        df["Time stamp"][0],
                        df["Time stamp"].iloc[-1],
                    ]
                )
                df_lon_lat = utils.get_track_by_index(
                    data_path, idx, keep_cols=["Longitude", "Latitude"]
                )
                self.plot_single_track(
                    df_lon_lat,
                    ax,
                    use_cmap=use_cmap,
                    color=color,
                    plot_start=plot_start,
                    plot_end=plot_end,
                    s=s,
                    progress_bar=progress_bar,
                    fig=fig,
                )

            # Plot the last trajectory and save the results
            self.save_figures = save_tmp
            self.plot_figures = plot_tmp
            df = utils.get_track_by_index(data_path, indicies[-1])
            trajectories.append(
                [
                    df["MMSI"][0],
                    df["Track length"][0],
                    df["Time stamp"][0],
                    df["Time stamp"].iloc[-1],
                ]
            )
            df_lon_lat = utils.get_track_by_index(
                data_path,
                indicies[-1],
                keep_cols=["Longitude", "Latitude"],
            )
            self.plot_single_track(
                df_lon_lat,
                ax,
                use_cmap=use_cmap,
                color=color,
                plot_start=plot_start,
                plot_end=plot_end,
                s=s,
                progress_bar=progress_bar,
                fig=fig,
            )
            return pd.DataFrame(
                trajectories,
                columns=["MMSI", "Track length", "Date time start", "Date time end"],
            )
        elif df is not None:
            # When the actual data set to plot is given
            indicies = df["Index"].unique()
            for idx in indicies[:-1]:
                df_idx = df.loc[df["Index"] == idx]
                trajectories.append(
                    [
                        df_idx.iloc[0]["MMSI"],
                        df_idx.iloc[0]["Track length"],
                        df_idx.iloc[0]["Time stamp"],
                        df_idx.iloc[-1]["Time stamp"],
                    ]
                )
                self.plot_single_track(
                    df_idx,
                    ax,
                    use_cmap=use_cmap,
                    color=color,
                    plot_start=plot_start,
                    plot_end=plot_end,
                    s=s,
                    progress_bar=progress_bar,
                    fig=fig,
                )

            # Plot the last trajectory and save the results
            self.save_figures = save_tmp
            self.plot_figures = plot_tmp
            id = indicies[-1]
            df_idx = df.loc[df["Index"] == idx]
            trajectories.append(
                [
                    df_idx.iloc[0]["MMSI"],
                    df_idx.iloc[0]["Track length"],
                    df_idx.iloc[0]["Time stamp"],
                    df_idx.iloc[-1]["Time stamp"],
                ]
            )
            self.plot_single_track(
                df_idx,
                ax,
                use_cmap=use_cmap,
                color=color,
                plot_start=plot_start,
                plot_end=plot_end,
                s=s,
                progress_bar=progress_bar,
                fig=fig,
            )
            return pd.DataFrame(
                trajectories,
                columns=["MMSI", "Track length", "Date time start", "Date time end"],
            )
        else:
            print("Either indicies or df_lon_lat must be provided")
            self.save_figures = save_tmp
            self.plot_figures = plot_tmp

    def read_visualise_static_map(self):
        """Gets an already created static Google Maps map for the ROI and plots it

        Returns
        ----------
        numpy.ndarray
            The mpimg static Google Maps map image of the ROI
        matplotlib.axes.Axes
            The axes for the plot
        """
        logger = logging.getLogger(__name__)  # For logging information
        logger.info("Calling self.read_static_map to get the static map")
        img = self.read_static_map()
        logger.info("Calling self.visualise_static_map to visualises the map")
        fig, ax = self.visualise_static_map(img)
        return fig, img, ax

    def read_map_plot_single_track(
        self,
        df_lon_lat,
        use_cmap=False,
        color=None,
        plot_start=True,
        plot_end=True,
        progress_bar=False,
        return_ax=False,
    ):
        """Reads the created static map and plots a single vessel trajectory

        Parameters
        ----------
        df_lon_lat : pandas.DataFrame
            Data frame of the longitudes and latitudes to plot

        use_cmap : bool (defaults to False)
            When False, uses a single color for the entire trajectory. Uses the color
            variable if it is specified, but otherwise a default color is used.
            When True, a color map will be used, such that the trajectory is
            blue in the beginning and yellow in the end.

        color : str (Defaults to None)
            Defines the color to use when plotting the trajectory

        plot_start : bool (Defaults to True)
            When True, the start of trajectory is plotted as a point on top of the figure

        plot_end : bool (Defaults to True)
            When True, the end of trajectory is plotted as a point on top of the figure

        progress_bar : bool (Defaults to False)
            When True, a progressbar.progressbar will be used when plotting points

        return_ax : bool (Defaults to False)
            Return the axis plotted on when True

        """
        logger = logging.getLogger(__name__)  # For logging information
        logger.info(
            "Calling self.read_visualise_static_map to get and visualises the map"
        )
        fig, _, ax = self.read_visualise_static_map()
        logger.info("Calling self.plot_single_track to plot a single track on the map")
        self.plot_single_track(
            df_lon_lat,
            ax,
            use_cmap,
            color,
            plot_start,
            plot_end,
            progress_bar=progress_bar,
            fig=fig,
        )
        if return_ax:
            return ax

    def plot_points(
        self,
        data,
        type,
        x,
        y,
        file_name=None,
        xlabel=None,
        ylabel=None,
        hue=None,
        hue_order=None,
        style=None,
        xlim=None,
        ylim=None,
        alpha=None,
        bins=None,
        vmax=None,
        cb_label=None,
        cmin=10,
    ):
        """Creates a histogram or bar plot

        Parameters
        ----------
        data : int
            Data set to plot

        type : str
            One of 'Scatter', 'Hist', 'Hex'

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

        hue : str (Defaults to None)
            Variable that determines the color of plot elements

        hue_order : list (Defaults to None)
            Specify the order of processing and plotting for categorical levels of hue

        style : str (Defaults to None)
            Grouping variable that will produce points with different markers

        xlim : list (Defaults to None)
            Limit of the x-axis

        ylim : list (Defaults to None)
            Limit of the y-axis

        alpha : float (Defaults to None)
            Proportional opacity of the points

        bins : int (Defaults to None)
            Number of bins for the hist and gridsize for hex plot

        vmax : int (Defaults to None)
            Max value to anchor the colormap

        cb_label : str (Dfaults to None)
            Color bar label
        """
        logger = logging.getLogger(__name__)  # For logging information
        logger.info(
            "Calling self.read_visualise_static_map to get and visualises the map"
        )
        fig, _, ax = self.read_visualise_static_map()
        logger.info("Plotting on top of the static map")
        if type == "Scatter":
            sns.scatterplot(
                x=x,
                y=y,
                hue=hue,
                style=style,
                hue_order=hue_order,
                alpha=alpha,
                data=data,
            )
        elif type == "Hist":
            if bins is None:
                bins = 200
            h = ax.hist2d(
                x, y, bins=bins, cmap=plt.cm.cividis, cmin=cmin, data=data, vmax=vmax
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(h[3], cax=cax)
            cb.set_label(cb_label)

        elif type == "Hex":
            if bins is None:
                bins = 200
            h = ax.hexbin(
                x,
                y,
                gridsize=bins,
                cmap=plt.cm.cividis,
                mincnt=cmin,
                vmax=vmax,
                data=data,
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(h, cax=cax)
            cb.set_label(cb_label)
            # cax.grid()

        else:
            print("Currently only implmented for 'Scatter', 'Hist', and 'Hex'")

        ax.grid()
        file_path = None
        if file_name is not None:
            file_name = self.region + "_" + file_name
            if self.zoom is not None:
                file_name = file_name + "_zoom_" + str(self.zoom)
            file_name = file_name + ".pdf"
            file_path = self.trajectories_fig_dir / file_name
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
