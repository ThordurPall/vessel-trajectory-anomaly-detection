from pathlib import Path
from src.Config import ROISCOGConfig
import src.utils.utils as utils
import src.utils.plotting as plotting
import logging
import matplotlib.pyplot as plt, matplotlib.style as style, matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import progressbar


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

    plot_single_track(df_lon_lat, ax, use_cmap, color, plot_start, plot_end)
        Plots a single vessel trajectory on an axis ax

    read_visualise_static_map()
        Gets an already created static Google Maps map for the ROI and plots it

    read_map_plot_single_track(df_lon_lat, use_cmap, color, plot_start, plot_end, s)
        Reads the created static map and plots a single vessel trajectory
    """

    def __init__(self, region, save_figures, plot_figures, fig_size, zoom=None):
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
        """

        super().__init__()
        self.region = region
        self.save_figures = save_figures
        self.plot_figures = plot_figures
        self.fig_size = fig_size

        # Setup the correct foldure structure
        project_dir = Path(__file__).resolve().parents[2]
        self.processed_data_dir = project_dir / "data" / "processed"
        self.trajectories_fig_dir = project_dir / "figures" / "regions" / self.region

        # Make sure that the trajectory figures path exists
        self.trajectories_fig_dir.mkdir(parents=True, exist_ok=True)

        # Use seaborn style defaults and set the default figure size
        sns.set(rc={"figure.figsize": self.fig_size})
        sns.set_theme(style="whitegrid")
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

    def visualise_static_map(self, img):
        """Visualises a static Google Maps map

        Returns
        ----------
        matplotlib.axes.Axes
            The axes for the plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.imshow(img, extent=self.bounds, aspect="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return fig, ax

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
        """
        logger = logging.getLogger(__name__)  # For logging information

        # Define the colors to use
        n = len(df_lon_lat.index)
        cmap = plt.cm.cividis  # Blue in the beginning and yellow in the end
        colors = [cmap(float(i) / (n - 1)) for i in range(n - 1)]

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
            for i in progressbar.progressbar(range(n - 1)):
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
            plt.scatter(
                x=df_lon_lat.loc[df_lon_lat.index[[0, -1]], "Longitude"],
                y=df_lon_lat.loc[df_lon_lat.index[[0, -1]], "Latitude"],
                color=[colors[0], colors[-1]],
                s=s,
            )

        elif plot_start & ~plot_end:
            plt.scatter(
                x=df_lon_lat.loc[df_lon_lat.index[0], "Longitude"],
                y=df_lon_lat.loc[df_lon_lat.index[0], "Latitude"],
                color=colors[0],
                s=s,
            )
        elif ~plot_start & plot_end:
            plt.scatter(
                x=df_lon_lat.loc[df_lon_lat.index[-1], "Longitude"],
                y=df_lon_lat.loc[df_lon_lat.index[-1], "Latitude"],
                color=colors[-1],
                s=s,
            )

        # Check whether to save or plot the figure
        if self.save_figures:
            plt.savefig(
                self.trajectories_fig_dir
                / (self.region + "_single_trajectory_zoom_" + str(self.zoom) + ".pdf"),
                bbox_inches="tight",
            )
        if self.plot_figures:
            plt.show()

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
        self, df_lon_lat, use_cmap=False, color=None, plot_start=True, plot_end=True
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
        """
        logger = logging.getLogger(__name__)  # For logging information
        logger.info(
            "Calling self.read_visualise_static_map to get and visualises the map"
        )
        _, _, _, ax = self.read_visualise_static_map()
        logger.info("Calling self.plot_single_track to plot a single track on the map")
        self.plot_single_track(df_lon_lat, ax, use_cmap, color, plot_start, plot_end)

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
                x, y, bins=bins, cmap=plt.cm.cividis, cmin=10, data=data, vmax=vmax
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
                mincnt=10,
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