import numpy as np


class Config:
    """
    A class used for configuration management in the project

    ...

    Attributes
    ----------
    config : dict
        A Python dict specifying key-value config pairs

    Methods
    -------
    get_property(property)
        Gets the a property of the config dictionary
    """

    def __init__(self):
        self.config = {}

        # Define for which navigational status the vessel is static and when it is moving
        self.config["STAT_NAV_STATUSES"] = ["At anchor", "Moored", "Aground"]
        self.config["MOV_NAV_STATUSES"] = [
            "Under way using engine",
            "Restricted maneuverability",
            "Constrained by her draught",
            "Not under command",
            "Engaged in fishing",
            "Under way sailing",
            "Reserved for future use [11]",
            "Reserved for future use [12]",
            "Reserved for future use [13]",
            "Reserved for future amendment [HSC]",
            "Reserved for future amendment [WIG]",
        ]

        # Define ship type mapping to start of file name
        self.config["SHIPTYPE_ANTIPOLUTTION"] = "Anti"
        self.config["SHIPTYPE_FISHING"] = "Fish"
        self.config["SHIPTYPE_TOWING"] = "Towi"
        self.config["SHIPTYPE_TUG"] = "Tug"
        self.config["SHIPTYPE_DREDGING"] = "Dred"
        self.config["SHIPTYPE_DIVING"] = "Divi"
        self.config["SHIPTYPE_HIGHSPEEDCRAFT"] = "HSC"
        self.config["SHIPTYPE_LAWENFORCEMENT"] = "Law"
        self.config["SHIPTYPE_MILITARY"] = "Mili"
        self.config["SHIPTYPE_OTHER"] = "Othe"
        self.config["SHIPTYPE_SAILING"] = "Sail"
        self.config["SHIPTYPE_PLEASURE"] = "Plea"
        self.config["SHIPTYPE_PILOT"] = "Pilo"
        self.config["SHIPTYPE_PASSENGER"] = "Pass"
        self.config["SHIPTYPE_PORTTENDER"] = "Port"
        self.config["SHIPTYPE_CARGO"] = "Carg"
        self.config["SHIPTYPE_RESERVED"] = "Rese"
        self.config["SHIPTYPE_SEARCHANDRESCUE"] = "SAR"
        self.config["SHIPTYPE_SEARCHANDRESCUE"] = "Spar"
        self.config["SHIPTYPE_TANKER"] = "Tank"
        self.config["SHIPTYPE_WINGINGROUND"] = "WIG"
        self.config["SHIPTYPE_UNKNOWN"] = "xxxx"

        # Define the min and max values for the regions of interest (ROIs)
        """
        Setup with no restrictions (to get the entire dataset): region = "All"
        """
        region = "All"
        self.config["LAT_MIN_" + region] = -90.0
        self.config["LAT_MAX_" + region] = 90.0
        self.config["LON_MIN_" + region] = -180.0
        self.config["LON_MAX_" + region] = 180.0

        # Set the resolution to use
        self.config["LAT_RES_" + region] = 0.01
        self.config["LON_RES_" + region] = 0.01
        self.config["SOG_RES_" + region] = 0.5
        self.config["COG_RES_" + region] = 5

        # Set a maximum number for the speed-over-ground (SOG)
        self.config["SOG_MAX_" + region] = 9999

        # Split the ROI, speed, and course into bins and define the edges of those
        self.config["LAT_EDGES_" + region] = self._calculate_edges("LAT", region, 2)
        self.config["LON_EDGES_" + region] = self._calculate_edges("LON", region, 2)
        self.config["SOG_EDGES_" + region] = self._calculate_edges("SOG", region, 1)
        self.config["COG_EDGES_" + region] = self._calculate_edges("COG", region, 0)

        # Define the default Google Maps zoom level of the map
        self.config["ZOOM_" + region] = 2

        """
        Setup with no restrictions (to get the entire dataset): region = "Denmark"
        """
        region = "Denmark"
        self.config["LAT_MIN_" + region] = 54.0
        self.config["LAT_MAX_" + region] = 60.0
        self.config["LON_MIN_" + region] = 5.5
        self.config["LON_MAX_" + region] = 16.5

        # Set the resolution to use
        self.config["LAT_RES_" + region] = 0.01
        self.config["LON_RES_" + region] = 0.01
        self.config["SOG_RES_" + region] = 0.5
        self.config["COG_RES_" + region] = 5

        # Set a maximum number for the speed-over-ground (SOG)
        self.config["SOG_MAX_" + region] = 9999

        # Split the ROI, speed, and course into bins and define the edges of those
        self.config["LAT_EDGES_" + region] = self._calculate_edges("LAT", region, 2)
        self.config["LON_EDGES_" + region] = self._calculate_edges("LON", region, 2)
        self.config["SOG_EDGES_" + region] = self._calculate_edges("SOG", region, 1)
        self.config["COG_EDGES_" + region] = self._calculate_edges("COG", region, 0)

        # Define the default Google Maps zoom level of the map
        self.config["ZOOM_" + region] = 6

        """
        Setup with no restrictions (to get the entire dataset): region = "Skagen"
        """
        region = "Skagen"
        self.config["LAT_MIN_" + region] = 55.5
        self.config["LAT_MAX_" + region] = 59.5
        self.config["LON_MIN_" + region] = 5.5
        self.config["LON_MAX_" + region] = 12.5

        # Set the resolution to use
        self.config["LAT_RES_" + region] = 0.01
        self.config["LON_RES_" + region] = 0.01
        self.config["SOG_RES_" + region] = 0.5
        self.config["COG_RES_" + region] = 5

        # Set a maximum number for the speed-over-ground (SOG)
        self.config["SOG_MAX_" + region] = 9999

        # Split the ROI, speed, and course into bins and define the edges of those
        self.config["LAT_EDGES_" + region] = self._calculate_edges("LAT", region, 2)
        self.config["LON_EDGES_" + region] = self._calculate_edges("LON", region, 2)
        self.config["SOG_EDGES_" + region] = self._calculate_edges("SOG", region, 1)
        self.config["COG_EDGES_" + region] = self._calculate_edges("COG", region, 0)

        # Define the default Google Maps zoom level of the map
        self.config["ZOOM_" + region] = 7

        """
        Setup for region A: region = "Bornholm"
        """
        region = "Bornholm"
        self.config["LAT_MIN_" + region] = 54.5
        self.config["LAT_MAX_" + region] = 56
        self.config["LON_MIN_" + region] = 13
        self.config["LON_MAX_" + region] = 16

        # Set the resolution to use
        self.config["LAT_RES_" + region] = 0.01
        self.config["LON_RES_" + region] = 0.01
        self.config["SOG_RES_" + region] = 0.5
        self.config["COG_RES_" + region] = 5

        # Set a maximum number for the speed-over-ground (SOG)
        self.config["SOG_MAX_" + region] = 15.5

        # Split the ROI, speed, and course into bins and define the edges of those
        self.config["LAT_EDGES_" + region] = self._calculate_edges("LAT", region, 2)
        self.config["LON_EDGES_" + region] = self._calculate_edges("LON", region, 2)
        self.config["SOG_EDGES_" + region] = self._calculate_edges("SOG", region, 1)
        self.config["COG_EDGES_" + region] = self._calculate_edges("COG", region, 0)

        # Define the Google Maps zoom level of the map
        self.config["ZOOM_" + region] = 8

    def get_property(self, property):
        """Gets the a property of the config dictionary

        Parameters
        ----------
        property : str
            A string that is the key into the config dictionary
        """
        return self.config.get(property)

    def _calculate_edges(self, attribute, region, decimals):
        """Calculates the edges for the given attribute

        Parameters
        ----------
        attribute : str
            A string that one of "LAT"/"LON"/"SOG"/"COG"

        region : str
            One of the specified regions in the config file (like "Bornholm")

        decimals : int
            Number of decimal places to round to
        """
        start = 0  # Default to zero for SOG and COG
        if attribute in ["LAT", "LON"]:
            start = self.get_property(attribute + "_MIN_" + region)

        attribute_max = 360  # Default to 360 for COG
        if attribute != "COG":
            attribute_max = self.get_property(attribute + "_MAX_" + region)
        step = self.get_property(attribute + "_RES_" + region)
        stop = attribute_max + (step / 10000)
        return np.around(np.arange(start, stop, step), decimals=decimals)


class ROISCOGConfig(Config):
    """
    A class used for configs related to ROI, SOG, and COG

    ...

    Attributes
    ----------
    region : str
            One of the specified regions in the config (like "Bornholm")

    Methods
    -------
    roi()
        Gets the set of four min & max lat & lon numbers (min lat, max lat, min lon, max lon)

    binedges()
        Gets the set of bin edges (for lat, lon, SOG, COG edges)

    SOG_MAX()
        Gets the maximum speed-over-ground (SOG) to consider
    """

    def __init__(self, region):
        """
        Parameters
        ----------
        ship_types_included : list
            List of ship type strings to include in the data set

        region : str
            One of the specified regions in the config (like "Bornholm")
        """
        super().__init__()
        self.region = "_" + region

    @property
    def roi(self):
        return (
            self.get_property("LAT_MIN" + self.region),
            self.get_property("LAT_MAX" + self.region),
            self.get_property("LON_MIN" + self.region),
            self.get_property("LON_MAX" + self.region),
        )

    @property
    def binedges(self):
        return (
            self.get_property("LAT_EDGES" + self.region),
            self.get_property("LON_EDGES" + self.region),
            self.get_property("SOG_EDGES" + self.region),
            self.get_property("COG_EDGES" + self.region),
        )

    @property
    def SOG_MAX(self):
        return self.get_property("SOG_MAX" + self.region)

    @property
    def ZOOM(self):
        return self.get_property("ZOOM" + self.region)
