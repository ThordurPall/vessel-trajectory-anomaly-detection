from math import cos, pi
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import requests
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def findcenters(edges):
    lat_edges, lon_edges, speed_edges, course_edges = edges

    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1

    lat_centers = [
        round((lat_edges[i] + lat_edges[i + 1]) / 2, 3)
        for i in range(len(lat_edges) - 1)
    ]
    lon_centers = [
        round((lon_edges[i] + lon_edges[i + 1]) / 2, 3)
        for i in range(len(lon_edges) - 1)
    ]
    speed_centers = [
        round((speed_edges[i] + speed_edges[i + 1]) / 2, 3)
        for i in range(len(speed_edges) - 1)
    ]
    course_centers = [
        round((course_edges[i] + course_edges[i + 1]) / 2, 3)
        for i in range(len(course_edges) - 1)
    ]

    return lat_centers, lon_centers, speed_centers, course_centers


def get_static_map_bounds(lat, lng, zoom, sx, sy):
    # lat, lng - center
    # sx, sy - map size in pixels

    # 256 pixels - initial map size for zoom factor 0
    sz = 256 * 2 ** zoom

    # resolution in degrees per pixel
    res_lat = cos(lat * pi / 180.0) * 360.0 / sz
    res_lng = 360.0 / sz

    d_lat = res_lat * sy / 2
    d_lng = res_lng * sx / 2

    return ((lat - d_lat, lng - d_lng), (lat + d_lat, lng + d_lng))


def createStaticMap(edges, api_key, file_path, zoom=8):
    # zoom defines the zoom level of the map. YOU MIGHT NEED TO MODIFY THIS DEPENDING ON THE ROI
    # edges = (LAT_EDGES, LON_EDGES, SOG_EDGES, COG_EDGES)
    # LAT_EDGES, LON_EDGES, ... are a list of bin edges obtained by np.arange(LAT_MIN, LAT_MAX+(LAT_RES/10000), LAT_RES) ect.
    # I use (config.LAT_EDGES, config.LON_EDGES, config.SOG_EDGES, config.COG_EDGES)

    lat_centers, lon_centers, speed_centers, course_centers = findcenters(edges)

    lat_center = lat_centers[int(len(lat_centers) / 2)]
    lon_center = lon_centers[int(len(lon_centers) / 2)]

    # Enter your api key here. Make a google cloud account and get a maps API. You might need to put in payment infomation but you get a free number of requests each month which should be enough
    # api_key =

    # url variable store url
    url = "https://maps.googleapis.com/maps/api/staticmap?"

    # center defines the center of the map,
    # equidistant from all edges of the map.
    center = (
        str(lat_center) + "," + str(lon_center)
    )  # {latitude,longitude} pair (e.g. "40.714728,-73.998672")

    url = (
        url
        + "center="
        + center
        + "&zoom="
        + str(zoom)
        + "&size=640x640&key="
        + api_key
        + "&sensor=false&maptype=terrain"
    )

    # get method of requests module
    # return response object
    r = requests.get(url)
    print(url)

    # wb mode is stand for write binary mode
    # f = open("plots/" + filename + ".png", "wb")
    f = open(file_path, "wb")

    # r.content gives content,
    # in this case gives image
    f.write(r.content)

    # close method of file object
    # save and close the file
    f.close()

    SW_corner, NE_corner = get_static_map_bounds(lat_center, lon_center, zoom, 640, 640)
    lat_min, lon_min = SW_corner
    lat_max, lon_max = NE_corner


def getPositionalBoundaries(edges, zoom=8):

    lat_centers, lon_centers, speed_centers, course_centers = findcenters(edges)

    lat_center = lat_centers[int(len(lat_centers) / 2)]
    lon_center = lon_centers[int(len(lon_centers) / 2)]

    SW_corner, NE_corner = get_static_map_bounds(lat_center, lon_center, zoom, 640, 640)
    lat_min, lon_min = SW_corner
    lat_max, lon_max = NE_corner

    return lat_min, lat_max, lon_min, lon_max


def PlotTrack(encodedTrack, edges, ax, color=None, lsty="solid", print_=False):
    # Plots a four hot encoded track on the axis ax

    seq_len, data_dim = encodedTrack.shape
    lat_edges, lon_edges, speed_edges, course_edges = edges

    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1

    lat_centers = [
        round((lat_edges[i] + lat_edges[i + 1]) / 2, 3)
        for i in range(len(lat_edges) - 1)
    ]
    lon_centers = [
        round((lon_edges[i] + lon_edges[i + 1]) / 2, 3)
        for i in range(len(lon_edges) - 1)
    ]
    speed_centers = [
        round((speed_edges[i] + speed_edges[i + 1]) / 2, 3)
        for i in range(len(speed_edges) - 1)
    ]
    course_centers = [
        round((course_edges[i] + course_edges[i + 1]) / 2, 3)
        for i in range(len(course_edges) - 1)
    ]

    lat = np.zeros((seq_len))
    lon = np.zeros((seq_len))
    speed = np.zeros((seq_len))
    course = np.zeros((seq_len))

    for i in range(seq_len):
        lat[i] = lat_centers[np.argmax(encodedTrack[i, 0:lat_dim])]
        lon[i] = lon_centers[np.argmax(encodedTrack[i, lat_dim : (lat_dim + lon_dim)])]
        speed[i] = speed_centers[
            np.argmax(
                encodedTrack[i, (lat_dim + lon_dim) : (lat_dim + lon_dim + speed_dim)]
            )
        ]
        course[i] = course_centers[
            np.argmax(
                encodedTrack[
                    i,
                    (lat_dim + lon_dim + speed_dim) : (
                        lat_dim + lon_dim + speed_dim + course_dim
                    ),
                ]
            )
        ]

    if print_:
        print(lat)
        print(lon)

    points = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = plt.get_cmap("inferno")  # Black is start, yellow is end
    if color is None:
        colors = [cmap(float(ii) / (seq_len - 1)) for ii in range(seq_len - 1)]
    else:
        colors = [color] * (seq_len - 1)

    for ii in range(2, seq_len - 1):
        segii = segments[ii]
        (lii,) = ax.plot(segii[:, 0], segii[:, 1], color=colors[ii], linestyle=lsty)

        lii.set_solid_capstyle("round")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    return ax


def plotDataset(dataset, ax, edges, n=5000):
    # Plots n tracks from the data set on axis ax
    xlist = []
    ylist = []
    for i in progressbar.progressbar(range(0, n)):
        index, mmsi, time_stamps, ship_type, track_length, inputs, track = dataset[i]
        lon, lat, speed, course = PlotDatasetTrack(track, edges)
        xlist.extend(lon)
        ylist.extend(lat)
        xlist.append(None)
        ylist.append(None)

    ax.plot(xlist, ylist, "gray")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    return ax


def PlotDatasetTrack(encodedTrack, edges):
    # Returns the lat and lon of a four hot encoded track

    seq_len, data_dim = encodedTrack.shape
    lat_edges, lon_edges, speed_edges, course_edges = edges

    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1

    lat_centers = [
        round((lat_edges[i] + lat_edges[i + 1]) / 2, 3)
        for i in range(len(lat_edges) - 1)
    ]
    lon_centers = [
        round((lon_edges[i] + lon_edges[i + 1]) / 2, 3)
        for i in range(len(lon_edges) - 1)
    ]
    speed_centers = [
        round((speed_edges[i] + speed_edges[i + 1]) / 2, 3)
        for i in range(len(speed_edges) - 1)
    ]
    course_centers = [
        round((course_edges[i] + course_edges[i + 1]) / 2, 3)
        for i in range(len(course_edges) - 1)
    ]

    lat = np.zeros((seq_len))
    lon = np.zeros((seq_len))
    speed = np.zeros((seq_len))
    course = np.zeros((seq_len))

    for i in range(seq_len):
        lat[i] = lat_centers[np.argmax(encodedTrack[i, 0:lat_dim])]
        lon[i] = lon_centers[np.argmax(encodedTrack[i, lat_dim : (lat_dim + lon_dim)])]
        speed[i] = speed_centers[
            np.argmax(
                encodedTrack[i, (lat_dim + lon_dim) : (lat_dim + lon_dim + speed_dim)]
            )
        ]
        course[i] = course_centers[
            np.argmax(
                encodedTrack[
                    i,
                    (lat_dim + lon_dim + speed_dim) : (
                        lat_dim + lon_dim + speed_dim + course_dim
                    ),
                ]
            )
        ]
    # return lon, lat
    return lon, lat, speed, course


def Plot4HotEncodedTrack(encodedTrack, edges, ax=None):

    seq_len, data_dim = encodedTrack.shape
    lat_edges, lon_edges, speed_edges, course_edges = edges

    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1

    lat_centers = [
        round((lat_edges[i] + lat_edges[i + 1]) / 2, 3)
        for i in range(len(lat_edges) - 1)
    ]
    lon_centers = [
        round((lon_edges[i] + lon_edges[i + 1]) / 2, 3)
        for i in range(len(lon_edges) - 1)
    ]
    speed_centers = [
        round((speed_edges[i] + speed_edges[i + 1]) / 2, 3)
        for i in range(len(speed_edges) - 1)
    ]
    course_centers = [
        round((course_edges[i] + course_edges[i + 1]) / 2, 3)
        for i in range(len(course_edges) - 1)
    ]

    lat = np.zeros((seq_len))
    lon = np.zeros((seq_len))
    speed = np.zeros((seq_len))
    course = np.zeros((seq_len))

    for i in range(seq_len):
        lat[i] = lat_centers[np.argmax(encodedTrack[i, 0:lat_dim])]
        lon[i] = lon_centers[np.argmax(encodedTrack[i, lat_dim : (lat_dim + lon_dim)])]
        speed[i] = speed_centers[
            np.argmax(
                encodedTrack[i, (lat_dim + lon_dim) : (lat_dim + lon_dim + speed_dim)]
            )
        ]
        course[i] = course_centers[
            np.argmax(
                encodedTrack[
                    i,
                    (lat_dim + lon_dim + speed_dim) : (
                        lat_dim + lon_dim + speed_dim + course_dim
                    ),
                ]
            )
        ]

    points = np.array([lon, lat, speed]).T.reshape(-1, 1, 3)
    segments_speed = np.concatenate([points[:-1], points[1:]], axis=1)
    points = np.array([lon, lat, [0] * seq_len]).T.reshape(-1, 1, 3)
    segments_0 = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = plt.get_cmap("viridis")  # Blue is start, yellow is end
    colors = [cmap(float(ii) / (seq_len - 1)) for ii in range(seq_len - 1)]

    if ax == None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        for ii in range(2, seq_len - 1):
            segii = segments_speed[ii]
            (lii,) = ax.plot(segii[:, 0], segii[:, 1], segii[:, 2], color=colors[ii])
            segii = segments_0[ii]
            (lii,) = ax.plot(
                segii[:, 0], segii[:, 1], segii[:, 2], "--", color=colors[ii]
            )

            lii.set_solid_capstyle("round")

        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_zlabel("speed m/s")
    else:
        for ii in range(2, seq_len - 1):
            segii = segments_speed[ii]
            (lii,) = ax.plot(segii[:, 0], segii[:, 1], segii[:, 2], color=colors[ii])
            segii = segments_0[ii]
            (lii,) = ax.plot(
                segii[:, 0], segii[:, 1], segii[:, 2], "--", color=colors[ii]
            )

            lii.set_solid_capstyle("round")

        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_zlabel("speed m/s")

    return ax


def logitToTrack(logits, edges):
    lat_edges, lon_edges, speed_edges, course_edges = edges

    logits = logits.squeeze()

    seq_len = logits.shape[0]

    lat_dim = len(lat_edges) - 1
    lon_dim = len(lon_edges) - 1
    speed_dim = len(speed_edges) - 1
    course_dim = len(course_edges) - 1

    recon = torch.zeros(seq_len, lat_dim + lon_dim + speed_dim + course_dim)
    for t in range(seq_len):
        lat_idx = torch.argmax(logits[t, 0:lat_dim])
        lon_idx = torch.argmax(logits[t, lat_dim : (lat_dim + lon_dim)])
        speed_idx = torch.argmax(
            logits[t, (lat_dim + lon_dim) : (lat_dim + lon_dim + speed_dim)]
        )
        course_idx = torch.argmax(
            logits[
                t,
                (lat_dim + lon_dim + speed_dim) : (
                    lat_dim + lon_dim + speed_dim + course_dim
                ),
            ]
        )

        recon[t, lat_idx] = 1
        recon[t, lat_dim + lon_idx] = 1
        recon[t, lat_dim + lon_dim + speed_idx] = 1
        recon[t, lat_dim + lon_dim + speed_dim + course_idx] = 1

    return recon


def plot_recon(datapoint, binedges, model, device):

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    _, _, _, _, _, length, datainput, datatarget = datapoint
    datadim = datainput.shape[1]
    datainput = datainput.to(device)
    datatarget_gpu = datatarget.to(device)

    logits = torch.zeros(length.int().item(), 1, datadim, device=device)
    _, _, _, logits, _, _, _, _ = model(
        datainput.unsqueeze(0), datatarget_gpu.unsqueeze(0), logits=logits
    )

    logits = logits.cpu()
    recon = logitToTrack(logits, binedges)

    ax[0].remove()
    ax[0] = fig.add_subplot(1, 2, 1, projection="3d")
    Plot4HotEncodedTrack(datatarget, binedges, ax[0])

    ax[1].remove()
    ax[1] = fig.add_subplot(1, 2, 2, projection="3d")
    Plot4HotEncodedTrack(recon, binedges, ax[1])


def make_vae_plots(
    losses,
    model,
    datapoints,
    validationdata,
    binedges,
    device,
    figure_path,
    savefig=True,
):

    loss_tot, kl_tot, recon_tot, val_loss_tot, val_kl_tot, val_recon_tot = losses

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))

    ax[0, 0].plot(loss_tot, label="Training Loss")
    ax[0, 0].plot(val_loss_tot, label="Validation Loss")
    ax[0, 0].set_title("Loss")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].legend()

    ax[0, 1].plot(kl_tot, label="Training KL-divergence")
    ax[0, 1].plot(val_kl_tot, label="Validation KL-divergence")
    ax[0, 1].set_title("KL divergence")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].legend()

    ax[0, 2].plot(recon_tot, label="Training Reconstruction")
    ax[0, 2].plot(val_recon_tot, label="Validation Reconstruction")
    ax[0, 2].set_title("Reconstruction probability log_prob(x)")
    ax[0, 2].set_xlabel("Epoch")
    ax[0, 2].legend()

    for i, idx in enumerate(datapoints):
        _, _, _, _, _, length, datainput, datatarget = validationdata[idx]
        datainput = datainput.to(device)
        datatarget_gpu = datatarget.to(device)

        logits = torch.zeros(
            length.int().item(), 1, validationdata.data_dim, device=device
        )
        _, _, _, logits, _, _, _, _ = model(
            datainput.unsqueeze(0), datatarget_gpu.unsqueeze(0), logits=logits
        )

        logits = logits.cpu()
        recon = logitToTrack(logits, binedges)

        ax[1, i].remove()
        ax[1, i] = fig.add_subplot(3, 3, 4 + i, projection="3d")
        Plot4HotEncodedTrack(datatarget, binedges, ax[1, i])

        ax[2, i].remove()
        ax[2, i] = fig.add_subplot(3, 3, 7 + i, projection="3d")
        Plot4HotEncodedTrack(recon, binedges, ax[2, i])

    if savefig:
        plt.savefig(figure_path)
        plt.close()
    else:
        plt.show()


def plot_KL_evolution(kullbackLeiblerDivergence, title="", figurename="", savefig=True):

    # kullbackLeiblerDivergence Epoch X latent matrix
    fig = plt.figure(figsize=(7, 7))
    ax = sns.heatmap(-np.sort(-kullbackLeiblerDivergence, axis=-1), cmap="Greys")
    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Epoch")

    if savefig:
        plt.savefig("plots/KLdivergence" + figurename + ".png")
        plt.close()
    else:
        plt.show()


def plot_latents(ax, z, labels, classnames):
    z = z.to("cpu")
    palette = sns.color_palette()
    z = TSNE(n_components=2).fit_transform(z)

    for class_ in np.unique(labels):
        ix = np.where(labels == class_)
        color = np.expand_dims(np.array(palette[class_]), axis=0)
        ax.scatter(z[ix, 0], z[ix, 1], c=color, label=classnames[class_])

    ax.legend()
