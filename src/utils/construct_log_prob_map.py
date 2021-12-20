import numpy as np
import pandas as pd
import torch

import src.utils.dataset_utils as dataset_utils


def run_VRNN(model, train_loader):
    # Setup variables
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_n = len(train_loader.dataset)
    maxLength = train_loader.dataset.max_length
    latent_dim = model.latent_shape
    recurrent_dim = model.recurrent_shape
    batch_size = train_loader.batch_size
    trainset = train_loader.dataset

    # Initialize variables
    zmus = np.zeros((maxLength, train_n, latent_dim))
    hs = np.zeros((maxLength, train_n, recurrent_dim))
    activatedBins = np.zeros((maxLength, train_n, 4))
    recon_loss = np.zeros((maxLength, train_n))
    lengths_tot = np.zeros((train_n))  # Keep track of all the lenghts
    j = 0

    # Run through the entire  trainign set
    for i, (
        data_set_indices,
        file_location_indices,
        mmsis,
        time_stamps,
        ship_types,
        lengths,
        inputs,
        targets,
    ) in enumerate(train_loader):
        max_len = int(torch.max(lengths).item())  # Max length for this batch

        endIndex = (
            (batch_size * (i + 1)) if (batch_size * (i + 1)) <= train_n else train_n
        )  # The index this batch goes up to (not included)
        lengths_tot[(batch_size * i) : endIndex] = lengths.numpy()  # Track all lengths

        # Find out which bins are activated at each time points - Use the actual values
        for target, length in zip(targets, lengths):
            if target.shape[1] != 4:  # Four-hot encoded targets (and inputs)
                # Get the activated (non zero) bins up to the current tracks length
                # There is one bin for each variable so reshape to four
                activatedBins[: int(length), j, :] = (
                    torch.nonzero(target[: int(length), :], as_tuple=True)[1]
                    .reshape(int(length), 4)
                    .numpy()
                )  # activated bins in target
            else:  # Four-dimensional targets (and inputs)
                # Four hot encode the current trajectory to find the activated bins
                df = pd.DataFrame(
                    {
                        "lat": target[:, 0],
                        "lon": target[:, 1],
                        "speed": target[:, 2],
                        "course": target[:, 3],
                    }
                )
                encoded_track = torch.Tensor(
                    dataset_utils.FourHotEncode(df, trainset.data_info["binedges"])
                )

                # Get the activated (non zero) bins up to the current tracks length
                # There is one bin for each variable so reshape to four
                activatedBins[: int(length), j, :] = (
                    torch.nonzero(encoded_track[: int(length), :], as_tuple=True)[1]
                    .reshape(int(length), 4)
                    .numpy()
                )  # activated bins in target
            j += 1
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        # Keep track of the approximate posterior distributions mean and hidden states
        means = torch.zeros(max_len, len(lengths), latent_dim, device=device)
        hidden_states = torch.zeros(max_len, len(lengths), recurrent_dim, device=device)

        # Run the VRNN model
        log_px, _, _, _, hidden_states, _, means, _ = model(
            inputs, targets, logits=None, z_mus=means, hs=hidden_states
        )

        # Calculate the temporal mask for this batch size - Dimension: max_len X batch_size
        curmask = torch.arange(max_len, device=device)[:, None] < lengths[None, :]

        # log_px is really an array of reconstruction log probabilities (batch size many for each time point)
        # Stack that to a matrix where the first dimension is the time and multiply mask.
        # The mask will zero out reconstructions at time that goes over the trajcetory length.
        # That is, do not use the reconstructions for lengths past the corresponding track length
        log_px = torch.stack(log_px, dim=0) * curmask

        # Store the current batch reconstruction, posterior mean, and hidden states
        recon_loss[:max_len, (batch_size * i) : endIndex] = (
            log_px.detach().cpu().numpy()
        )
        zmus[:max_len, (batch_size * i) : endIndex, :] = means.detach().cpu().numpy()
        hs[:max_len, (batch_size * i) : endIndex, :] = (
            hidden_states.detach().cpu().numpy()
        )
    return recon_loss, activatedBins, zmus, hs, lengths_tot.astype(int)


def constuct_logprob_map(model, train_loader):
    # Run the VRNN model for the training set
    recon_loss, activatedBins, zmus, hs, lengths = run_VRNN(model, train_loader)

    # Split recon_loss into originating geographical bin
    trainset = train_loader.dataset
    lat_dim = len(trainset.data_info["binedges"][0]) - 1
    lon_dim = len(trainset.data_info["binedges"][1]) - 1
    map_logprob = dict()
    for row in range(lat_dim):
        for col in range(lon_dim):
            # Make a grid of all possible lat, lon bin combinations
            map_logprob[str(row) + "," + str(col)] = []

    for i, trackLength in enumerate(lengths):  # Go through all tracks in training data
        for t in range(trackLength):  # Go through each time point for current track
            # Get the geographical bin the AIS point is in
            activatedlat = activatedBins[t, i, 0]
            activatedlon = activatedBins[t, i, 1] - lat_dim

            # Place the reconstruction log probability for this AIS point in the correct cell
            map_logprob[str(int(activatedlat)) + "," + str(int(activatedlon))].append(
                recon_loss[t, i]
            )
    return map_logprob, zmus, hs, lengths
