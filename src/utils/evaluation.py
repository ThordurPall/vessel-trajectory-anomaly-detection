import numpy as np
import pandas as pd
import torch

import src.utils.dataset_utils as dataset_utils


def evaluate_VRNN(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_n = len(test_loader.dataset)
    maxLength = test_loader.dataset.max_length
    latent_dim = model.latent_shape
    recurrent_dim = model.recurrent_shape
    batch_size = test_loader.batch_size
    testset = test_loader.dataset

    zmus = np.zeros((maxLength, test_n, latent_dim))
    hs = np.zeros((maxLength, test_n, recurrent_dim))
    activatedBins = np.zeros((maxLength, test_n, 4))
    recon_loss = np.zeros((maxLength, test_n))
    j = 0
    for i, (
        data_set_indices,
        file_location_indices,
        mmsis,
        time_stamps,
        ship_types,
        lengths,
        inputs,
        targets,
    ) in enumerate(test_loader):
        max_len = int(torch.max(lengths).item())

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
                    dataset_utils.FourHotEncode(df, testset.data_info["binedges"])
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

        means = torch.zeros(max_len, len(lengths), latent_dim, device=device)
        hidden_states = torch.zeros(max_len, len(lengths), recurrent_dim, device=device)

        log_px, _, _, _, hidden_states, _, means, _ = model(
            inputs, targets, logits=None, z_mus=means, hs=hidden_states
        )

        curmask = (
            torch.arange(max_len, device=device)[:, None] < lengths[None, :]
        )  # max_seq_len X Batch

        log_px = torch.stack(log_px, dim=0) * curmask  # max_seq_len X Batch

        endIndex = (
            (batch_size * (i + 1)) if (batch_size * (i + 1)) <= test_n else test_n
        )

        recon_loss[:max_len, (batch_size * i) : endIndex] = (
            log_px.detach().cpu().numpy()
        )
        zmus[:max_len, (batch_size * i) : endIndex, :] = means.detach().cpu().numpy()
        hs[:max_len, (batch_size * i) : endIndex, :] = (
            hidden_states.detach().cpu().numpy()
        )
    return recon_loss, activatedBins, zmus, hs
