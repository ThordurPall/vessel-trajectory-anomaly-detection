import pickle

import numpy as np
import torch

from utils import plotting


def computeLoss(log_px, log_pz, log_qz, lengths, beta=1):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_len = len(log_px)
    curmask = (
        torch.arange(max_len, device=device)[:, None] < lengths[None, :]
    )  # max_seq_len X Batch

    log_px = torch.stack(log_px, dim=0) * curmask
    log_px = log_px.sum(dim=0)  # Sum over time

    log_pz = torch.stack(log_pz, dim=0) * curmask
    log_qz = torch.stack(log_qz, dim=0) * curmask
    kl = log_qz.sum(dim=0) - log_pz.sum(dim=0)  # Sum over time

    loss = log_px - beta * kl  # recon loss - beta_kl
    loss = torch.mean(loss / lengths)  # mean over batch

    return -loss, log_px, kl, curmask


def train_VRNN(
    num_epochs,
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    kl_weight,
    kl_anneling_start,
    modelName,
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testset = test_loader.dataset
    train_n = len(train_loader.dataset)
    test_n = len(testset)

    loss_tot = []
    kl_tot = []
    recon_tot = []
    val_loss_tot = []
    val_kl_tot = []
    val_recon_tot = []

    beta_weight = kl_anneling_start
    kl_weight_step = abs(kl_weight - kl_anneling_start) / (
        10 * len(train_loader)
    )  # Annealing over 10 epochs.

    for epoch in range(1, num_epochs + 1):
        # Begin training loop
        loss_epoch = 0
        kl_epoch = 0
        recon_epoch = 0
        model.train()
        for i, (_, _, _, lengths, inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            (
                log_px,
                log_pz,
                log_qz,
                _,
                _,
                _,
                _,
                _,
            ) = model(inputs, targets, logits=None)

            loss, log_px, kl, _ = computeLoss(
                log_px, log_pz, log_qz, lengths, beta=beta_weight
            )

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Anneal beta
            beta_weight += kl_weight_step
            beta_weight = min(beta_weight, kl_weight)

            loss_epoch += loss.item() * len(lengths)
            kl_epoch += torch.sum(kl / lengths).item()
            recon_epoch += torch.sum(log_px / lengths).item()

        loss_tot.append(loss_epoch / train_n)
        kl_tot.append(kl_epoch / train_n)
        recon_tot.append(recon_epoch / train_n)

        # Begin validation loop
        val_loss = 0
        val_kl = 0
        val_recon = 0
        model.eval()
        for i, (_, _, _, lengths, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            log_px, log_pz, log_qz, _, _, _, _, _ = model(inputs, targets, logits=None)

            loss, log_px, kl, _ = computeLoss(
                log_px, log_pz, log_qz, lengths, beta=beta_weight
            )

            val_loss += loss.item() * len(lengths)
            val_kl += torch.sum(kl / lengths).item()
            val_recon += torch.sum(log_px / lengths).item()

        val_loss_tot.append(val_loss / test_n)
        val_kl_tot.append(val_kl / test_n)
        val_recon_tot.append(val_recon / test_n)

        scheduler.step()

        datapoints = np.random.choice(test_n, size=3, replace=False)
        plotting.make_vae_plots(
            (loss_tot, kl_tot, recon_tot, val_loss_tot, val_kl_tot, val_recon_tot),
            model,
            datapoints,
            testset,
            testset.params["binedges"],
            device,
            figurename=modelName,
        )

        print(
            "Epoch {} of {} finished. Trainingloss = {}. Validationloss = {}".format(
                epoch, num_epochs, loss_epoch / train_n, val_loss / test_n
            )
        )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                "models/saved_models/" + modelName + "_" + str(epoch) + ".pth",
            )

            trainingCurves = {
                "loss_tot": loss_tot,
                "kl_tot": kl_tot,
                "recon_tot": recon_tot,
                "val_loss_tot": val_loss_tot,
                "val_kl_tot": val_kl_tot,
                "val_recon_tot": val_recon_tot,
            }
            with open("models/saved_models/" + modelName + "_curves.pkl", "wb") as f:
                pickle.dump(trainingCurves, f)

    trainingCurves = {
        "loss_tot": loss_tot,
        "kl_tot": kl_tot,
        "recon_tot": recon_tot,
        "val_loss_tot": val_loss_tot,
        "val_kl_tot": val_kl_tot,
        "val_recon_tot": val_recon_tot,
    }

    torch.save(model.state_dict(), "models/" + modelName + ".pth")
    with open("models/" + modelName + "_curves.pkl", "wb") as f:
        pickle.dump(trainingCurves, f)

    return model
