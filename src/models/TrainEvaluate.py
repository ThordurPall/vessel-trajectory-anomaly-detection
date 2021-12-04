# Please note that some code in this class builds upon work done by Kristoffer Vinther Olesen (@DTU)
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import src.models.VRNN as VRNN
import src.utils.dataset_utils as dataset_utils
import src.utils.plotting as plotting
from src.data.Datasets import AISDataset


class TrainEvaluate:
    """
    A class that handles everything related to training and evaluating models in this project

    ...

    Attributes
    ----------
    model_dir : pathlib.WindowsPath
        Path to the project model directory

    model_intermediate_dir : pathlib.WindowsPath
        Path where the intermediate model results are stored

    train_mean : Tensor
        Training set feature mean values

    input_shape : int
        Dimension of the input feature space

    training_n : int
        Size of the training set

    validation_n : int
        Size of the validation set

    test_n : int
        Size of the test set

    training_dataloader : torch.utils.data.DataLoader
        Training set DataLoader

    validation_dataloader : torch.utils.data.DataLoader
        Validation set DataLoader

    test_dataloader : torch.utils.data.DataLoader
        Test set DataLoader

    model : src.models.VRNN
        The VRNN model (trained or to train)

    model_name : str
        String that indentifies the current model setup

    evaluate_on_fishing_vessles : bool
        When true, validation and testing will also be done on a data set
        containing only fishing vessel using the same dates and ROI

    evaluate_on_new_fishing_vessles : bool
        When true, validation and testing will also be done on a data set
        containing only fishing vessel using different dates and ROI

    fishing_validation_n : int
        Size of the fishing vessels only validation set

    fishing_new_validation_n : int
        Size of the new fishing vessels only validation set

    fishing_test_n : int
        Size of the fishing vessels only test set

    fishing_new_test_n : int
        Size of the new fishing vessels only test set

    fishing_validation_dataloader : torch.utils.data.DataLoader
        Fishing vessel only validation set DataLoader

    fishing_new_validation_dataloader : torch.utils.data.DataLoader
        New fishing vessel only validation set DataLoader

    fishing_test_dataloader : torch.utils.data.DataLoader
        Fishing vessel only test set DataLoader

    fishing_new_test_dataloader : torch.utils.data.DataLoader
        New fishing vessel only test set DataLoader

    generative_dist : str (Defaults to 'Bernoulli')
        The observation model to use

    GMM_components : int (Defaults to 4)
        The number of components to use as part of the GMM

    GMM_equally_weighted : bool (Defaults to True)
        When True, all GMM components are equally weighted

    Methods
    -------
    loss_function(log_px, log_pz, log_qz, lengths, beta)
        Computes the loss function

    train_loop(optimizer, beta_weight, kl_weight_step, kl_weight)
        The train Loop

    evaluate_loop(data_loader, data_n, beta_weight)
        The Validation/Test Loop

    train_VRNN(num_epochs, learning_rate, kl_weight, kl_anneling_start)
        Train (and validate with validation set) a deep learning VRNN model

    """

    def __init__(
        self,
        file_name,
        batch_size=32,
        num_workers=1,
        pin_memory=True,
        latent_dim=100,
        recurrent_dim=100,
        batch_norm=False,
        is_trained=False,
        fishing_file=None,
        fishing_new_file=None,
        inject_cargo_proportion=0.0,
        intermediate_epoch=None,
        generative_dist="Bernoulli",
        trained_model_name=None,
        GMM_components=4,
        GMM_equally_weighted=True,
    ):
        """
        Parameters
        ----------
        file_name : str
            Name of the main part of the file where the results are saved

        batch_size : int (Defaults to 32)
            Size of the batch of train features and targets retured in each DataLoader iteration

        num_workers : int (Defaults to 1)
            How many subprocesses to use for data loading (if 0 data will is loaded in the
            main process). If running on Windows and you get a BrokenPipeError, try setting to zero

        pin_memory : bool (Defaults tot True)
            Setting to True will automatically put a fetched data Tensors in pinned memory,
            and thus enables faster data transfer to CUDA-enabled GPUs (might need to set to False)

        latent_dim : int (Defaults to 100)
            Latent space size in the VRNN networks

        recurrent_dim : int (Defaults to 100)
            Recurrent latent space size in the VRNN networks

        batch_norm : bool (Defaults to False)
            When set to True, batch normalization is included in the networks

        is_trained : bool (Defaults to False)
            If True the model has already been trained and will be loaded from the .pth file

        fishing_file : str (Defaults to None)
            Name of the main part of the fishing vessel only data file for the same ROI and dates as in file_name

        fishing_new_file : str (Defaults to None)
            Name of the main part of a new fishing vessel only data file with a different ROI or dates (or both)

        inject_cargo_proportion : float (Defaults to 0.0)
            Inject additional cargo/tanker vessel trajectories in inject_cargo_proportion proportion to the training trajectories

        intermediate_epoch : int (Defaults to None)
            When not None, the intermediate model saved at epoch intermediate_epoch will be loaded

        generative_dist : str (Defaults to 'Bernoulli')
            The observation model to use

        trained_model_name : str (Defautls to None)
            Provides the ability to provide the complete name of the trained model

        GMM_components : int (Defaults to 4)
            The number of components to use as part of the GMM

        GMM_equally_weighted : bool (Defaults to True)
            When True, all GMM components are equally weighted
        """
        logger = logging.getLogger(__name__)  # For logging information
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using {} device".format(self.device))

        # Check what type of input feature representation to use
        self.generative_dist = generative_dist
        self.discrete = True if self.generative_dist == "Bernoulli" else False
        self.GMM_components = GMM_components
        self.GMM_equally_weighted = GMM_equally_weighted

        # Setup the correct foldure structure
        project_dir = Path(__file__).resolve().parents[2]
        self.model_dir = project_dir / "models" / "saved-models"
        self.model_intermediate_dir = self.model_dir / "intermediate"

        # Make sure that the model paths exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Start by preparing the data for training, validation, and testing using DataLoaders. They are basically
        # a fancy generator/iterator that abstract away all of the data handling and pre-processing
        # (and are useful for processing batches of data). That is, the custom Datasets retrieve the data set’s
        # features and labels one sample at a time, but while training a model the samples should be passed in minibatches,
        # the date is reshuffled at every epoch to reduce model overfitting, and Python’s multiprocessing can be used
        # to speed up data retrieval. DataLoader is an iterable that abstracts this complexity with an easy API

        # Initialize the custom data sets - Create instances of AISDataset
        logger.info("Initialize the custom data sets (AISDataset)")

        # Make sure that the same fishing training (when training on fishing vessels),
        # validation, and test sets are always used
        self.is_FishCargTank = False
        self.train_std = None
        if "FishCargTank" in file_name:
            self.is_FishCargTank = True

            # Handle this case to always use the same fishing vessels for training
            file_name_fish = file_name.replace("FishCargTank", "Fish")
            file_name_carg_tank = file_name.replace("FishCargTank", "CargTank")

            # Load the fishing vessel only and cargo/tanker only training sets
            training_set_fish = AISDataset(file_name_fish, discrete=self.discrete)
            training_set_carg_tank = AISDataset(
                file_name_carg_tank, discrete=self.discrete
            )

            # Combine the two data sets and update the mean values to the overall training mean value
            training_set = ConcatDataset([training_set_fish, training_set_carg_tank])
            self.train_mean = (
                training_set_fish.total_training_updates * training_set_fish.mean
                + training_set_carg_tank.total_training_updates
                * training_set_carg_tank.mean
            ) / (
                training_set_fish.total_training_updates
                + training_set_carg_tank.total_training_updates
            )
            if not self.discrete:
                # Update the standard deviation as well
                tmp1 = training_set_fish.get_all_input_points_df()
                tmp2 = training_set_carg_tank.get_all_input_points_df()
                tmp = pd.concat([tmp1, tmp2])
                self.train_std = torch.tensor(tmp.std(), dtype=torch.float)
                training_set.std = self.train_std
                training_set_carg_tank.std = self.train_std

            training_set_fish.mean = self.train_mean
            training_set_carg_tank.mean = self.train_mean
            self.input_shape = training_set_carg_tank.data_dim
        else:
            training_set = AISDataset(file_name, discrete=self.discrete)
            self.train_mean = training_set.mean
            self.train_std = None if self.discrete else training_set.std
            self.input_shape = training_set.data_dim

        if inject_cargo_proportion != 0.0:
            # Inject additional cargo/tanker trajectories into the training set
            file_name_carg_tank = file_name.replace("Fish", "CargTank")
            training_set_carg_tank = AISDataset(
                file_name_carg_tank, discrete=self.discrete
            )
            n = int(len(training_set) * inject_cargo_proportion)
            indices = range(0, n)
            training_set_carg_tank = torch.utils.data.Subset(
                training_set_carg_tank, indices
            )

            # Update the indicies and the mean (and std) value
            training_set_carg_tank.dataset.indicies = (
                training_set_carg_tank.dataset.indicies[:n]
            )
            training_set_carg_tank.dataset.mean = (
                training_set_carg_tank.dataset.compute_mean()
            )
            self.train_mean = (
                training_set.total_training_updates * training_set.mean
                + training_set_carg_tank.dataset.total_training_updates
                * training_set_carg_tank.dataset.mean
            ) / (
                training_set.total_training_updates
                + training_set_carg_tank.dataset.total_training_updates
            )

            if not self.discrete:
                # Update the standard deviation as well
                tmp1 = training_set.get_all_input_points_df()
                tmp2 = training_set_carg_tank.dataset.get_all_input_points_df()
                tmp = pd.concat([tmp1, tmp2])
                self.train_std = torch.tensor(tmp.std(), dtype=torch.float)
                training_set.std = self.train_std
                training_set_carg_tank.dataset.std = self.train_std

            # Do the mean value updates and concat the two datasets
            training_set.mean = self.train_mean
            training_set_carg_tank.dataset.mean = self.train_mean
            training_set_carg_tank = torch.utils.data.Subset(
                training_set_carg_tank, indices
            )
            training_set = ConcatDataset([training_set, training_set_carg_tank])
        self.training_n = len(training_set)
        self.batch_size = batch_size

        # Also make sure to always use the same fishing vessel for validation and test
        if self.is_FishCargTank:
            # Load the fishing vessel only and cargo/tanker only validation/test sets
            validation_set_fish = AISDataset(
                file_name_fish,
                self.train_mean,
                train_std=self.train_std,
                validation=True,
                discrete=self.discrete,
            )
            validation_set_carg_tank = AISDataset(
                file_name_carg_tank,
                self.train_mean,
                train_std=self.train_std,
                validation=True,
                discrete=self.discrete,
            )
            test_set_fish = AISDataset(
                file_name_fish,
                self.train_mean,
                train_std=self.train_std,
                validation=False,
                discrete=self.discrete,
            )
            test_set_carg_tank = AISDataset(
                file_name_carg_tank,
                self.train_mean,
                train_std=self.train_std,
                validation=False,
                discrete=self.discrete,
            )

            # Combine the two validation/test data sets
            validation_set = ConcatDataset(
                [validation_set_fish, validation_set_carg_tank]
            )
            test_set = ConcatDataset([test_set_fish, test_set_carg_tank])
        else:
            validation_set = AISDataset(
                file_name,
                self.train_mean,
                train_std=self.train_std,
                validation=True,
                discrete=self.discrete,
            )
            test_set = AISDataset(
                file_name,
                self.train_mean,
                train_std=self.train_std,
                validation=False,
                discrete=self.discrete,
            )
        self.validation_n = len(validation_set)
        self.test_n = len(test_set)

        # Define the training, validation, and test DataLoaders
        logger.info("Define the DataLoaders")
        self.training_dataloader = DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=dataset_utils.PadSequence,
        )
        self.validation_dataloader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=dataset_utils.PadSequence,
        )
        self.test_dataloader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=dataset_utils.PadSequence,
        )

        # Create fishing vessel only valdiation and test sets that have the same dates and ROI when requested
        self.evaluate_on_fishing_vessles = False
        if fishing_file is not None:
            self.evaluate_on_fishing_vessles = True
            logger.info(
                "Initialize fishing vessel only valdiation and test sets that have the same dates and ROI"
            )
            fishing_validation_set = AISDataset(
                fishing_file,
                self.train_mean,
                train_std=self.train_std,
                validation=True,
                discrete=self.discrete,
            )
            self.fishing_validation_n = len(fishing_validation_set)
            fishing_test_set = AISDataset(
                fishing_file,
                self.train_mean,
                train_std=self.train_std,
                validation=False,
                discrete=self.discrete,
            )
            self.fishing_test_n = len(fishing_test_set)

            # Define the validation and test DataLoaders
            logger.info("Initialize fishing vessel only val/test DataLoaders")
            self.fishing_validation_dataloader = DataLoader(
                fishing_validation_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=dataset_utils.PadSequence,
            )
            self.fishing_test_dataloader = DataLoader(
                fishing_test_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=dataset_utils.PadSequence,
            )

        # Create fishing vessel only valdiation and test sets that have different dates or ROI when requested
        self.evaluate_on_new_fishing_vessles = False
        if fishing_new_file is not None:
            self.evaluate_on_new_fishing_vessles = True
            logger.info(
                "Initialize fishing vessel only valdiation and test sets that have different dates or ROI"
            )
            fishing_new_validation_set = AISDataset(
                fishing_new_file,
                self.train_mean,
                train_std=self.train_std,
                validation=True,
                discrete=self.discrete,
            )
            self.fishing_new_validation_n = len(fishing_new_validation_set)
            fishing_new_test_set = AISDataset(
                fishing_new_file,
                self.train_mean,
                train_std=self.train_std,
                validation=False,
                discrete=self.discrete,
            )
            self.fishing_new_test_n = len(fishing_new_test_set)

            # Define the validation and test DataLoaders
            logger.info("Initialize new fishing vessel only val/test DataLoaders")
            self.fishing_new_validation_dataloader = DataLoader(
                fishing_new_validation_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=dataset_utils.PadSequence,
            )
            self.fishing_new_test_dataloader = DataLoader(
                fishing_new_test_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=dataset_utils.PadSequence,
            )

        # Define the neural network model that has some learnable parameters (or weights)
        # Initialize the variational recurrent neural network
        self.model = VRNN.VRNN(
            input_shape=self.input_shape,
            latent_shape=latent_dim,
            recurrent_shape=recurrent_dim,
            batch_norm=batch_norm,
            generative_bias=self.train_mean,
            device=self.device,
            generative_dist=self.generative_dist,
            GMM_components=GMM_components,
            GMM_equally_weighted=GMM_equally_weighted,
        ).to(self.device)

        # String that describes the model setup used
        batchNorm = "_batchNormTrue" if batch_norm else "_batchNormFalse"
        cargo_injected = (
            "_Injected" + str(inject_cargo_proportion).replace(".", "")
            if inject_cargo_proportion != 0.0
            else ""
        )
        GenerativeDist = "_" + self.generative_dist if not self.discrete else ""

        GMMComponents = ""
        GMMEquallyWeighted = ""
        if self.generative_dist == "GMM":
            GMMComponents = str(self.GMM_components)
            if self.GMM_equally_weighted:
                GMMEquallyWeighted = "EW"
            else:
                GMMEquallyWeighted = "NEW"
        self.model_name = (
            "VRNN"
            + "_"
            + file_name
            + cargo_injected
            + "_latent"
            + str(latent_dim)
            + "_recurrent"
            + str(recurrent_dim)
            + batchNorm
            + GenerativeDist
            + GMMComponents
            + GMMEquallyWeighted
        )
        logger.info("Model name: " + self.model_name)

        if is_trained:
            # Load the previously trained model
            if trained_model_name is not None:
                self.model_name = trained_model_name
                model_path = self.model_dir / (self.model_name + ".pth")
            elif intermediate_epoch is not None:
                model_path = self.model_intermediate_dir / (
                    self.model_name + "_" + str(intermediate_epoch) + ".pth"
                )
            else:
                model_path = self.model_dir / (self.model_name + ".pth")
            print(model_path)

            logger.info("Loading previously trained model: " + str(model_path))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)

    def loss_function(self, log_px, log_pz, log_qz, lengths, beta=1):
        """Computes the loss function

        as the negative of the timestep-wise variational lower bound objective function. Such that,
        learning can be done by maximizing the variational lower bound with respect to their parameters
        or equivalently minimizing the returned loss function

        Parameters
        ----------
        log_px : list
            Log probability of observing the target given the generating distribution p(x_t|z_t)

        log_pz : list
            Log probability of observing the sampled latent random variables under the prior distribution p(z_t)

        log_qz : list
            Log probability of observing the sampled latent random variables under the approximate posterior q(z_t|x_t)

        lengths : Tensor
            The actual sequence lengths

        beta : int (Defaults to 1)
            Weight to put on the Kullback–Leibler divergence part of the overall loss

        Returns
        -------
        Tensor :
            The calculated loss function

        Tensor :
         The log probabilities of observing the target (reconstructions) given the generating distribution

        Tensor :
            Tensor of Kullback–Leibler divergences for each seqeunce

        Tensor :
            The temporal mask (Tensor of booleans) that is False after the length of the corresponding sequence
        """
        # Get the sequence length for this batch (Can vary from batch to batch)
        max_seq_len = len(log_px)

        # Calculate the temporal mask for this batch size - Dimension: max_seq_len X batch_size
        curmask = (
            torch.arange(max_seq_len, device=self.device)[:, None] < lengths[None, :]
        )

        # log_px is really an array of reconstruction log probabilities
        # Stack that to a matrix where the first dimension is the time and multiply mask.
        # The mask will zero out reconstructions at time that goes over the trajcetory length.
        # That is, do not use the reconstructions for lengths past the corresponding track length
        log_px = torch.stack(log_px, dim=0) * curmask

        # Sum the log probabilities over time - Want to maximize this. That is, want to maximize
        # the log probability of observing the target (reconstructions) given the generating distribution
        log_px = log_px.sum(dim=0)

        # Do the same things for the prior and approximate posterior
        log_pz = torch.stack(log_pz, dim=0) * curmask
        log_qz = torch.stack(log_qz, dim=0) * curmask

        # Sum over time and subtrack the two values to get the KL loss
        # Want to miminize this loss such that the distribtuions are similar
        KL = log_qz.sum(dim=0) - log_pz.sum(dim=0)

        # The objective function becomes a timestep-wise variational lower bound. That is the
        # reconstruction - beta*KL (where the KL loss is weighted by beta)
        loss = log_px - beta * KL

        # Take the mean over the batch, where each input is  weighted in the batch by its track length
        # such that long tracks are not valued too higly. That is, equal weight no matter the length of the tracks
        loss = torch.mean(loss / lengths)  # This function should be maximized
        return -loss, log_px, KL, curmask

    def train_loop(
        self, optimizer, beta_weight, kl_weight_step, kl_weight, opt_steps, num_samples
    ):
        """The Train Loop:
        Iterate over the training data set and try to converge to optimal parameters.
        Inside the training loop, optimization happens in three steps:

        1. Call optimizer.zero_grad() to reset the gradients of model parameters
        2. Backpropagate the prediction loss (loss.backwards()). PyTorch deposits the gradients of the loss w.r.t. each parameter
        3. With the gradients, call optimizer.step() to adjust the parameters by the gradients collected in the backward pass

        Parameters
        ----------
        optimizer : torch.optim
            All optimization logic is encapsulated in the optimizer object

        beta_weight : float
            Current value of the Kullback–Leibler divergence loss weight

        kl_weight_step : float
            Step size to use when increasing the Kullback–Leibler divergence loss weight

        kl_weight : int
            Maximum weight of the Kullback–Leibler divergence loss

        opt_steps : int
            Current number of optimization steps taken

        num_samples : int
            The current number of training samples already processed

        Returns
        -------
        list :
            List of things to track in the main training loop
        """
        # Iterate over a data set of inputs - Begin training loop
        loss_epoch, kl_epoch, recon_epoch = 0, 0, 0
        self.model.train()
        for _, (
            data_set_indices,
            file_location_indices,
            mmsis,
            time_stamps,
            ship_types,
            lengths,
            inputs,
            targets,
        ) in enumerate(self.training_dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)

            # Process input through the network - Pass the input data to the model (executes the model’s forward)
            (
                log_px,
                log_pz,
                log_qz,
                _,
                _,
                _,
                _,
                _,
            ) = self.model(inputs, targets)

            # Compute the loss (how far is the output from being correct)
            loss, log_px, kl, _ = self.loss_function(
                log_px, log_pz, log_qz, lengths, beta=beta_weight
            )

            # Reset the gradients of model parameters (gradients add up by default) - To prevent double-counting
            optimizer.zero_grad()

            # Propagate gradients back into the network’s parameters
            loss.backward()  #  All tensors (with requires_grad=True) in the whole graph are differentiated w.r.t. the network parameters

            # Update the weights of the network using gradient decent
            # Call .step() to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in .grad.
            # Train the network by calculating the gradient w.r.t the cost function and update the parameters in direction of the negative gradient
            optimizer.step()  # Parameters are updated each time optimizer.step() is called

            # Keep track of the number of optimization steps and number of samples processed
            opt_steps += 1
            num_samples += len(lengths)

            # Variables for KL anneling (when included) - By default KL annealing is not introduced
            beta_weight += kl_weight_step
            beta_weight = min(beta_weight, kl_weight)

            loss_epoch += loss.item() * len(lengths)
            kl_epoch += torch.sum(kl / lengths).item()
            recon_epoch += torch.sum(log_px / lengths).item()
        return [
            loss_epoch / self.training_n,
            kl_epoch / self.training_n,
            recon_epoch / self.training_n,
            beta_weight,
            opt_steps,
            num_samples,
        ]

    def evaluate_loop(self, data_loader, data_n, beta_weight=1):
        """The Validation/Test Loop:
        Iterate over the validation/test data set to check if model performance is improving

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Validation/Test set DataLoader

        data_n : int
            Size of the validation/test set

        beta_weight : float (Default to 1)
            Current value of the Kullback–Leibler divergence loss weight

        Returns
        -------
        list :
            List of validation/test related things to keep track of.
            Also returns reconstruction log probability, track lengths and vessel types
        """
        # Iterate over the evaluation data set to check if model performance is improving -  Begin evaluation loop
        loss_epoch, kl_epoch, recon_epoch = 0, 0, 0
        all_file_location_indices, all_data_set_indices, all_mmsis = [], [], []
        all_log_px, all_lengths, all_ship_types = [], [], []

        self.model.eval()
        for _, (
            data_set_indices,
            file_location_indices,
            mmsis,
            time_stamps,
            ship_types,
            lengths,
            inputs,
            targets,
        ) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)

            # Process input through the network - Pass the input data to the model (executes the model’s forward)
            log_px, log_pz, log_qz, _, _, _, _, _ = self.model(inputs, targets)

            # Compute the loss (how far is the output from being correct)
            loss, log_px, kl, _ = self.loss_function(
                log_px, log_pz, log_qz, lengths, beta=beta_weight
            )

            loss_epoch += loss.item() * len(lengths)
            kl_epoch += torch.sum(kl / lengths).item()
            recon_epoch += torch.sum(log_px / lengths).item()
            all_log_px += log_px.tolist()
            all_lengths += lengths.tolist()
            all_ship_types += [
                dataset_utils.convertShipLabelToType(ship_type.item())
                for ship_type in ship_types
            ]
            all_file_location_indices += file_location_indices.tolist()
            all_data_set_indices += data_set_indices.tolist()
            all_mmsis += mmsis.tolist()
        return [
            loss_epoch / data_n,
            kl_epoch / data_n,
            recon_epoch / data_n,
            all_log_px,
            all_lengths,
            all_ship_types,
            all_file_location_indices,
            all_mmsis,
            all_data_set_indices,
        ]

    def save_training_curves(
        self,
        val_loss,
        val_kl,
        val_recon,
        dir,
        loss=None,
        kl=None,
        recon=None,
        name_prefix="",
        level=None,
        level_values=None,
    ):
        """Stores the training and validation learning curves as a .csv file

        Parameters
        ----------

        Returns
        -------
        val_loss : list
            Validation loss at each epoch

        val_kl : list
            Validation Kullback–Leibler divergences at each epoch

        val_recon : list
            Validation log probabilities of observing the target (reconstructions)

        dir : pathlib.WindowsPath
            The directory to store the training curves

        loss : list (Defaults to None)
            Training loss at each epoch

        kl : list (Defaults to None)
            Training Kullback–Leibler divergences at each epoch

        recon : list (Defaults to None)
            Training log probabilities of observing the target (reconstructions)

        name_prefix : str (Defaults to empty string)
            String that comes in front of the saved file name

        level : str (Defaults to None)
            The level of the learning curve values (will be a column name in the data frame)

        level_values : list (Defaults to None)
            The actual value of the levels (will be the values in the level data frame column)
        """
        # Check if both the training and validation curves should be stored or only validation
        if loss is not None and kl is not None and recon is not None:
            training_curves = pd.DataFrame(
                {
                    "Training_Loss": loss,
                    "Training_KL_Divergence": kl,
                    "Training_Reconstruction": recon,
                    "Validation_Loss": val_loss,
                    "Validation_KL_Divergence": val_kl,
                    "Validation_Reconstruction": val_recon,
                }
            )
        else:
            training_curves = pd.DataFrame(
                {
                    "Validation_Loss": val_loss,
                    "Validation_KL_Divergence": val_kl,
                    "Validation_Reconstruction": val_recon,
                }
            )

        if level is not None and level_values is not None:
            training_curves[level] = level_values

        training_curves.to_csv(
            dir / (name_prefix + self.model_name + "_curves.csv"),
            index=False,
        )

    def train_VRNN(
        self,
        num_epochs=50,
        learning_rate=0.001,
        kl_weight=1,
        kl_anneling_start=1,
        num_opt_steps=None,
        num_training_samples=None,
        plot_after_epoch=False,
        scheduler_gamma=None,
        scheduler_step_size=1,
        scheduler_milestones=None,
    ):
        """Train (and validate with validation set) a deep learning VRNN model

        Training consists of two general steps:
        1. Forward Propagation: The input data goes through each of the models functions to
                                make a best guess at the correct output
        2. Backward Propagation: Model parameters are adjusted by traversing backwards from the output,
                                 collecting the derivatives of the error with respect to the parameters of the
                                 functions (gradients), and optimizing the parameters using gradient descent

        The function performs the following steps:
            1. Iterate over a data set of inputs
            2. Process input through the network
            3. Compute the loss (how far is the output from being correct)
            4. Propagate gradients back into the network’s parameters
            5. Update the weights of the network using gradient decent
            6. Iterate over the validation data set to check if model performance is improving

        Parameters
        ----------
        num_epochs : int (Defaults to 30)
            The number times to iterate over the entire data set

        learning_rate : float (Defaults to 0.001)
            How much to update models parameters at each batch. Smaller values yield slow learning speed,
            while large values may result in unpredictable behavior during training

        kl_weight : int (Defaults to 1)
            Maximum weight of the Kullback–Leibler divergence loss

        kl_anneling_start : int (Defaults to 1)
            Starting value of the Kullback–Leibler divergence loss. When set to 0, the value is
            annealed to kl_weight over 10 epochs

        num_opt_steps : int (Defaults to None)
            Total nmber of optimization steps to perform during training. When this is not None,
            the num_epoch will be ignored and calculated based on num_opt_steps

        num_training_samples : int (Defaults to None)
            Total nmber of training samples to process during training. When this is not None,
            the num_epoch will be ignored and calculated based on num_training_samples

        plot_after_epoch : bool (Defaults to False)
            When True, plots are generated to show learning after each epoch

        scheduler_gamma : float (Defaults to None)
            Multiplicative factor of learning rate decay. When None,
            no learning rate decay is applied

        scheduler_step_size : int (Defaults to 1)
            Decays the learning rate of each parameter group by gamma every step_size epochs

        scheduler_milestones : list (Defaults to None)
            The epochs where the learning rate is decreased by a factor of scheduler_gamma

        Returns
        -------
        model : src.models.VRNN
            The trained VRNN model
        """
        logger = logging.getLogger(__name__)  # For logging information

        # Define the optimizer -  Optimization is the process of adjusting model parameters
        # to reduce model error in each training step.  All optimization logic is encapsulated in the optimizer object
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Calculate the number of epoch to run when num_opt_steps or num_training_samples are given
        if num_opt_steps is not None:
            num_epochs = int(np.ceil(num_opt_steps * self.batch_size / self.training_n))
        elif num_training_samples is not None:
            num_epochs = int(np.ceil(num_training_samples / self.training_n))

        if learning_rate != 0.001:
            self.model_name = (
                self.model_name + "_lr" + str(learning_rate).replace(".", "")
            )
            logger.info("Model name with a different learning rate: " + self.model_name)

        use_scheduler = False
        if scheduler_gamma is not None:
            # Using a scheduler
            use_scheduler = True
            if scheduler_milestones is not None:
                self.model_name = (
                    self.model_name
                    + "_S"
                    + "".join([str(i) for i in scheduler_milestones])
                    + "_"
                    + str(scheduler_gamma).replace(".", "")
                )
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
                )
            else:
                self.model_name = (
                    self.model_name
                    + "_S"
                    + str(scheduler_step_size)
                    + "_"
                    + str(scheduler_gamma).replace(".", "")
                )
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
                )
            logger.info("Model name with scheduler: " + self.model_name)

        # Keep track of losses, KL divergence, and reconstructions on an epoch level
        loss_tot, kl_tot, recon_tot = [], [], []
        val_loss_tot, val_kl_tot, val_recon_tot = [], [], []

        # Keep track of the number of optimization steps and number of samples processed
        opt_steps = 0
        num_samples = 0
        opt_steps_all = []
        num_samples_all = []

        if self.evaluate_on_fishing_vessles:
            # Also validate on fishing vessels only and keep track of those values
            # Keep track of losses, KL divergence, and reconstructions on an epoch level
            fish_val_loss_tot, fish_val_kl_tot, fish_val_recon_tot = [], [], []

        if self.evaluate_on_new_fishing_vessles:
            # Also validate on new (different ROI/dates) fishing vessels only and keep track of those values
            # Keep track of losses, KL divergence, and reconstructions on an epoch level
            fish_new_val_loss_tot = []
            fish_new_val_kl_tot = []
            fish_new_val_recon_tot = []

        # Variables for KL anneling (when included) - By default KL annealing is not introduced
        if kl_anneling_start != 1:
            # Using KL annealing
            self.model_name = self.model_name + "_KLTrue"
            logger.info("Model name with KL annealing: " + self.model_name)
        beta_weight = kl_anneling_start

        # Annealing over 10 epochs - Weight step to update after each mini batch
        kl_weight_step = abs(kl_weight - kl_anneling_start) / (
            10 * len(self.training_dataloader)
        )

        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch} Start ----------------------------------------")
            logger.info("Run the training loop")
            train_results = self.train_loop(
                optimizer,
                beta_weight,
                kl_weight_step,
                kl_weight,
                opt_steps,
                num_samples,
            )
            loss_tot.append(train_results[0])
            kl_tot.append(train_results[1])
            recon_tot.append(train_results[2])
            beta_weight = train_results[3]
            opt_steps = train_results[4]
            opt_steps_all.append(opt_steps)
            num_samples = train_results[5]
            num_samples_all.append(num_samples)

            logger.info("Run the validation loop")
            val_results = self.evaluate_loop(
                self.validation_dataloader, self.validation_n, beta_weight
            )
            val_loss_tot.append(val_results[0])
            val_kl_tot.append(val_results[1])
            val_recon_tot.append(val_results[2])

            if self.evaluate_on_fishing_vessles:
                # Also validate on a data set consisting only of fishing vessels
                logger.info("Run the validation loop - Fishing vessels only")
                val_results = self.evaluate_loop(
                    self.fishing_validation_dataloader,
                    self.fishing_validation_n,
                    beta_weight,
                )
                fish_val_loss_tot.append(val_results[0])
                fish_val_kl_tot.append(val_results[1])
                fish_val_recon_tot.append(val_results[2])

            if self.evaluate_on_new_fishing_vessles:
                # Also validate on a data set consisting only of fishing vessels
                logger.info("Run the validation loop - NEW Fishing vessels only")
                val_results = self.evaluate_loop(
                    self.fishing_new_validation_dataloader,
                    self.fishing_new_validation_n,
                    beta_weight,
                )
                fish_new_val_loss_tot.append(val_results[0])
                fish_new_val_kl_tot.append(val_results[1])
                fish_new_val_recon_tot.append(val_results[2])

            # Check if a scheduler step should be taken as well
            if use_scheduler:
                logger.info(
                    "Learning rate before scheduler.step: " + str(scheduler.get_lr())
                )
                scheduler.step()
                logger.info(
                    "Learning rate after scheduler.step: " + str(scheduler.get_lr())
                )

            # Plot three random validation trajectories
            if self.is_FishCargTank:
                # Select randomly either fishing or cargo/tankers to plot
                validation_set = self.validation_dataloader.dataset
                validation_set = validation_set.datasets[np.random.choice([0, 1])]
                datapoints = np.random.choice(
                    len(validation_set), size=3, replace=False
                )
            else:
                validation_set = self.validation_dataloader.dataset
                datapoints = np.random.choice(self.validation_n, size=3, replace=False)
            if self.discrete:
                # TODO: Update for continuous also
                if plot_after_epoch:
                    plotting.make_vae_plots(
                        (
                            loss_tot,
                            kl_tot,
                            recon_tot,
                            val_loss_tot,
                            val_kl_tot,
                            val_recon_tot,
                        ),
                        self.model,
                        datapoints,
                        validation_set,
                        validation_set.data_info["binedges"],
                        self.device,
                        figure_path=self.model_intermediate_dir
                        / (self.model_name + "_Results_" + str(epoch) + ".pdf"),
                    )
            logger.info(
                "Epoch {} of {} finished. Training loss = {}. Validation loss = {}".format(
                    epoch, num_epochs, loss_tot[-1], val_loss_tot[-1]
                )
            )

            # Save the model every 10 epochs
            if epoch % 10 == 0:
                # Save the current model version
                torch.save(
                    self.model.state_dict(),
                    self.model_intermediate_dir
                    / (self.model_name + "_" + str(epoch) + ".pth"),
                )

                # Store the learning curves as a .csv file
                self.save_training_curves(
                    val_loss=val_loss_tot,
                    val_kl=val_kl_tot,
                    val_recon=val_recon_tot,
                    dir=self.model_intermediate_dir,
                    loss=loss_tot,
                    kl=kl_tot,
                    recon=recon_tot,
                )
        logger.info("Training End ----------------------------------------")
        logger.info("--- %s seconds to train ---" % (time.time() - start_time))

        # Save the training and validation learning curves on an epoch level
        self.save_training_curves(
            val_loss=val_loss_tot,
            val_kl=val_kl_tot,
            val_recon=val_recon_tot,
            dir=self.model_dir,
            loss=loss_tot,
            kl=kl_tot,
            recon=recon_tot,
        )

        # Save the training and validation learning curves on step and sample level
        self.save_training_curves(
            val_loss=val_loss_tot,
            val_kl=val_kl_tot,
            val_recon=val_recon_tot,
            dir=self.model_dir,
            loss=loss_tot,
            kl=kl_tot,
            recon=recon_tot,
            name_prefix="opt_step_lvl_",
            level="Number of optimiser steps",
            level_values=opt_steps_all,
        )

        self.save_training_curves(
            val_loss=val_loss_tot,
            val_kl=val_kl_tot,
            val_recon=val_recon_tot,
            dir=self.model_dir,
            loss=loss_tot,
            kl=kl_tot,
            recon=recon_tot,
            name_prefix="sample_lvl_",
            level="Number of training samples processed",
            level_values=num_samples_all,
        )

        if self.evaluate_on_fishing_vessles:
            # Also save the validation learning curve for fishing vessels only
            self.save_training_curves(
                val_loss=fish_val_loss_tot,
                val_kl=fish_val_kl_tot,
                val_recon=fish_val_recon_tot,
                dir=self.model_dir,
                name_prefix="Fishing_vessels_only_",
            )
            self.save_training_curves(
                val_loss=fish_val_loss_tot,
                val_kl=fish_val_kl_tot,
                val_recon=fish_val_recon_tot,
                dir=self.model_dir,
                name_prefix="Fishing_vessels_only_opt_step_lvl_",
                level="Number of optimiser steps",
                level_values=opt_steps_all,
            )
            self.save_training_curves(
                val_loss=fish_val_loss_tot,
                val_kl=fish_val_kl_tot,
                val_recon=fish_val_recon_tot,
                dir=self.model_dir,
                name_prefix="Fishing_vessels_only_sample_lvl_",
                level="Number of training samples processed",
                level_values=num_samples_all,
            )

        if self.evaluate_on_new_fishing_vessles:
            # Also save the validation learning curve for the new fishing vessels only
            self.save_training_curves(
                val_loss=fish_new_val_loss_tot,
                val_kl=fish_new_val_kl_tot,
                val_recon=fish_new_val_recon_tot,
                dir=self.model_dir,
                name_prefix="New_Fishing_vessels_only_",
            )
            self.save_training_curves(
                val_loss=fish_new_val_loss_tot,
                val_kl=fish_new_val_kl_tot,
                val_recon=fish_new_val_recon_tot,
                dir=self.model_dir,
                name_prefix="New_Fishing_vessels_only_opt_step_lvl_",
                level="Number of optimiser steps",
                level_values=opt_steps_all,
            )
            self.save_training_curves(
                val_loss=fish_new_val_loss_tot,
                val_kl=fish_new_val_kl_tot,
                val_recon=fish_new_val_recon_tot,
                dir=self.model_dir,
                name_prefix="New_Fishing_vessels_only_sample_lvl_",
                level="Number of training samples processed",
                level_values=num_samples_all,
            )

        # Save the final (trained) model
        torch.save(self.model.state_dict(), self.model_dir / (self.model_name + ".pth"))

    def track_reconstructions(
        self,
        data_set,
        data_set_idx,
    ):
        """Reconstruct the requested trajectory in the given data set

        Parameters
        ----------
        data_set : src.data.Datasets
            The data set that contains the trajectory to reconstruct

        data_set_index : int
            The index of the trajectory to reconstruct

        Returns
        -------
        pandas.DataFrame
            Data frame containing the reconstructions and other useful information

        """
        self.model.eval()
        (
            data_set_index,
            file_location_index,
            mmsi,
            time_stamps,
            ship_type,
            length,
            input,
            target,
        ) = data_set[data_set_idx]
        input = input.to(self.device)
        target_device = target.to(self.device)
        if self.is_FishCargTank:
            # Select the first data set to get general data information
            data_set = data_set.datasets[np.random.choice([0, 1])]

        reconstruction_discrete = None
        if self.generative_dist == "Bernoulli":
            # Initialize a variable to keep track the logits from the model. The dimension
            # here are the sequence length (t) X 1 (one sample) X input data dimension
            logits = torch.zeros(
                length.int().item(), 1, data_set.data_dim, device=self.device
            )

            # Use the pretrained model
            log_px, _, _, logits, _, _, _, _ = self.model(
                input.unsqueeze(0), target_device.unsqueeze(0), logits=logits
            )
            logits = logits.cpu()

            # Go from the log odds to a four hot encoded discrete representation
            # Each of lat/lon/speed/course will be one at the max logit but zero everywhere else
            reconstruction_discrete = plotting.logitToTrack(
                logits, data_set.data_info["binedges"]
            )

            # Get the reconstructed continuous lat and lon values
            (
                reconstruction_lon,
                reconstruction_lat,
                reconstruction_speed,
                reconstruction_course,
            ) = plotting.PlotDatasetTrack(
                reconstruction_discrete, data_set.data_info["binedges"]
            )

            reconstruction = pd.DataFrame(
                {
                    "Longitude": reconstruction_lon,
                    "Latitude": reconstruction_lat,
                    "Speed": reconstruction_speed,
                    "Course": reconstruction_course,
                }
            )

        elif (
            self.generative_dist == "Isotropic_Gaussian"
            or self.generative_dist == "Diagonal"
        ):
            # Initialize variables to keep track of the mean and variance-covariance matrix from the model
            # The location dimensions are: Sequence length (t) X 1 (one sample) X input data dimension (four-dimensional input)
            mus = torch.zeros(
                length.int().item(), 1, data_set.data_dim, device=self.device
            )
            # The matrix dimensions are: Sequence length (t) X 1 (one sample) X input data dimension X input data dimension (four-dimensional input)
            Sigmas = torch.zeros(
                length.int().item(),
                1,
                data_set.data_dim,
                data_set.data_dim,
                device=self.device,
            )
            # Use the pretrained model
            log_px, _, _, _, _, _, _, _ = self.model(
                input.unsqueeze(0),
                target_device.unsqueeze(0),
                obs_mus=mus,
                obs_Sigmas=Sigmas,
            )
            mus = mus.cpu()
            Sigmas = Sigmas.cpu()

            # Get the standard deviation components from the variance-covariance matrix
            sigmas2 = torch.zeros(
                length.int().item(),
                1,
                data_set.data_dim,
                device="cpu",
            )
            for t in range(length.int().item()):
                sigmas2[t, 0, :] = torch.FloatTensor(
                    [Sigmas[t, 0, i, i] for i in range(data_set.data_dim)]
                )
            # Get the mean values as lists
            reconstruction_lat = [x[0] for x in mus[:, :, 0].tolist()]
            reconstruction_lon = [x[0] for x in mus[:, :, 1].tolist()]
            reconstruction_speed = [x[0] for x in mus[:, :, 2].tolist()]
            reconstruction_course = [x[0] for x in mus[:, :, 3].tolist()]

            # Get the scale values as lists
            sigma_reconstruction_lat = [
                math.sqrt(x[0]) for x in sigmas2[:, :, 0].tolist()
            ]
            sigma_reconstruction_lon = [
                math.sqrt(x[0]) for x in sigmas2[:, :, 1].tolist()
            ]
            sigma_reconstruction_speed = [
                math.sqrt(x[0]) for x in sigmas2[:, :, 2].tolist()
            ]
            sigma_reconstruction_course = [
                math.sqrt(x[0]) for x in sigmas2[:, :, 3].tolist()
            ]

            reconstruction = pd.DataFrame(
                {
                    "Longitude": reconstruction_lon,
                    "Latitude": reconstruction_lat,
                    "Speed": reconstruction_speed,
                    "Course": reconstruction_course,
                    "Longitude sigma": sigma_reconstruction_lon,
                    "Latitude sigma": sigma_reconstruction_lat,
                    "Speed sigma": sigma_reconstruction_speed,
                    "Course sigma": sigma_reconstruction_course,
                }
            )

        elif self.generative_dist == "GMM" and not self.GMM_equally_weighted:
            # Initialize variables to keep track of the mixing probabilities, location, and scale parameters from the models
            # The mixing probabilities dimensions are: Sequence length (t) X 1 (one sample) X GMM_components
            mix_probs = torch.zeros(
                length.int().item(),
                1,
                self.GMM_components,
                device=self.device,
            )

            # The location dimensions are: Sequence length (t) X 1 (one sample) X GMM_components X input data dimension (four-dimensional input)
            mus = torch.zeros(
                length.int().item(),
                1,
                self.GMM_components,
                data_set.data_dim,
                device=self.device,
            )

            # The Gaussians have a diagonal variance-covariance structure so only the scale parameter is stored
            # The dimensions are: Sequence length (t) X 1 (one sample) X GMM_components X input data dimension (four-dimensional input)
            sigmas = torch.zeros(
                length.int().item(),
                1,
                self.GMM_components,
                data_set.data_dim,
                device=self.device,
            )
            # Use the pretrained model
            log_px, _, _, _, _, _, _, _ = self.model(
                input.unsqueeze(0),
                target_device.unsqueeze(0),
                obs_mus=mus,
                obs_Sigmas=sigmas,
                obs_probs=mix_probs,
            )
            mus = mus.cpu()
            sigmas = sigmas.cpu()
            mix_probs = mix_probs.cpu()

            # Get the parameter values as lists and by the max component
            reconstruction_lat, reconstruction_lon = [], []
            reconstruction_speed, reconstruction_course = [], []
            sigma_reconstruction_lon, sigma_reconstruction_lat = [], []
            sigma_reconstruction_speed, sigma_reconstruction_course = [], []
            max_mixing_i, max_mixing_probs = [], []

            for t in range(length.int().item()):
                # Find the maximum component and its probability
                mix_probs_t = mix_probs[t, :, :].tolist()[0]
                max_i = mix_probs_t.index(max(mix_probs_t))
                max_mixing_i.append(max_i)
                max_mixing_probs.append(max(mix_probs_t))

                # Store the corresponding location and scale
                reconstruction_lat.append(mus[t, :, max_i, 0].item())
                reconstruction_lon.append(mus[t, :, max_i, 1].item())
                reconstruction_speed.append(mus[t, :, max_i, 2].item())
                reconstruction_course.append(mus[t, :, max_i, 3].item())

                sigma_reconstruction_lat.append(sigmas[t, :, max_i, 0].item())
                sigma_reconstruction_lon.append(sigmas[t, :, max_i, 1].item())
                sigma_reconstruction_speed.append(sigmas[t, :, max_i, 2].item())
                sigma_reconstruction_course.append(sigmas[t, :, max_i, 3].item())

            reconstruction = pd.DataFrame(
                {
                    "Longitude": reconstruction_lon,
                    "Latitude": reconstruction_lat,
                    "Speed": reconstruction_speed,
                    "Course": reconstruction_course,
                    "Longitude sigma": sigma_reconstruction_lon,
                    "Latitude sigma": sigma_reconstruction_lat,
                    "Speed sigma": sigma_reconstruction_speed,
                    "Course sigma": sigma_reconstruction_course,
                    "Max mixing index": max_mixing_i,
                    "Max mixing probability": max_mixing_probs,
                }
            )
        return {
            "Reconstruction": reconstruction,
            "Reconstruction four-hot encoded": reconstruction_discrete,
            "Reconstruction log probability": [x.item() for x in log_px],
        }
