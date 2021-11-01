# Please note that some code in this class builds upon work done by Kristoffer Vinther Olesen (@DTU)
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.models.VRNN as VRNN
import src.utils.dataset_utils as dataset_utils
import src.utils.plotting as plotting
from src.data.Datasets import AISDiscreteRepresentation


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

        batch_norm : bool (Defaults to True)
            When set to True, natch normalization is included in the networks
        """
        logger = logging.getLogger(__name__)  # For logging information
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using {} device".format(self.device))

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

        # Initialize the custom data sets - Create instances of AISDiscreteRepresentation
        logger.info("Initialize the custom data sets (AISDiscreteRepresentation)")
        training_set = AISDiscreteRepresentation(file_name)
        self.train_mean = training_set.mean
        self.input_shape = training_set.data_dim
        self.training_n = len(training_set)
        validation_set = AISDiscreteRepresentation(
            file_name, self.train_mean, validation=True
        )
        self.validation_n = len(validation_set)
        test_set = AISDiscreteRepresentation(
            file_name, self.train_mean, validation=False
        )
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

        # Define the neural network model that has some learnable parameters (or weights)
        # Initialize the variational recurrent neural network
        self.model = VRNN.VRNN(
            input_shape=training_set.data_dim,
            latent_shape=latent_dim,
            recurrent_shape=recurrent_dim,
            batch_norm=batch_norm,
            generative_bias=self.train_mean,
            device=self.device,
        ).to(self.device)

        # String that describes the model setup used
        batchNorm = "_batchNormTrue" if batch_norm else "_batchNormFalse"
        self.model_name = (
            "VRNN"
            + "_"
            + file_name
            + "_latent"
            + str(latent_dim)
            + "_recurrent"
            + str(recurrent_dim)
            + batchNorm
        )
        logger.info("Model name: " + self.model_name)

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

    def train_loop(self, optimizer, beta_weight, kl_weight_step, kl_weight):
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

        Returns
        -------
        list :
            List of things to track in the main training loop
        """
        # Iterate over a data set of inputs - Begin training loop
        loss_epoch, kl_epoch, recon_epoch = 0, 0, 0
        self.model.train()
        for _, (_, _, _, lengths, inputs, targets) in enumerate(
            self.training_dataloader
        ):
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
        ]

    def evaluate_loop(self, data_loader, data_n, beta_weight):
        """The Validation/Test Loop:
        Iterate over the validation/test data set to check if model performance is improving

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Validation/Test set DataLoader

        data_n : int
            Size of the validation/test set

        beta_weight : float
            Current value of the Kullback–Leibler divergence loss weight

        Returns
        -------
        list :
            List of validation/test related things to keep track of
        """
        # Iterate over the evaluation data set to check if model performance is improving -  Begin evaluation loop
        loss_epoch, kl_epoch, recon_epoch = 0, 0, 0
        self.model.eval()
        for _, (_, _, _, lengths, inputs, targets) in enumerate(data_loader):
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
        return [
            loss_epoch / data_n,
            kl_epoch / data_n,
            recon_epoch / data_n,
        ]

    def train_VRNN(
        self,
        num_epochs=50,
        learning_rate=0.001,
        kl_weight=1,
        kl_anneling_start=1,
        use_scheduler=False,
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

        use_scheduler : bool (Defaults to False)
            When set to true a Scheduler will be introduced and used

        Returns
        -------
        model : src.models.VRNN
            The trained VRNN model
        """
        logger = logging.getLogger(__name__)  # For logging information
        validation_set = self.validation_dataloader.dataset

        # Define the optimizer -  Optimization is the process of adjusting model parameters
        # to reduce model error in each training step.  All optimization logic is encapsulated in the optimizer object
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if use_scheduler:
            # Using a  scheduler
            self.model_name = self.model_name + "_SchedulerTrue"
            logger.info("Model name with scheduler: " + self.model_name)
            # Milestones are epochs where the learning rate is decreased by a factor of 0.3
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[2, 10, 20], gamma=0.3
            )

        # Keep track of losses, KL divergence, and reconstructions
        loss_tot, kl_tot, recon_tot = [], [], []
        val_loss_tot, val_kl_tot, val_recon_tot = [], [], []

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
                optimizer, beta_weight, kl_weight_step, kl_weight
            )
            loss_tot.append(train_results[0])
            kl_tot.append(train_results[1])
            recon_tot.append(train_results[2])
            beta_weight = train_results[3]

            logger.info("Run the validation loop")
            val_results = self.evaluate_loop(
                self.validation_dataloader, self.validation_n, beta_weight
            )
            val_loss_tot.append(val_results[0])
            val_kl_tot.append(val_results[1])
            val_recon_tot.append(val_results[2])

            if use_scheduler:
                scheduler.step()

            # Plot three random validation trajectories
            datapoints = np.random.choice(self.validation_n, size=3, replace=False)
            plotting.make_vae_plots(
                (loss_tot, kl_tot, recon_tot, val_loss_tot, val_kl_tot, val_recon_tot),
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
                    epoch, num_epochs, train_results[0], val_results[0]
                )
            )

            # Save the model every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    self.model_intermediate_dir
                    / (self.model_name + "_" + str(epoch) + ".pth"),
                )
                training_curves = pd.DataFrame(
                    {
                        "Training_Loss": loss_tot,
                        "Training_KL_Divergence": kl_tot,
                        "Training_Reconstruction": recon_tot,
                        "Validation_Loss": val_loss_tot,
                        "Validation_KL_Divergence": val_kl_tot,
                        "Validation_Reconstruction": val_recon_tot,
                    }
                )
                training_curves.to_csv(
                    self.model_intermediate_dir / (self.model_name + "_curves.csv"),
                    index=False,
                )
        logger.info("Training End ----------------------------------------")
        logger.info("--- %s seconds to train ---" % (time.time() - start_time))

        training_curves = pd.DataFrame(
            {
                "Training_Loss": loss_tot,
                "Training_KL_Divergence": kl_tot,
                "Training_Reconstruction": recon_tot,
                "Validation_Loss": val_loss_tot,
                "Validation_KL_Divergence": val_kl_tot,
                "Validation_Reconstruction": val_recon_tot,
            }
        )
        training_curves.to_csv(
            self.model_dir / (self.model_name + "_curves.csv"), index=False
        )
        torch.save(self.model.state_dict(), self.model_dir / (self.model_name + ".pth"))
