# Please note that some code in this class builds upon work done by Kristoffer Vinther Olesen (@DTU)
import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli, MixtureSameFamily, MultivariateNormal, Normal

from src.models.Distributions import ReparameterizedDiagonalGaussian


# Define the neural network by subclassing nn.Module, and initialize the neural network layers in __init__
class VRNN(nn.Module):
    """
    A Variational Recurrent Neural Network (VRNN) with

    * an isotropic Gaussian prior p(z_t)
    * an isotropic Gaussian approximate posterior q(z_t|x_t)
    * a Bernoulli observation model p(x_t|z_t)

    ...

    Attributes
    ----------
    input_shape : torch.Size
        Feature dimensions

    latent_shape : int
        Size of the latent space

    recurrent_shape : int
        Size of the recurrent hidden space

    batch_norm : bool
        Whether or not to use batch normalization

    generative_bias :
        The training set mean vector

    device : str
        The device to use. Either 'cuda' for a hardware accelerator like the GPU or 'cpu'

    phi_x : torch.nn.modules.container.Sequential
        The feature extractor of the input which extract features from x_t

    phi_z : torch.nn.modules.container.Sequential
        The feature extractor of the of the latent random variables which extract features from z_t

    prior_network : torch.nn.modules.container.Sequential
        The prior function/network

    encoder : torch.nn.modules.container.Sequential
        The encoder network

    decoder : torch.nn.modules.container.Sequential
        The decoder network

    rnn : torch.nn.modules.container.Sequential
        LSTM RNN network

    generative_dist : str (Defaults to 'Bernoulli')
        The observation model to use

    GMM_components : int (Defaults to 4)
        The number of components to use as part of the GMM

    GMM_equally_weighted : bool (Defaults to True)
        When True, all GMM components are equally weighted

    Methods
    -------
    prior(h, sigma_min, raw_sigma_bias)
        Returns the distribution p(z_t)

    posterior(h, x_features, sigma_min, raw_sigma_bias)
        Returns the approximate posterior distribution q(z_t|x_t)

    generative(z_features, h):
        Returns the generating distribution p(x_t|z_t)

    forward(inputs, targets, logits, hs, zs, z_mus, z_sigmas):
        Computes the log probabilities that can be used in the VRNN loss function
    """

    def __init__(
        self,
        input_shape,
        latent_shape,
        recurrent_shape,
        batch_norm,
        generative_bias,
        device,
        generative_dist="Bernoulli",
        GMM_components=4,
        GMM_equally_weighted=True,
    ):

        super(VRNN, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.recurrent_shape = recurrent_shape
        self.batch_norm = batch_norm
        self.generative_bias = generative_bias.to(device)
        self.device = device
        self.generative_dist = generative_dist
        self.GMM_components = GMM_components
        self.GMM_equally_weighted = GMM_equally_weighted

        # Start by defining the two feature extractors that use a fully connected
        # network with one hidden layer with ReLU activation:
        layers_phi_x = [nn.Linear(self.input_shape, self.latent_shape)]
        if self.batch_norm:  # Add batch normalization when requested
            layers_phi_x.append(nn.BatchNorm1d(self.latent_shape))
        layers_phi_x = layers_phi_x + [
            nn.ReLU(),  # Non-linear activation to introduce non-linearity in the model
            nn.Linear(
                self.latent_shape, self.latent_shape
            ),  # Apply a linear transformation using its stored weights and biases
        ]

        # 1) The feature extractor of the input which extract features from x_t
        # Use nn.Sequential as an ordered container of modules, where data is passed through in the defined orderd
        self.phi_x = torch.nn.Sequential(*layers_phi_x)

        # 2) The feature extractor of the of the latent random variables which extract features from z_t
        # That is, on the stochastic vectors that will be sampled
        layers_phi_z = [nn.Linear(self.latent_shape, self.latent_shape)]
        if self.batch_norm:
            layers_phi_z.append(nn.BatchNorm1d(self.latent_shape))
        layers_phi_z = layers_phi_z + [
            nn.ReLU(),
            nn.Linear(self.latent_shape, self.latent_shape),
        ]
        self.phi_z = torch.nn.Sequential(*layers_phi_z)

        # The VRNN contains a VAE at every timestep. Unlike a standard VAE, the prior on the latent random
        # variable is not a standard Gaussian distribution, but the prior function can be a highly flexible function
        # such as neural networks. Define it here as a fully connected network with one hidden layer and ReLU activation
        # Prior network starts with the RNN (LSTM) hidden state h_{t-1}
        layers_prior = [nn.Linear(self.recurrent_shape, self.latent_shape)]
        if self.batch_norm:
            layers_prior.append(nn.BatchNorm1d(self.latent_shape))

        # and returns the Gaussian location and scale parameter vectors mu and sigma.
        # That is, the prior on the latent random variable is an isotropic Gaussian so, it is fully
        # characterised by its mean mu and variance sigma^2 (2*[latent feature dimensions] parameters)
        layers_prior = layers_prior + [
            nn.ReLU(),
            nn.Linear(self.latent_shape, 2 * self.latent_shape),
        ]
        self.prior_network = torch.nn.Sequential(*layers_prior)

        # Inference Network - The approximate posterior is a function of x_t and h_{t-1}. Encode the observation
        # x and RNN (LSTM) hidden state h_{y-1 }into the parameters of the posterior distribution:
        # q(z_t|x_t) = N(z_t|mu_{z,t}, diag(sigma^2_{z,t})), [mu_{z,t}, sigma_{z,t}] = phi^{enc}(phi^x(x_t), h_{t-1})
        # As before, define this network as a fully connected network with one hidden layer and ReLU activation.
        # Encoder network starts with feature extracted input x_t and the RNN (LSTM) hidden state h_{t-1}
        # and returns the Gaussian location and scale parameter vectors mu and sigma
        layers_encoder = [
            nn.Linear(self.latent_shape + self.recurrent_shape, self.latent_shape)
        ]
        if self.batch_norm:
            layers_encoder.append(nn.BatchNorm1d(self.latent_shape))
        layers_encoder = layers_encoder + [
            nn.ReLU(),
            nn.Linear(self.latent_shape, 2 * self.latent_shape),
        ]
        self.encoder = torch.nn.Sequential(*layers_encoder)

        # Generative Model - The generating distribution is conditioned on z_t and h_{t-1} such that it decodes
        # the latent sample z_t and the RNN (LSTM) hidden state h_{y-1 } into the parameters of the observation model.
        # The choice of the observation model depends on the nature of the features. Here there are three choises:
        # 1) When the observation model is a Bernoulli distribution:
        #     * p(x_t|z_t) = Prod_i B(x_{t,i} | theta_i), where theta_i = phi^{dec}(phi^z(z_t), h_{t-1})
        # 2) When the observation model is a multivariate isotropic Gaussian distribution:
        #     * p(x_t|z_t) = N(x_t|mu_{x,t}, diag(sigma^2_{x,t})), [mu_{x,t}, sigma_{x,t}] = phi^{dec}(phi^z(z_t), h_{t-1})
        # 3) When the observation model is a Gaussian mixture model, with M > 1 modes accounting for the multimodality:
        #     * p(x_t|z_t) = Sum_{m=1}^M pi_{t,m} N(x_t|mu_{x,t,m}, diag(sigma^2_{x,t,m})), where where the parameters
        #       [pi_t, mu_t, sigma_t] = [pi_{t,1}, ..., pi_{t,M}, mu_{t,1}, ..., mu_{t,M}, sigma_{t,1}, ..., sigma_{t,M}] =
        #       = phi^{dec}(phi^z(z_t), h_{t-1}) govern the form of the distribution (M+2*M*input_shape many).
        # As before, define this network as a fully connected network with one hidden layer and ReLU activation.
        # Decoder network begins with feature extracted latent random variables z_t and RNN hidden state h_{t-1}
        # and returns the Bernoulli probability parameter vector theta (one for each binary input?) <- Assuming Bernoulli
        # Change if Gaussian generating:
        layers_decoder = [
            nn.Linear(self.latent_shape + self.recurrent_shape, self.latent_shape)
        ]
        if self.batch_norm:
            layers_decoder.append(nn.BatchNorm1d(self.latent_shape))

        if self.generative_dist == "Bernoulli":
            # Return the input_shape many logits
            layers_decoder = layers_decoder + [
                nn.ReLU(),
                nn.Linear(self.latent_shape, self.input_shape),
            ]
        elif (
            self.generative_dist == "Isotropic_Gaussian"
            or self.generative_dist == "Diagonal"
        ):
            # Return the means and standard deviations (2*input_shape many parameters)
            layers_decoder = layers_decoder + [
                nn.ReLU(),
                nn.Linear(self.latent_shape, 2 * self.input_shape),
            ]
        elif self.generative_dist == "GMM" and self.GMM_equally_weighted:
            # Return the means and standard deviations for each component (GMM_components*2*input_shape many parameters)
            layers_decoder = layers_decoder + [
                nn.ReLU(),
                nn.Linear(
                    self.latent_shape, self.GMM_components * 2 * self.input_shape
                ),
            ]

        elif self.generative_dist == "GMM" and not self.GMM_equally_weighted:
            # Return the means and standard deviations for each component.
            # Also return the mixing probabilities (GMM_components + GMM_components*2*input_shape many parameters)
            layers_decoder = layers_decoder + [
                nn.ReLU(),
                nn.Linear(
                    self.latent_shape,
                    self.GMM_components + self.GMM_components * 2 * self.input_shape,
                ),
            ]

        elif self.generative_dist == "Gaussian":
            # Return the means and variance-covariance matrix (input_shape + (input_shape^2 + input_shape)/2 many parameters)
            layers_decoder = layers_decoder + [
                nn.ReLU(),
                nn.Linear(
                    self.latent_shape,
                    self.input_shape
                    + int((self.input_shape * self.input_shape + self.input_shape) / 2),
                ),
            ]
        else:
            print("Currently only implmented for 'Bernoulli', 'Diagonal', 'Gaussian'")
        self.decoder = torch.nn.Sequential(*layers_decoder)

        # Define an RNN that updates its hidden state with a recurrence equation that uses feature extracted x_t and z_t
        self.rnn = nn.LSTM(
            input_size=2 * self.latent_shape,
            hidden_size=self.recurrent_shape,
            num_layers=1,
            bidirectional=False,
        )
        torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)

    def prior(self, h, sigma_min=0.0, raw_sigma_bias=0.5):
        """Returns the distribution p(z_t)

        Parameters
        ----------
        h : Tensor
            The RNN hidden state h_{t-1}

        sigma_min : float
            Minimum standard deviation value (Defaults to 0.0)

        raw_sigma_bias : float
            Bias on top of the standard deviation value (Defaults to 0.5)

        Returns
        -------
        Distribution
            The distribution p(z_t)
        """
        # To get the parameters of the conditional prior distribution
        h_prior = self.prior_network(h)
        mu, sigma = h_prior.chunk(2, dim=-1)

        # Make sure that sigma is non-negative
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.maximum(
            torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min
        )

        # Return the prior distribution p(z_t)
        return ReparameterizedDiagonalGaussian(mu, sigma)

    def posterior(self, h, x_features, sigma_min=0.0, raw_sigma_bias=0.5):
        """Returns the approximate posterior distribution q(z_t|x_t) = N(z_t|mu_{z,t}, diag(sigma^2_{z,t}))

        Parameters
        ----------
        h : Tensor
            The RNN hidden state h_{t-1}

        x_features : Tensor
            The feature extracted input x_t

        sigma_min : float
            Minimum standard deviation value (Defaults to 0.0)

        raw_sigma_bias : float
            Bias on top of the standard deviation value (Defaults to 0.5)

        Returns
        -------
        Distribution
            The distribution q(z_t|x_t)
        """

        # Compute the parameters of the posterior - Encoder uses the RNN (LSTM) hidden state vector and the inputs
        encoder_input = torch.cat([h, x_features], dim=1)
        h_encoder = self.encoder(encoder_input)
        mu, sigma = h_encoder.chunk(
            2, dim=-1
        )  # Returns the mean and standard deviation after from the encoder

        # Make sure that sigma is non-negative
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.maximum(
            torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min
        )

        # Return a distribution q(z_t|x_t) = N(z_t|mu_{z,t}, diag(sigma^2_{z,t}))
        return ReparameterizedDiagonalGaussian(mu, sigma)

    def generative(self, z_features, h, sigma_min=0.000001, raw_sigma_bias=0.5, use_generative_bias=True):
        """Returns the generating distribution p(x_t|z_t)

        Parameters
        ----------
        z_features : Tensor
            The feature extracted latent random variables z_t

        h : Tensor
            The RNN hidden state h_{t-1}

        Returns
        -------
        Distribution
            The distribution p(x_t|z_t)
        """

        # Compute the parameters of the generating distribution - Decoder uses the RNN (LSTM) hidden state vector
        # and the feature extracted latent random variables
        decoder_input = torch.cat([z_features, h], dim=1)
        x_decoder = self.decoder(decoder_input)

        if self.generative_dist == "Bernoulli":
            # The log odds is non-linear in z_features and h
            if use_generative_bias:
                x_log_odds = x_decoder + self.generative_bias
            else:
                x_log_odds = x_decoder

            # Create a Bernoulli distribution parameterized by the log-odds of sampling one
            dist = Bernoulli(logits=x_log_odds)
        elif (
            self.generative_dist == "Isotropic_Gaussian"
            or self.generative_dist == "Diagonal"
        ):  # Changed to Diagonal Gaussian
            # Return the means and standard deviations from the decoder
            mu, sigma = x_decoder.chunk(2, dim=-1)

            # Make sure that sigma is non-negative
            sigma_min = torch.full_like(sigma, sigma_min)
            sigma = torch.maximum(
                torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min
            )
            location = mu
            if use_generative_bias:
                location = mu + self.generative_bias
            dist = MultivariateNormal(
                loc=location,
                covariance_matrix=torch.diag_embed(
                    torch.square(sigma), offset=0, dim1=-2, dim2=-1
                ),
            )

        elif self.generative_dist == "GMM" and self.GMM_equally_weighted:
            # Return all (for every component) means and standard deviations from the decoder
            mu, sigma = x_decoder.chunk(2, dim=-1)

            # Make sure that sigma is non-negative
            sigma_min = torch.full_like(sigma, sigma_min)
            sigma = torch.maximum(
                torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min
            )
            # sigma = torch.maximum(sigma, sigma_min)

            # Get the right shapes for the mean and stanard deviation alues
            batch_n = mu.shape[0]
            mu = mu.view(batch_n, self.GMM_components, self.input_shape)
            sigma = sigma.view(batch_n, self.GMM_components, self.input_shape)

            # Construct a batch of Gaussian Mixture Modles in input_shape-D consisting of
            # GMM_components equally weighted input_shape-D Gaussian distributions
            mix_probs = torch.ones(batch_n, self.GMM_components, device=self.device)
            mix = D.Categorical(mix_probs)
            comp = D.Independent(D.Normal(mu, sigma), 1)
            gmm = MixtureSameFamily(mix, comp)
            dist = gmm

        elif self.generative_dist == "GMM" and not self.GMM_equally_weighted:
            # Get the mixing probabilities from the decoder
            mix_probs = x_decoder[:, : self.GMM_components]

            # Return all (for every component) means and standard deviations from the decoder
            mu, sigma = x_decoder[:, self.GMM_components :].chunk(2, dim=-1)

            # Make sure that sigma is non-negative
            sigma_min = torch.full_like(sigma, sigma_min)
            sigma = torch.maximum(
                torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min
            )
            # sigma = torch.maximum(sigma, sigma_min)

            # Get the right shapes for the mean and stanard deviation alues
            batch_n = mu.shape[0]
            mu = mu.view(batch_n, self.GMM_components, self.input_shape)
            sigma = sigma.view(batch_n, self.GMM_components, self.input_shape)

            # Ensure that the mixing probabilities are between zero and one and sum to one
            mix_probs = F.softmax(mix_probs, dim=1)

            # Construct a batch of Gaussian Mixture Modles in input_shape-D consisting of
            # GMM_components equally weighted input_shape-D Gaussian distributions
            mix = D.Categorical(mix_probs)
            comp = D.Independent(D.Normal(mu, sigma), 1)
            gmm = MixtureSameFamily(mix, comp)
            dist = gmm
        else:
            print("Currently only implmented for 'Bernoulli', 'Diagonal', and 'GMM'")

        # Return a distribution p(x_t|z_t)
        return dist

    def forward(
        self,
        inputs,
        targets,
        logits=None,
        hs=None,
        zs=None,
        z_mus=None,
        z_sigmas=None,
        obs_mus=None,
        obs_Sigmas=None,
        obs_probs=None,
        use_generative_bias=True,
    ):
        """Computes the log probabilities that can be used in the VRNN loss function

        Parameters
        ----------
        input : Tensor
            The model inputs (features)

        targets : Tensor
            The true/actual targets (labels)

        logits :
            When not None, the logits of the Bernoulli distribution are saved and returned (Defaults to None)

        hs :
            When not None, the LSTM hidden states are saved and returned (Defaults to None)

        zs :
            When not None, the sampled latent random variables are saved and returned (Defaults to None)

        z_mus :
            When not None, the approximate posterior distributions mean are saved and returned (Defaults to None)

        z_sigmas :
            When not None, the approximate posterior distributions SD are saved and returned (Defaults to None)

        obs_mus :
            When not None, the location values of the Gaussian observation model are saved and returned (Defaults to None)

        obs_Sigmas :
            When not None, the variance-covariance matrix values of the Gaussian observation model are saved and returned (Defaults to None)

        obs_probs :
            When not None, the mixing probabilities values of the GMM observation model are saved and returned (Defaults to None)

        Returns
        -------
        list
            Log probability of observing the target given the generating distribution

        list
            Log probability of observing the sampled latent random variables under the prior distribution

        list
            Log probability of observing the sampled latent random variables under the approximate posterior
        """

        # Get sizes from input and make permutations
        batch_size, seq_len, data_dim = inputs.shape
        inputs = inputs.permute(
            1, 0, 2
        )  # The dimensions are seq_len X batch_size X data_dim
        targets = targets.permute(1, 0, 2)

        # Initialize LSTM output, hidden state and cell state (for t = 0) - The dimensions are batch_size X recurrent_shape
        output_t = torch.zeros(batch_size, self.recurrent_shape, device=self.device)
        h_t = torch.zeros(1, batch_size, self.recurrent_shape, device=self.device)
        c_t = torch.zeros(1, batch_size, self.recurrent_shape, device=self.device)

        # Initialize variables to keep track of
        log_px, log_pz, log_qz = [], [], []

        # Loop through each time step ("sequence length" always one) - Cannot parallelize with current implementation
        for t in range(seq_len):
            # Take one time step at a time since it starts out sampling the latent random variables that are needed to
            # update the RNN (LSTM). Similarly, the updated RNN hidden state is required to produce the next random vector.
            # That is: Update RNN -> sample random vetor -> Update RNN -> sample random vetor ... (Cannot parallelize)
            x_t = inputs[
                t, :, :
            ]  # Inputs at time t: The dimensions are batch_size X data_dim
            y_t = targets[t, :, :]

            # Use the feature extractor of the input which extract features from x_t
            x_t_features = self.phi_x(
                x_t
            )  # Embed input: The dimensions are batch_size X latent_dim

            # Create the prior distribution p(z_t)
            # pz = self._prior(out) #out is batch X latent
            pz_t = self.prior(output_t)  # output_t at this point is really output_{t-1}

            # Create the approximate posterior q(z_t|x_t)
            # qz = self.posterior(out, x_hat, prior_mu=pz.mu)
            qz_t = self.posterior(
                output_t, x_t_features
            )  # output_t at this point is still output_{t-1}

            # Sample and embed z_t from the inference approximate posterior q(z_t|x_t)
            z_t = (
                qz_t.rsample()
            )  # Sample the posterior using the reparameterization trick: z ~ q(z_t|x_t)

            # Use the eature extractor of the of the latent random variables which extract features from z_t
            z_features = self.phi_z(
                z_t
            )  # Embed latent variables: The dimensions are batch_size X latent_dim

            # Create the observation model (generating distribution) p(x_t|z_t)
            px_t = self.generative(
                z_features, output_t, use_generative_bias=use_generative_bias
            )  # output_t at this point is still output_{t-1}

            # Update the recurrence - Update hidden state output_{t-1} to output_t
            rnn_input_t = torch.cat([x_t_features, z_features], dim=1)
            rnn_input_t = rnn_input_t.unsqueeze(
                0
            )  # The dimensions are 1 (seq_len) X batch_size X 2*latent_dim
            output_t, (h_t, c_t) = self.rnn(
                rnn_input_t, (h_t, c_t)
            )  # Update the LSTM. output_t is 1 X batch_size X latent_dim
            output_t = output_t.squeeze(
                axis=0
            )  # The output_t dimensions are now batch_size X recurrent_shape

            # Use the created distributions to evaluate the log probabilities that will be used in the loss function
            if self.generative_dist == "Bernoulli":
                # Get the log probability of observing the tarrget given the Bernoulli distribution defined above
                # Sum over input dimension to get the joint log probablity for the [data dimension] many Bernoulli distributions
                log_px.append(
                    px_t.log_prob(y_t).sum(dim=1)
                )  # Aim to maximize this log probability
            else:
                # Use multivariate distribution to get the log likelihood directly
                log_px.append(px_t.log_prob(y_t))

            # Log probability of observing the sampled latent random variables in the prior and approximate posterior
            # Want the probability to be similar under those two distributions. That is, the posterior should look
            # similar to the prior distribution, limiting the flexibility of the posterior.
            log_pz.append(pz_t.log_prob(z_t).sum(dim=1))
            log_qz.append(
                qz_t.log_prob(z_t).sum(dim=1)
            )  # Check why to sum over the input feature dimension

            # Save the requested variables
            if logits is not None:
                # Save the logits from the Bernoulli distribution - To look at that distribution
                logits[t, :, :] = px_t.logits
            if hs is not None:  # Save the hidden state from the LSTM cells
                hs[t, :, :] = output_t
            if zs is not None:
                zs[t, :, :] = z_t  # Save the sampled latent random variables
            if z_mus is not None:
                # Save the approximate posterior distributions mean and standard deviation
                z_mus[t, :, :] = qz_t.mu
            if z_sigmas is not None:
                z_sigmas[t, :, :] = qz_t.sigma
            if obs_mus is not None:
                if self.generative_dist == "GMM":
                    # Save the observation model mean and variance-covariance matrix
                    obs_mus[t, :, :, :] = px_t.component_distribution.base_dist.loc
                else:
                    # Save the observation model mean and variance-covariance matrix
                    obs_mus[t, :, :] = px_t.loc
            if obs_Sigmas is not None:
                if self.generative_dist == "GMM":
                    obs_Sigmas[t, :, :, :] = px_t.component_distribution.base_dist.scale
                else:
                    obs_Sigmas[t, :, :, :] = px_t.covariance_matrix
            if obs_probs is not None:
                # Track the mixing probabilities
                obs_probs[t, :, :] = px_t.mixture_distribution.probs
        return log_px, log_pz, log_qz, logits, hs, zs, z_mus, z_sigmas
