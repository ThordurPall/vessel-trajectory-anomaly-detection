# Vessel Trajectory Anomaly Detection
This is a Git repository accompanying the "Vessel Trajectory Anomaly Detection Using Deep Learning Methods" master project. Based on work done in https://arxiv.org/abs/1912.00682, this project constructs trajectory-based models to discriminate between legitimate fishing vessel activities and anomalous events using historical AIS data. The proposed approach uses a variational recurrent neural network (VRNN) that learns a distribution representing the data that can state the reconstruction likelihood of AIS trajectories. Then for anomaly detection, a geographically-dependent a contrario detection rule is used to account for the fact that the learnt distribution may be location-dependent. The project objectives are to

 - Analyse the VRNN-based model performance for fishing vessels around Denmark.
 - Evaluate the generalisation of the VRNN-based model to a larger ROI.
 - Examine the impact of different input AIS data representations, in particular of moving from discrete to continuous inputs. The image below gives the intuition about the difference between the continuous inputs and the discrete binned concatenated vector of the one-hot vectors.
 - Assess the impact of switching from a diagonal multivariate Gaussian distribution to a Gaussian mixture model for the generating distribution of the VRNN.


![Explain binning](https://github.com/ThordurPall/vessel-trajectory-anomaly-detection/blob/main/figures/regions/Bornholm/Explain_Binning_Bornholm.png?raw=true)
