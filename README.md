# Vessel Trajectory Anomaly Detection
This is a Git repository accompanying the "Vessel Trajectory Anomaly Detection Using Deep Learning Methods" master project. Based on work done in https://arxiv.org/abs/1912.00682, this project constructs trajectory-based models to discriminate between legitimate fishing vessel activities and anomalous events using historical AIS data. The proposed approach uses a variational recurrent neural network (VRNN) that learns a distribution representing the data that can state the reconstruction likelihood of AIS trajectories. Then for anomaly detection, a geographically-dependent a contrario detection rule is used to account for the fact that the learnt distribution may be location-dependent. The project objectives are to

 - Analyse the VRNN-based model performance for fishing vessels around Denmark.
 - Evaluate the generalisation of the VRNN-based model to a larger ROI.
 - Examine the impact of different input AIS data representations (discrete vs. continuous inputs).
 - Assess the impact of switching from a Gaussian to a GMM for the generating distribution of the VRNN.

 The following image gives the intuition about the difference between the continuous inputs and the discrete binned concatenated vector of the one-hot vectors:

![Explain binning](https://github.com/ThordurPall/vessel-trajectory-anomaly-detection/blob/main/figures/regions/Bornholm/Explain_Binning_Bornholm.png?raw=true)


Please note that some code in this project builds upon excellent work done by Kristoffer Vinther Olesen (in particular the util files). The overall directory structure is as follows: 

```
├── data
│   ├── processed      <- The final data sets used for modelling.
│   └── raw            <- The original, immutable data dump.
│
├── figures            <- Figures used for reporting and providing explanations.
│
├── models             <- Trained and serialized models and their learning curves.
│
├── notebooks          <- Jupyter notebooks (only used for presenting examples during meetings).
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module.
│   │
│   ├── data           <- Scripts to preprocess data for modelling.
│   │
│   ├── models         <- Scripts to train models and then use the trained models.
│   │
│   ├── report         <- Scripts to create report ready figures.
│   │
│   ├── util           <- Util files used throughout the project.
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
│
└── requirements.txt   <- The requirements file for reproducing the results environment.
```
