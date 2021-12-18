import numpy as np
import os
import sys
import scipy
import math
import operator as op
from functools import reduce
import multiprocessing 
from joblib import Parallel, delayed
from Config import config

class anomalydetection():
    def __init__(self, lat_dim, runLenghts=[], trendLenghts=[], p0=0.1, p1=0.3, method='CUSUM', pfa=1e-9):
        super(anomalydetection, self).__init__()

        self.method = method
        self.pfa = pfa
        self.lat_dim = lat_dim
        
        if self.method=='6Sigma':
            self.runLenghts = runLenghts #[5, 9, 15]
            self.trendLenghts = trendLenghts #[6, 9, 12]
            
            c = np.log10(1-self.pfa)/(len(runLenghts)+len(trendLenghts))
            self.epsilon = 1-10**c
        elif self.method=='CUSUM':
            self.p0 = p0
            self.p1 = p1
            
            r1 = -np.log((1-p1)/(1-p0))
            r2 = np.log((p1*(1-p0))/(p0*(1-p1)))
            self.gamma = r1/r2
            
            e_p0 =  0.410 - 0.0842*np.log(p0) - 0.0391*(np.log(p0)**3) - 0.00376 *(np.log(p0)**4) - 0.000008 * (np.log(p0)**7)
            q0 = 1-p0
        
            def func(x,ANOS,r1,r2,p0):
                return ANOS - (np.exp(x * r2) - x * r2 -1)/np.abs(r2*p0-r1)

            adjusted_h_b = scipy.optimize.fsolve(func, 30, args=(1/self.pfa, r1, r2, self.p0))[0]

            self.threshold = adjusted_h_b - e_p0 * np.sqrt(p0*q0)
            self.probDetection = (np.exp(-adjusted_h_b * r2) + adjusted_h_b * r2 -1)/np.abs(r2*self.p1-r1)
        elif self.method=='AContrario':
            self.max_segment_length = config.MAX_SEGMENT_LENGTH

    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom

    def nonzero_segments(self, x_):
        """Return list of consecutive nonzeros from x_"""
        run = []
        result = []
        for d_i in range(len(x_)):
            if x_[d_i] != 0:
                run.append(d_i)
            else:
                if len(run) != 0:
                    result.append(run)
                    run = []
        if len(run) != 0:
            result.append(run)
            run = []
        return result

    def NFA(self, ns,k,NS): #cite https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/geotracknet.py
        """Number of False Alarms"""
        B = 0
        for t in range(k,ns+1):
            B += self.ncr(ns,t)*(0.1**t)*(0.9**(ns-t))
        return NS*B


    def contrario_detection(self, v_A_,epsilon=1e-9): #cite https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/geotracknet.py
        """
        A contrario detection algorithms
        INPUT:
            v_A_: abnormal point indicator vector
            epsilon: threshold
        OUTPUT:
            v_anomalies: abnormal segment indicator vector
        """
        v_anomalies = np.zeros(len(v_A_))
        max_seq_len = len(v_A_)
        Ns = max_seq_len*(max_seq_len+1)*0.5
        for d_ns in range(max_seq_len,0,-1): #loop over all possible segments lengths
            for d_ci in range(max_seq_len+1-d_ns): #loop all over possible segment start positions
                v_xi = v_A_[d_ci:d_ci+d_ns] #Take out the current segment
                d_k_xi = int(np.count_nonzero(v_xi)) #number of abnormal points in segment
                if self.NFA(d_ns,d_k_xi, Ns)<epsilon: #Is number of abnormal points higher than expected
                    v_anomalies[d_ci:d_ci+d_ns] = 1 #Mark the abnormal segment
        return v_anomalies #return vector of abnormal segments

    def determineOutliers(self, log_probs, activatedBins, Map_logprob, lengths):

        #log_probs: numpy array of shape [time, dataset_size]
        #activatedBins: numpy array of shape [time, dataset_size, 4]
        #Map_logprob: dict with keys of the form 'row#,col#' eg. 5,6 for the 6th row (latitude bin) and 7th column (longitude bin). Values are the log_probs in this bin
        #The dimension size of the latitude binning

        #return array of length dataset_size. 
        
        outliers = []
        abnormal_segments = []
        abnormal_points = []
        for i in range(log_probs.shape[1]):
            if self.method=='6Sigma':
                outlier, v_anomalies, p_anomalies = self.isTrackOutlier6Sigma(log_probs[:,i], activatedBins[:,i,:], Map_logprob, lengths[i])
            elif self.method=='CUSUM':
                outlier, v_anomalies, p_anomalies = self.isTrackOutlierCUSUM(log_probs[:,i], activatedBins[:,i,:], Map_logprob, lengths[i])
            elif self.method=='AContrario':
                outlier, v_anomalies, p_anomalies = self.isTrackOutlier(log_probs[:,i], activatedBins[:,i,:], Map_logprob, lengths[i])
            outliers.append(outlier)
            abnormal_segments.append(v_anomalies)
            abnormal_points.append(p_anomalies)
        
        return outliers, abnormal_segments, abnormal_points

    def determineOutliersParallel(self, log_probs, activatedBins, Map_logprob, lengths, n_cores=None):

        #OUTDATED

        #log_probs: numpy array of shape [time, dataset_size]
        #activatedBins: numpy array of shape [time, dataset_size, 4]
        #Map_logprob: dict with keys of the form 'row#,col#' eg. 5,6 for the 6th row (latitude bin) and 7th column (longitude bin). Values are the log_probs in this bin
        #The dimension size of the latitude binning

        def my_function(i, log_probs, activatedBins, Map_logprob, lengths):
            outlier, v_anomalies = self.isTrackOutlier(log_probs[:,i], activatedBins[:,i,:], Map_logprob, lengths[i])
            return i, outlier, v_anomalies
        
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()

        processed_list = Parallel(n_jobs=n_cores)(delayed(my_function)(i,log_probs, activatedBins, Map_logprob, lengths) for i in range(log_probs.shape[1]))

        processed_list = list(map(list, zip(*processed_list)))
        index, outliers, anomalies = processed_list
        
        return index, outliers, anomalies
    
    def getDetectionThresholds(self, trackLength):
        
        runThresholds = []
        for runLength in self.runLenghts:
            c = np.log10(1-self.epsilon)/(trackLength-runLength+1)
            c = 10**c
            c = np.log10(1-c)/runLength
            runThresholds.append(10**c)
        
        trendThresholds = []
        for trendLength in self.trendLenghts:
            f = math.factorial(trendLength)
            c = np.log10(1-self.epsilon)/(trackLength-trendLength+1)
            c = 10**c
            c = np.log10(f-f*c)/trendLength
            trendThresholds.append(10**c)
        
        return np.array(runThresholds), np.array(trendThresholds)
    
    def detectRuns(self, cdf,runLength):

        #Modify threshold to account for track length
        c = np.log10(1-self.epsilon)/(len(cdf)-runLength+1)
        c = 10**c
        c = np.log10(1-c)/runLength
        threshold = 10**c

        lower = cdf < threshold

        v_A = np.zeros(len(cdf))
        cur_run = 0
        for i, test in enumerate(lower):
            if test:
                cur_run += 1
                if cur_run >= runLength:
                    for j in range(cur_run):
                        v_A[i-j] = 1
            else:
                cur_run = 0

        return v_A

    def detectTrend(self, cdf,trendLength):

        #Modify threshold to account for track length
        f = math.factorial(trendLength)
        c = np.log10(1-self.epsilon)/(len(cdf)-trendLength+1)
        c = 10**c
        c = np.log10(f-f*c)/trendLength
        threshold = 10**c

        lower = cdf < threshold

        v_A = np.zeros(len(cdf))
        cur_run = 0

        if lower[0]==1:
            cur_run = 1

        for i, test in enumerate(lower[1:]):
            if test and cdf[i-1]>cdf[i]:
                cur_run += 1
                if cur_run >= trendLength:
                    for j in range(cur_run):
                        v_A[i-j] = 1
            else:
                cur_run = 0

        return v_A

    def isTrackOutlier6Sigma(self, log_probs, activatedBins, Map_logprob, trackLength):
        #log_probs: numpy array of shape [time]
        #activatedBins: numpy array of shape [time, 4]
        #Map_logprob: dict with keys of the form 'row#,col#' eg. 5,6 for the 6th row (latitude bin) and 7th column (longitude bin). Values are the log_probs in this bin
        #The dimension size of the latitude binning

        v_A = np.zeros(trackLength)
        cdf = np.zeros(trackLength)
        for t in range(trackLength):
            activatedlat = activatedBins[t,0] 
            activatedlon = activatedBins[t,1]-self.lat_dim

            local_log_probs = Map_logprob[ str(int(activatedlat))+","+str(int(activatedlon))]     

            if len(local_log_probs) < 2:
                v_A[t] = 1
            else:
                kernel = scipy.stats.gaussian_kde(local_log_probs)
                cdf[t] = kernel.integrate_box_1d(-np.inf,log_probs[t])

        #Use detection rules:
        #Runs p^n
        for runLength in self.runLenghts:
            if trackLength >= runLength:
                v_A = np.logical_or(v_A,self.detectRuns(cdf,runLength))
        #Trends 1/(n!)*p^n
        for trendLength in self.trendLenghts:
            if trackLength >= trendLength:
                v_A = np.logical_or(v_A,self.detectTrend(cdf,trendLength))

        if len(self.nonzero_segments(v_A)) > 0: 
            return True, v_A, cdf
        else:
            return False, v_A, cdf
        
    def isTrackOutlierCUSUM(self, log_probs, activatedBins, Map_logprob, trackLength):
        #log_probs: numpy array of shape [time]
        #activatedBins: numpy array of shape [time, 4]
        #Map_logprob: dict with keys of the form 'row#,col#' eg. 5,6 for the 6th row (latitude bin) and 7th column (longitude bin). Values are the log_probs in this bin
        #The dimension size of the latitude binning

        v_A = np.zeros(trackLength)
        Bk = np.zeros(trackLength)
        B = 0
        for t in range(trackLength):
            activatedlat = activatedBins[t,0] 
            activatedlon = activatedBins[t,1]-self.lat_dim

            local_log_probs = Map_logprob[ str(int(activatedlat))+","+str(int(activatedlon))]     

            if len(local_log_probs) < 2:
                v_A[t] = 1
            else:
                kernel = scipy.stats.gaussian_kde(local_log_probs)
                cdf = kernel.integrate_box_1d(-np.inf,log_probs[t])
                
                if cdf < 0.1:
                    v_A[t] = 1
                
            #Update Bk
            B = np.maximum(0,B) + v_A[t] - self.gamma
            Bk[t] = B

        test = (Bk >= self.threshold).astype(int)
            
        if len(self.nonzero_segments(test)) > 0: 
            return True, Bk, v_A
        else:
            return False, Bk, v_A
    
    def isTrackOutlier(self, log_probs, activatedBins, Map_logprob, trackLength):
        #log_probs: numpy array of shape [time]
        #activatedBins: numpy array of shape [time, 4]
        #Map_logprob: dict with keys of the form 'row#,col#' eg. 5,6 for the 6th row (latitude bin) and 7th column (longitude bin). Values are the log_probs in this bin
        #The dimension size of the latitude binning
                
        v_A = np.zeros(trackLength)
        for t in range(trackLength):
            activatedlat = activatedBins[t,0] 
            activatedlon = activatedBins[t,1]-self.lat_dim
                
            local_log_probs = Map_logprob[ str(int(activatedlat))+","+str(int(activatedlon))]     

            if len(local_log_probs) < 2:
                v_A[t] = 2
            else:
                kernel = scipy.stats.gaussian_kde(local_log_probs)
                cdf = kernel.integrate_box_1d(-np.inf,log_probs[t])
                if cdf < 0.1:
                    v_A[t] = 1
        
        v_anomalies = np.zeros(len(v_A))
        end_index = 1 if len(v_A)+1-self.max_segment_length < 1 else len(v_A)+1-self.max_segment_length
        for seq_i in range(0,end_index):
            v_A_segment = v_A if end_index==1 else v_A[seg_i:seg_i+self.max_segment_length] #Take out segment of tracks longer than max_segment_length
            v_anomalies_i = self.contrario_detection(v_A) #vector of abnormal segments. vector of length #updates indicating if the update is part of abnormal segments
            if end_index == 1:
                v_anomalies[v_anomalies_i==1] = 1
            else:
                v_anomalies[seg_i:seg_i+self.max_segment_length][v_anomalies_i==1] = 1
        
        if len(self.nonzero_segments(v_anomalies)) > 0: 
            return True, v_anomalies, v_A
        else:
            return False, v_anomalies, v_A

