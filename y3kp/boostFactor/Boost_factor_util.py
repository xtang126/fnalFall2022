#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ebtihal Abdelaziz
# This file includes multiple modules that are utilized in boost_factor1 and Boost_factor_like. 

import numpy as np

def read_desy1_data(path = "/global/cfs/cdirs/des/jesteves/data/boost_factor/y1/profiles"):
    B= dict()
    # z = np.round((np.unique(options["BoostFactor","zo_low"]) +np.unique(options["BoostFactor","zo_high"]))/2.,2)
    # l = np.round((options["BoostFactor","lo_low"] +options["BoostFactor","lo_high"])/2.,2)
    # bins = Boost_factor_util.lookup_table(l,z)

    for L in range(7):
        for Z in range(3):
            R,data_vector,sigma_B = np.genfromtxt(path+"/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat".format(l = L, z = Z),unpack=True)
            covariance = np.genfromtxt(path+"/full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat".format(l = L, z = Z),unpack=True)
# the error bars on the last two points of the data are tiny, so the MCMC is doing its best to fit for them. (They are tiny due to bad data)
#The following rewrites them. 
            ix=np.nonzero(sigma_B < 10**(-6))
            sigma_B[ix]=10**6
            covariance[ix,:]= 10**6
            covariance[:,ix]=10**6
            B[L,Z]=data_vector,sigma_B,covariance
    return R,B
    
def Boost_Factor_param(Z,L,Rs,alpha_Rs,beta_Rs,B0,alpha_B0,beta_B0):
    """This function takes the priors from the values file and calculates rs and b0 values for the model given redshift and lambda
    """
    rs =Rs*((Z)/(1+0.3))**alpha_Rs*(L/30.)**beta_Rs
    b0= B0*((Z)/(1+0.3))**alpha_B0*(L/30.)**beta_B0
    return rs,b0

def Boost_Factor_log_param(Z,L,Rs0,alpha_Rs,beta_Rs,B0,alpha_B0,beta_B0):
    """This function takes the priors from the values file and calculates rs and b0 values for the model given redshift and lambda
    """
    rslog = np.log(Rs0) + alpha_Rs*np.log((1+Z)/(1+0.3)) + beta_Rs*np.log(L/30.)
    b0log = np.log(B0) + alpha_B0*np.log((1+Z)/(1+0.3)) + beta_B0*np.log(L/30.)
    return rslog, b0log

def Boost_Factor_Model(R, Rs, B0):
    """This function is where the boost factor model is calculated. It takes np.arrays for the radius R, Rs and B0."""
    x = R/Rs
    fx = np.zeros(R.size)
    ix,= np.where(x>1)
    fx[ix] = ((np.arctan(np.sqrt((x[ix])**2 -1)))/np.sqrt((x[ix])**2 -1))
    ix,= np.where(x==1)
    fx[ix] = 1
    ix,= np.where(x<1)
    fx[ix] = (np.arctanh(np.sqrt(1 -x[ix]**2))/np.sqrt(1 -x[ix]**2))
    B= 1+B0*((1-fx)/(x**2-1))
    ix,=np.where(np.isnan(B))
    B[ix]=(B0+3)/3
    return B

def lookup_table(a,b):
    """This function takes two np.arrays and turns them into dictionary with keys corresponding to the indices of items in each array."""
    a = np.round(a,2)
    b = np.round(b,2)
    bins = dict()
    bin_a = 0
    for A in a:
        bin_b = 0
        for B in b:
            bins[A,B] = bin_a, bin_b
            bin_b +=1
        bin_a +=1
    return bins

def fake_data_vector(R,z,l):
    """This function was used to create mock data for the boost factor model given np.arrays of radius R, redshift z, and rishness l."""
    B= dict()  
    variance = np.ones(R.size)*0.1**2
#creating data vector
    bins = lookup_table(z,l)
    for Z in z:
        Rs = (1-Z)
        for L in l:
            #print(bins.keys())
            This_bin = bins[Z,L]
            B0= (1000+L)**(1/2) 
            sigma_B = np.random.normal(loc=0, scale=variance**(1/2), size=R.size)
            #adding scatter
            data_vector = Boost_Factor_Model(R, Rs, B0) + sigma_B
            #thinking about error bars. 
            sigma_B = np.random.normal(loc=0, scale=variance**(1/2), size=R.size)
            B[This_bin]=data_vector,sigma_B 
    return B,z,l,bins
