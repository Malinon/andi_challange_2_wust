# A simple program to test the of estimation procedures.
#
# The program uses the trajectories stored in the datasets 
# subdirectory.
#
# JS, 13.03.2024

from andi_datasets.models_phenom import models_phenom

# auxiliaries
import glob
import re
import sys
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp

__DT = 0.1


def moving_average(data, window_size):
    """
    Apply a moving average filter to the data.
    
    Parameters:
        data (numpy.ndarray): Input data.
        window_size (int): Size of the moving average window.
        
    Returns:
        numpy.ndarray: Denoised data.
    """
    cumsum = np.cumsum(data,axis=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def empirical_msd(trajectory,lag,k):
    """Generate empirical MSD for a single lag.

    :param trajectory: numpy.ndarray, list of (x,y) coordinates of a particle
    :param lag: int, time lag
    :param k: int, power of msd
    :return: msd for given lag
    """
    x = trajectory[:,0]
    y = trajectory[:,1]

    N = len(x)
    x1 = np.array(x[:N - lag])
    x2 = np.array(x[lag:N])
    y1 = np.array(y[:N - lag])
    y2 = np.array(y[lag:N])
    #c = np.sqrt(np.array(list(x2 - x1)) ** 2 + np.array(list(y2 - y1))**2) ** k
    c = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)**k
    r = np.mean(c)
    return r
    
def generate_empirical_msd(trajectory,mlag,k=2):
    """Generate empirical MSD for a list of lags from 1 to mlag.

    :param trajectory: numpy.ndarray, array of (x,y) coordinates
    :param mlag: int, max time lag for msd
    :param k: int, power of msd
    :return: array of empirical msd
    """

    x = trajectory[:,0]
    y = trajectory[:,1]

    r = []
    for lag in range(1,mlag+1):
        r.append(empirical_msd(trajectory, lag,k))
    #print(len(trajectory), " ",mlag , " msd ",r)
    return np.array(r)

def generate_theoretical_msd_with_noise(n_list, D, dt, alpha, sigma_2, dim=2):
    """
    Function for generating msd of anomalous diffusion
    :param n_list: number of points in msd
    :param D: float, diffusion coefficient
    :param dt: float, time between steps
    :param alpha: float, anomalous exponent (alpha<1)
    :param sigma_2: float, noise
    :param dim: int, dimension (1,2,3)
    :return: array of theoretical msd
    """
    r = 2 * dim * D * (dt * n_list) ** alpha + sigma_2
    return r



def estimate_with_noise_1(trajectory,mlag,k=2):
    """
    The estimation of diffusion exponent with noise, according to method I from
    Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
    "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
    Phys. Rev. E 98, 062139 (2018).
    """
    mlag = min(mlag, len(trajectory) - 1)
    max_number_of_points_in_msd = mlag+1 #inaczej niż w oryginale
    log_msd = np.log(generate_empirical_msd(trajectory,mlag,k))
    log_n = np.array([np.log(i) for i in range(1,max_number_of_points_in_msd)]) 
    alpha = ((max_number_of_points_in_msd+1) * np.sum(log_n * log_msd) - np.sum(log_n * np.sum(log_msd))) / \
                    ((max_number_of_points_in_msd+1) * np.sum(log_n ** 2) - (np.sum(log_n)) ** 2)
    D = np.exp(log_msd[0])/4                
    return D, alpha

    
def estimate_with_noise_2(trajectory,mlag,k=2):
    """
    The estimation of diffusion exponent with noise, according to method II from
    Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
    "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
    Phys. Rev. E 98, 062139 (2018).
    """
    dt = __DT
    mlag = min(mlag, len(trajectory) - 1)
    empirical_msd = generate_empirical_msd(trajectory,mlag,k)
    n_list = np.array(range(1,mlag+1)) #inaczej niż w oryginale
    s2_max = empirical_msd[0]
    alpha_0 = 1.0
    D_0 = empirical_msd[0]/4  #INACZEJ NIŻ W ORYGINALE 
    s2_0 = empirical_msd[0]/2
    eps = 0.001

    N = len(trajectory[:,0])
    #dt = 
    try:
        popt, cov = sp.optimize.curve_fit(
                  lambda x, D, a, s2: generate_theoretical_msd_with_noise(x, D, dt, a, s2),
                  n_list, empirical_msd, p0 =(D_0,alpha_0,s2_0) , bounds=([0,0,0],[np.inf,2,s2_max]), 
                  method = 'trf', ftol = eps)                   
        D_est = popt[0]
        alpha_est = popt[1]
        return D_est, alpha_est
    except RuntimeError:
        return 0, 0
    
def estimate_with_noise_3(trajectory,mlag,k=2):
    """
    The estimation of diffusion exponent with noise, according to method III from with n_min fixed from
    Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
    "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
    Phys. Rev. E 98, 062139 (2018).
    """
    
    mlag = min(mlag, len(trajectory) - 1)
    empirical_msd = generate_empirical_msd(trajectory,mlag,k)
    n_list = np.array(range(1,mlag+1)) #POTENCJALNY BLAD w RANGE
    alpha_0 = 1
    D_0 = empirical_msd[0]/4 #INACZEJ NIŻ W ORYGINALE (BLAD?)
    eps = 0.01

    def msd_fitting(n_list,de,dt,al):
        r = 4 * de * dt ** al * (n_list - 1) ** al
        return r

    popt, cov = sp.optimize.curve_fit(
                    lambda x, D, a: msd_fitting(x, D, __DT, a),
                    n_list, empirical_msd, p0 =(D_0,alpha_0), bounds=([0,0],[np.inf,2]), 
                    method = 'dogbox', ftol = eps)   
        
    D_est = popt[0]
    alpha_est = popt[1]
        
    return D_est, alpha_est
    

def msd_2d(trajectory):
    """
    Estimate the diffusion coefficient  and the anomalous exponent of 2D trajectory.
    
    Parameters:
        trajectory (numpy.ndarray): Trajectory of the particle (2D).
        
    Returns:
        float: Estimated diffusion coefficient.
        float: Estimated anomalous exponent.
    """
    T = len(trajectory)
    msd = np.zeros(T)
    for i in range(1,T):
        diff = trajectory[i:] - trajectory[:-i]
        sd = np.sum(diff**2, axis=1)
        msd[i] = np.mean(sd)

        #traj[i:] - traj[:-i] subtracts each row of the trajectory matrix 
        #from every other row i steps ahead. It essentially calculates 
        #the displacement vectors between successive positions of the particle 
        #separated by i time steps. So, for each i, it computes the displacement 
        #between points (t[i], t[0]), (t[i+1], t[1]), ..., (t[-1], t[-1-i]).

    msd = msd[1:]
    time_steps = np.arange(1, T)
    coeffs = np.polyfit(np.log(time_steps), np.log(msd), 1)
    anomalous_exponent = coeffs[0] / 2
    diffusion_coefficient = np.exp(coeffs[1])
    
    return diffusion_coefficient, anomalous_exponent


def psd_2d(trajectory, dt, window_size=5):
    """
    Estimate D and alpha of a 2D trajectory.
    
    Parameters:
        trajectory (numpy.ndarray): Trajectory of the particle (2D).
        dt (float): Time step.
        window_size (int): Size of the moving average window for denoising.
        
    Returns:
        float: Estimated diffusion coefficient.
        float: Estimated anomalous exponent.
    """
    
    T = len(trajectory)
    #denoised = moving_average(trajectory, window_size)
    denoised = np.stack([moving_average(trajectory[:, i], window_size) for i in range(trajectory.shape[1])], axis=1)
    #calculate PSD using the Welch method
    freq, psd_x = welch(denoised[:, 0], fs=1/dt, nperseg=T//2)
    freq, psd_y = welch(denoised[:, 1], fs=1/dt, nperseg=T//2)
    psd = 0.5 * (psd_x + psd_y)  # Average PSD over x and y dimensions
    coeffs = np.polyfit(np.log(freq[1:]), np.log(psd[1:]), 1)
    anomalous_exponent = -coeffs[0] / 2
    diffusion_coefficient = 0.5 * np.sum(psd[1:]) * (freq[1] - freq[0])

    
    return diffusion_coefficient, anomalous_exponent