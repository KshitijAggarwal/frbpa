#!/usr/bin/env python3
import numpy as np
import P4J
import tqdm, logging
logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)

def get_phase(bursts, period, ref_mjd = 58369.30):
    return ((bursts - ref_mjd)%period)/period


def mjd_to_f(in_mjd):
    return 1/in_mjd


def calc_chi_sq(obs_mjds, obs_durations, bursts, period, nbins = 8):
    obs_phases = get_phase(obs_mjds, period)
    burst_phases = get_phase(bursts, period)
    temp_phases = np.linspace(0,1,100)
    _ , phase_bins = np.histogram(temp_phases, bins=nbins)
    
    obs_phases_binned, _ = np.histogram(obs_phases, bins=phase_bins)
    burst_phases_binned, _ = np.histogram(burst_phases, bins=phase_bins)
    
    exposure_time = np.zeros(len(obs_phases_binned))
    for i in range(len(phase_bins)):
        if i>0:
            exposure_time[i-1] = obs_durations[(o_phases < phase_bins[i]) & 
                                           (o_phases > phase_bins[i-1])].sum()
            
    p = burst_phases_binned.sum()/exposure_time.sum()
    E = p*exposure_time
    N = burst_phases_binned
    
    return (((N - E)**2)/E).sum()


def get_continuous_frac(folded):
    count = 0
    maxcount = 0
    for i in folded==0:
        if i == True:
            count = count + 1
        else:
            count = 0
        if maxcount < count:
            maxcount = count
    return maxcount/len(folded)