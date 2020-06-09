#!/usr/bin/env python3
import numpy as np
import json, logging
import pylab as plt
from frbpa.utils import get_phase, get_params
logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def make_obs_phase_plot(data_json, period, ref_mjd=None, nbins=40, save=False, show=False):
    """
    Generates burst phase and observation phase distribution plot for a given period.
    
    :param data_json: json file with data
    :param period: period to use for phase calculation 
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param save: to save the plot
    :param show: to show the plot
    """
    
    with open(data_json, 'r') as f:
        data = json.load(f)

    assert 'obs_duration' in data.keys()
    assert 'bursts' in data.keys()
    assert 'obs_startmjds' in data.keys()
    
    burst_dict = data['bursts']
    obs_duration_dict = data['obs_duration']
    obs_startmjds_dict = data['obs_startmjds']
    
    assert len(obs_duration_dict.keys()) == len(obs_startmjds_dict.keys())
    assert len(obs_duration_dict.keys()) < 20
    assert len(burst_dict.keys()) < 10
    
    new_obs_startmjds_dict = {}
    new_obs_duration_dict = {}
    for k in obs_startmjds_dict.keys():
        start_times = obs_startmjds_dict[k]
        durations = obs_duration_dict[k]
        new_start_times = []
        new_durations = []
        for i, t in enumerate(start_times):
            new_start_times.append(t)
            new_durations.append(durations[i]//2)
            new_start_times.append(t + (durations[i]//2)/(60*60*24))
            new_durations.append(durations[i]//2)
        new_obs_startmjds_dict[k] = new_start_times
        new_obs_duration_dict[k] = new_durations

    obs_duration_dict = new_obs_duration_dict
    obs_startmjds_dict = new_obs_startmjds_dict    

    bursts = []
    for k in burst_dict.keys():
        bursts = bursts + burst_dict[k]

    obs_duration = []
    for k in obs_duration_dict.keys():
        obs_duration = obs_duration + obs_duration_dict[k]

    obs_startmjds = []
    for k in obs_startmjds_dict.keys():
        obs_startmjds = obs_startmjds + obs_startmjds_dict[k]

    assert len(obs_startmjds) == len(obs_duration)
    
    bursts = np.array(bursts)
    obs_duration = np.array(obs_duration)
    obs_startmjds = np.array(obs_startmjds)
    
    obs_start_phases = get_phase(obs_startmjds, period, ref_mjd=ref_mjd)
    hist, bin_edges_obs = np.histogram(obs_start_phases, bins=nbins)
    
    obs_start_phases_dict = {}
    duration_per_phase_dict = {}
    for k in obs_startmjds_dict.keys():
        obs_start_phases_dict[k] = get_phase(np.array(obs_startmjds_dict[k]), 
                                             period)
        durations = np.array(obs_duration_dict[k])
        start_phases = obs_start_phases_dict[k]

        d_hist = []
        for i in range(len(bin_edges_obs)):
            if i>0:
                d_hist.append(durations[(start_phases < bin_edges_obs[i]) & 
                                        (start_phases > bin_edges_obs[i-1])].sum())

        duration_per_phase_dict[k] = np.array(d_hist)/(60*60)
    
    obs_duration = np.array(obs_duration)
    duration_hist = []
    for i in range(len(bin_edges_obs)):
        if i>0:
            duration_hist.append(obs_duration[(obs_start_phases < bin_edges_obs[i]) & 
                                              (obs_start_phases > bin_edges_obs[i-1])].sum())

    duration_hist = np.array(duration_hist)/(60*60)
    bin_mids = (bin_edges_obs[:-1] + bin_edges_obs[1:])/2
    phase_lst = []
    for k in burst_dict.keys():
        phase_lst.append(list(get_phase(np.array(burst_dict[k]), period)))
    
    cm = plt.cm.get_cmap('tab20').colors
    burst_hist_colors = []
    obs_hist_colors = []
    e = 0
    o = 1
    for k in obs_duration_dict.keys():
        if k in burst_dict.keys():
            color = cm[e]
            e += 2
            burst_hist_colors.append(color)
        else:
            color = cm[o]
            o += 2
        obs_hist_colors.append(color)
        
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax1 = ax[0]
    ax1_right = ax1.twinx()
    ax1.hist(phase_lst, bins=bin_edges_obs, stacked=True, density=False, label=burst_dict.keys(), 
             edgecolor='black', linewidth=0.5, color=burst_hist_colors)

    ax1.set_xlabel('Phase')
    ax1.set_ylabel('No. of Bursts')

    ax1_right.scatter(bin_mids, duration_hist, label='Obs duration', c='k', alpha=0.5)
    ax1_right.set_ylabel('Observation Duration (hrs)')

    ax1.legend()
    ax1_right.legend(loc=2)

    ax2 = ax[1]
    cum_ds = np.zeros(nbins)
    for i, k in enumerate(duration_per_phase_dict):
        d = duration_per_phase_dict[k]
        ax2.bar(bin_edges_obs[:-1], d, width=bin_edges_obs[1]-bin_edges_obs[0], align='edge', bottom=cum_ds, 
                alpha=1, label=k, edgecolor='black', linewidth=0.2, color=obs_hist_colors[i])

        cum_ds += d     
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Observation Duration (hrs)')
    ax2.legend()
    plt.tight_layout()
    if save:
        plt.savefig('burst_obs_phase_hist.png', bbox_inches='tight')
        plt.savefig('burst_obs_phase_hist.pdf', bbox_inches='tight')
    if show:
        plt.show()
    

def make_phase_plot(data_json, period, ref_mjd=None, nbins=40, cmap=None, title=None, save=False, show=False):
    """
    Generates burst phase distribution plot at a given period.
    
    :param data_json: json file with data
    :param period: period to use for phase calculation 
    :param ref_mjd: reference MJD to use
    :param nbins: number of bins in the phase histogram
    :param cmap: matplotlib colormap to use 
    :param title: title of the plot 
    :param save: to save the plot
    :param show: to show the plot
    """
    with open(data_json, 'r') as f:
        data = json.load(f)

    burst_dict = data['bursts']
    all_bursts = []
    for k in burst_dict.keys():
        all_bursts += burst_dict[k]
    
    if not ref_mjd: 
        ref_mjd = np.min(all_bursts)
    
    l = []
    for k in burst_dict:
        l.append(get_phase(np.array(burst_dict[k]), period, ref_mjd=ref_mjd))
        
    refphases = np.linspace(0,1,1000)
    _, bin_edges = np.histogram(refphases, bins=nbins)
    
    names = burst_dict.keys()
    num_colors = len(names)
    
    plt.figure(figsize=(10,8))
    
    if not cmap:
        if num_colors < 20:
            cmap = 'tab20'
            colors = plt.get_cmap(cmap).colors[:num_colors]
        else:
            cmap = 'jet'
            cm = plt.get_cmap(cmap)
            colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    
    params = get_params()
    plt.rcParams.update(params)
    
    _ = plt.hist(l, bins=bin_edges, stacked=True, density=False, label=names, edgecolor='black', 
                 linewidth=0.5, color=colors)
    plt.xlabel('Phase')
    plt.ylabel('No. of Bursts')
    if not title:
        title = f'Burst phases of {len(all_bursts)} bursts at a period of {period} days'
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig('burst_phase_histogram.png', bbox_inches='tight')
    if show:
        plt.show()
