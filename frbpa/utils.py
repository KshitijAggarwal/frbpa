#!/usr/bin/env python3
import numpy as np
import P4J
import tqdm, logging, itertools
logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def get_phase(bursts, period, ref_mjd=58369.30):
    """
    Calculate phases of the bursts given a reference MJD and period

    :param bursts: Array of burst MJDs
    :param period: Period (in days)
    :param ref_mjd: Reference MJD
    :return:
    """
    return ((bursts - ref_mjd) % period) / period


def mjd_to_f(in_mjd):
    """

    :param in_mjd:
    :return:
    """
    return 1 / in_mjd


def calc_chi_sq(obs_mjds, obs_durations, bursts, period, nbins=8):
    """

    :param obs_mjds: Start MJDs of observations
    :param obs_durations: Durations of all the observations (seconds)
    :param bursts: List or array of burst MJDs
    :param period: Period to evaluate chi-square value
    :param nbins: Number of bins in folded profile
    :return: Chi-square value for that period
    """
    obs_phases = get_phase(obs_mjds, period)
    burst_phases = get_phase(bursts, period)
    temp_phases = np.linspace(0, 1, 100)
    _, phase_bins = np.histogram(temp_phases, bins=nbins)

    obs_phases_binned, _ = np.histogram(obs_phases, bins=phase_bins)
    burst_phases_binned, _ = np.histogram(burst_phases, bins=phase_bins)

    exposure_time = np.zeros(len(obs_phases_binned))
    for i in range(len(phase_bins)):
        if i > 0:
            exposure_time[i - 1] = obs_durations[(obs_phases < phase_bins[i]) &
                                                 (obs_phases > phase_bins[i - 1])].sum()

    p = burst_phases_binned.sum() / exposure_time.sum()
    E = p * exposure_time
    N = burst_phases_binned

    return (((N - E) ** 2) / E).sum()


def get_continuous_frac(folded):
    """
    Calculates the longest circular sequence of zeros 
    in the folded profile, assuming the array wraps around from 
    right to left. Returns as a fraction of the total length. 

    :param folded: Folded profile from riptide.TimeSeries.fold
    :return: Maximum fraction of folded profile without a detectable burst activity
    """
    arr = np.tile(folded, 2)
    group_lengths = [len(tuple(group)) for key, group in itertools.groupby(arr) if key == 0]
    
    if len(group_lengths):
        result = max(group_lengths)
    else:
        result = 0
    
    # if arr is all zeros 
    maxlength = min(result, len(folded))
    return maxlength / len(folded)


def figsize(scale, width_by_height_ratio):
    """
    Create figure size either a full page or a half page figure
    Args:
        scale (float): 0.5 for half page figure, 1 for full page
        width_by_height_ratio (float): ratio of width to height for the figure
    Returns:
        list: list of width and height
    """
    fig_width_pt = 513.17  # 469.755                  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, width_by_height_ratio * fig_height]
    return fig_size


def get_params(scale=0.5, width_by_height_ratio=1):
    """
    Create a dictionary for pretty plotting
    Args:
        scale (float): 0.5 for half page figure, 1 for full page
        width_by_height_ratio (float): ratio of width to height for the figure
    Returns:
        dict: dictionary of parameters
    """
    params = {'backend': 'pdf',
              'axes.labelsize': 10,
              'lines.markersize': 4,
              'font.size': 10,
              'xtick.major.size': 6,
              'xtick.minor.size': 3,
              'ytick.major.size': 6,
              'ytick.minor.size': 3,
              'xtick.major.width': 0.5,
              'ytick.major.width': 0.5,
              'xtick.minor.width': 0.5,
              'ytick.minor.width': 0.5,
              'lines.markeredgewidth': 1,
              'axes.linewidth': 1.2,
              'legend.fontsize': 7,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'savefig.dpi': 200,
              'path.simplify': True,
              'font.family': 'serif',
              'font.serif': 'Times',
              'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{amsbsy}',
                                      r'\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}'],
              'figure.figsize': figsize(scale, width_by_height_ratio)}
    return paramss