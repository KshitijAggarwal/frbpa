#!/usr/bin/env python3
import P4J
import pylab as plt
import numpy as np
import tqdm, logging
from frbpa.utils import mjd_to_f, calc_chi_sq, get_continuous_frac
from riptide import TimeSeries
logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def pr3_search(bursts, obs_mjds, obs_durations, pmin=1.57, pmax=62.8, nbins=8, pres = None):
    """
    Periodicity search using Pearson chi square method used in The CHIME/FRB Collaboration et al 2020

    :param bursts: List or array of burst MJDs
    :param obs_mjds: Start MJDs of observations
    :param obs_durations: Durations of all the observations (seconds)
    :param pmin: Minimum period to search (in units of days)
    :param pmax: Maximum period to search (in units of days)
    :param nbins: Number of bins to use in the folded profile
    :param pres: Period resolution
    :return: red_chi_sqrs, periods
    """

    try:
        assert len(obs_mjds) == len(obs_durations)
    except AssertionError as err:
        logging.exception("Number of start MJDs and durations should be same.")
        raise err

    if not pres:
        pres = (pmax-pmin)/(0.1/(np.max(bursts) - np.min(bursts)))
    periods = np.linspace(pmin, pmax, pres)
    chi_sqrs = []
    for period in tqdm.tqdm(periods):
        chi_sqrs.append(calc_chi_sq(obs_mjds, obs_durations, bursts, period, nbins=nbins))
    
    chi_sqrs = np.array(chi_sqrs)
    nan_mask = (np.isnan(chi_sqrs)) | (np.isinf(chi_sqrs))
    red_chi_sqrs = chi_sqrs[~nan_mask]/(nbins-1)
    periods = periods[~nan_mask]
    
    arg = np.argmax(red_chi_sqrs)
    logging.info(f'Max reduced chi square value is {red_chi_sqrs[arg]} at period of {periods[arg]}')
    return red_chi_sqrs, periods


def riptide_search(bursts, pmin=1*24*60*60, pmax=50*24*60*60,
                   ts_bin_width=0.05, nbins_profile = 40):
    """

    Periodicity search by evaluating the fraction of folded profile without any detectable activity, as used in
    Rajwade et al (2020)

    :param bursts: List or array of burst MJDs
    :param pmin: Minimum period to search (in units of days)
    :param pmax: Maximum period to search (in units of days)
    :param ts_bin_width: Time resolution for binning the burst MJDs
    :param nbins_profile: Number of bins in the folded profile
    :return: continuous_frac, periods
    """
    ts_arr = np.linspace(np.min(bursts), np.max(bursts), 
                     (np.max(bursts)-np.min(bursts))/ts_bin_width)
    hist, edges = np.histogram(bursts, bins=ts_arr)   
    bin_mids = (edges[1:] + edges[:-1])/2
    hist[hist >= 1] = 1
    
    tsamp = ts_bin_width*24*60*60
    ts = TimeSeries(hist*bin_mids, tsamp)
    fs = np.linspace(1/pmax, 1/pmin, (pmax-pmin)/max(bin_mids))
    periods = 1/fs
    
    valid_period_mask = periods/nbins_profile > ts.tsamp
    if valid_period_mask.sum() < len(periods):
        periods = periods[valid_period_mask]
        logging.warning(f'Period/nbins should be greater than tsamp. Not all periods in the given range are valid. '
                     'Selecting the valid periods till {np.max(periods)} for search.')
    
    continuous_frac = []
    for p in tqdm.tqdm(periods):
        folded = ts.fold(p, bins=nbins_profile, subints=1)
        continuous_frac.append(get_continuous_frac(folded))

    arg = np.argmax(continuous_frac)
    logging.info(f'Max continuous fraction without data is {continuous_frac[arg]} '
                 f'at a period of {periods[arg]/(24*60*60)} days')

    return np.array(continuous_frac), periods


def p4j_search(bursts, pmin, pmax, snrs=None, plot=True, save=True, method='QMIEU', mjd_err=0.1, pres=None):
    """

    :param bursts: List or array of burst MJDs
    :param pmin: Minimum period to search (in units of days)
    :param pmax: Maximum period to search (in units of days)
    :param snrs: SNRs of the bursts
    :param plot: To plot the results
    :param save: To save the plotted results
    :param method: Method for periodicity search ['QMIEU', 'LKSL', 'PDMI', 'MHAOV', 'QME', 'QMICS']
    :param mjd_err: Error (in seconds) in burst MJD measurement
    :param pres: Resolution of search periods array (unit days)
    :return: periodogram, periods
    """

    mjds = bursts
    if snrs:
        try:
            assert len(snrs) == len(mjds)
        except AssertionError as err:
            logging.exception("Number of burst MJDs should be equal to number of input SNRs")
            raise err
        mag = snrs
    else:
        mag = np.ones(len(mjds))
    err = np.ones(len(mjds))*(mjd_err/(24*60*60))
    
    my_per = P4J.periodogram(method=method)

    if pres:
        res = pres
    else:
        res = 0.1/(np.max(mjds) - np.min(mjds))
    my_per.set_data(mjds, mag, err)

    # frequency sweep parameters
    my_per.frequency_grid_evaluation(fmin=mjd_to_f(pmax), fmax=mjd_to_f(pmin), fresolution=res) 
    my_per.finetune_best_frequencies(fresolution=res/10, n_local_optima=10)
    freq, per = my_per.get_periodogram()
    fbest, pbest = my_per.get_best_frequencies() # Return best n_local_optima frequencies
    
    if plot:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(freq, per)
        ymin, ymax = ax.get_ylim()
        ax.plot([fbest[0], fbest[0]], [ymin, ymax], linewidth=8, alpha=0.2)
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel('Frequency [1/MJD]')
        ax.set_ylabel('QMI Periodogram')
        plt.title('Periodogram')
        plt.grid()

        ax = fig.add_subplot(1, 2, 2)
        phase = np.mod(mjds, 1.0/fbest[0])*fbest[0]
        idx = np.argsort(phase)
        ax.errorbar(np.concatenate([np.sort(phase), np.sort(phase)+1.0]), 
                    np.concatenate([mag[idx], mag[idx]]),
                    np.concatenate([err[idx], err[idx]]), fmt='.')
        plt.title('Best period')
        ax.set_xlabel('Phase @ %0.5f [1/d], %0.5f [d]' %(fbest[0], 1.0/fbest[0]))
        ax.set_ylabel('Magnitude')
        plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig('P4J_search_output.png', bbox_inches='tight')

    return per, 1/freq