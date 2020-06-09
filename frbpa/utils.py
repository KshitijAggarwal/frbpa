#!/usr/bin/env python3
import numpy as np
import P4J
import tqdm, logging
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

    :param folded: Folded profile from riptide.TimeSeries.fold
    :return: Maximum fraction of folded profile without a detectable burst activity
    """
    count = 0
    maxcount = 0
    for i in folded == 0:
        if i:
            count = count + 1
        else:
            count = 0
        if maxcount < count:
            maxcount = count
    return maxcount / len(folded)


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
    return params


def p4j_block_bootstrap(mjds, mag, err, block_length=10.0, rseed=None):
    """
    from https://github.com/phuijse/P4J/blob/master/examples/periodogram_demo.ipynb
    """

    np.random.seed(rseed)
    N = len(mjds)
    mjd_boot = np.zeros(shape=(N,))
    mag_boot = np.zeros(shape=(N,))
    err_boot = np.zeros(shape=(N,))
    k = 0
    last_time = 0.0
    for max_idx in range(2, N):
        if mjds[-1] - mjds[-max_idx] > block_length:
            break
    while k < N:
        idx_start = np.random.randint(N - max_idx - 1)
        for idx_end in range(idx_start + 1, N):
            if mjds[idx_end] - mjds[idx_start] > block_length or k + idx_end - idx_start >= N - 1:
                break
        mjd_boot[k:k + idx_end - idx_start] = mjds[idx_start:idx_end] - mjds[idx_start] + last_time
        mag_boot[k:k + idx_end - idx_start] = mag[idx_start:idx_end]
        err_boot[k:k + idx_end - idx_start] = err[idx_start:idx_end]
        last_time = mjds[idx_end] - mjds[idx_start] + last_time
        k += idx_end - idx_start
    return mjd_boot, mag_boot, err_boot


def p4j_bootstrap(bursts, pmin, pmax, nsurrogates=200, nlocalmaxima=20):
    """
    from https://github.com/phuijse/P4J/blob/master/examples/periodogram_demo.ipynb
    """

    import pylab as plt

    mjds = bursts
    mag = np.ones(len(mjds))
    err = np.ones(len(mjds)) * (0.01 / (24 * 60 * 60))  # 0.1sec error in MJD
    my_per = P4J.periodogram(method='QMIEU')
    res = 0.1 / (np.max(mjds) - np.min(mjds))
    my_per.set_data(mjds, mag, err)
    # frequency sweep parameters
    my_per.frequency_grid_evaluation(fmin=mjd_to_f(pmax), fmax=mjd_to_f(pmin), fresolution=res)
    my_per.finetune_best_frequencies(fresolution=res / 10, n_local_optima=10)
    freq, per = my_per.get_periodogram()
    fbest, pbest = my_per.get_best_frequencies()  # Return best n_local_optima frequencies

    pbest_bootstrap = np.zeros(shape=(nsurrogates, nlocalmaxima))
    for i in tqdm.tqdm(range(pbest_bootstrap.shape[0])):
        mjd_b, mag_b, err_b = p4j_block_bootstrap(mjds, mag, err, block_length=0.9973)
        my_per.set_data(mjd_b, mag_b, err_b)
        # frequency sweep parameters
        my_per.frequency_grid_evaluation(fmin=mjd_to_f(pmax), fmax=mjd_to_f(pmin), fresolution=res)
        my_per.finetune_best_frequencies(fresolution=res / 10, n_local_optima=pbest_bootstrap.shape[1])
        _, pbest_bootstrap[i, :] = my_per.get_best_frequencies()

    from scipy.stats import genextreme  # Generalized extreme value distribution, it has 3 parameters
    param = genextreme.fit(pbest_bootstrap.ravel())
    rv = genextreme(c=param[0], loc=param[1], scale=param[2])
    x = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(1, 2, 1)
    _ = ax.hist(pbest_bootstrap.ravel(), bins=20, density=True, alpha=0.2, label='Peak\'s histogram')
    ax.plot(x, rv.pdf(x), 'r-', lw=5, alpha=0.6, label='Fitted Gumbel PDF')
    ymin, ymax = ax.get_ylim()
    ax.plot([pbest[0], pbest[0]], [ymin, ymax], '-', linewidth=4, alpha=0.5, label="Max per value")
    for p_val in [1e-2, 1e-1]:
        ax.plot([rv.ppf(1. - p_val), rv.ppf(1. - p_val)], [ymin, ymax], '--', linewidth=4, alpha=0.5, label=str(p_val))
    ax.set_ylim([ymin, ymax])
    plt.xlabel('Periodogram value')
    plt.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(freq, per)
    ymin, ymax = ax.get_ylim()
    ax.plot([fbest[0], fbest[0]], [ymin, ymax], '-', linewidth=8, alpha=0.2)
    # Print confidence bars
    xmin, xmax = ax.get_xlim()
    for p_val in [1e-2, 1e-1]:
        ax.plot([xmin, xmax], [rv.ppf(1. - p_val), rv.ppf(1. - p_val)], '--', linewidth=4, alpha=0.5, label=str(p_val))
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Frequency [1/d]')
    ax.set_ylabel('Periodogram')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.save('P4J_bootstrap.png', bbox_inches='tight')
