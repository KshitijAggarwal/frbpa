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
            exposure_time[i-1] = obs_durations[(obs_phases < phase_bins[i]) & 
                                           (obs_phases > phase_bins[i-1])].sum()
            
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


def p4j_block_bootstrap(mjds, mag, err, block_length=10.0, rseed=None):
    '''
    from https://github.com/phuijse/P4J/blob/master/examples/periodogram_demo.ipynb
    '''
    
    np.random.seed(rseed)
    N = len(mjds)
    mjd_boot = np.zeros(shape=(N, )) 
    mag_boot = np.zeros(shape=(N, ))
    err_boot = np.zeros(shape=(N, ))
    k = 0
    last_time = 0.0
    for max_idx in range(2, N):
        if mjds[-1] - mjds[-max_idx] > block_length:
            break
    while k < N:
        idx_start = np.random.randint(N-max_idx-1) 
        for idx_end in range(idx_start+1, N):            
            if mjds[idx_end] - mjds[idx_start] > block_length or k + idx_end - idx_start >= N-1:
                break
        #print("%d %d %d %d" %(idx_start, idx_end, k, k + idx_end - idx_start))
        mjd_boot[k:k+idx_end-idx_start] = mjds[idx_start:idx_end] - mjds[idx_start] + last_time
        mag_boot[k:k+idx_end-idx_start] = mag[idx_start:idx_end]
        err_boot[k:k+idx_end-idx_start] = err[idx_start:idx_end]
        last_time = mjds[idx_end] - mjds[idx_start] + last_time
        k += idx_end - idx_start 
    return mjd_boot, mag_boot, err_boot


def p4j_bootstrap(bursts, pmin, pmax, nsurrogates=200, nlocalmaxima=20):
    '''
    from https://github.com/phuijse/P4J/blob/master/examples/periodogram_demo.ipynb
    '''

    mjds = bursts
    mag = np.ones(len(mjds))
    err = np.ones(len(mjds))*(0.01/(24*60*60)) # 0.1sec error in MJD
    my_per = P4J.periodogram(method='QMIEU')
    res = 0.1/(np.max(mjds) - np.min(mjds))
    my_per.set_data(mjds, mag, err)
    # frequency sweep parameters
    my_per.frequency_grid_evaluation(fmin=mjd_to_f(pmax), fmax=mjd_to_f(pmin), fresolution=res) 
    my_per.finetune_best_frequencies(fresolution=res/10, n_local_optima=10)
    freq, per = my_per.get_periodogram()
    fbest, pbest = my_per.get_best_frequencies() # Return best n_local_optima frequencies

    pbest_bootstrap = np.zeros(shape=(nsurrogates, nlocalmaxima))
    for i in tqdm.tqdm(range(pbest_bootstrap.shape[0])):
        mjd_b, mag_b, err_b = p4j_block_bootstrap(mjds, mag, err, block_length=0.9973)
        my_per.set_data(mjd_b, mag_b, err_b)
        # frequency sweep parameters
        my_per.frequency_grid_evaluation(fmin=mjd_to_f(pmax), fmax=mjd_to_f(pmin), fresolution=res)  
        my_per.finetune_best_frequencies(fresolution=res/10, n_local_optima=pbest_bootstrap.shape[1])
        _, pbest_bootstrap[i, :] = my_per.get_best_frequencies()
        
    from scipy.stats import  genextreme  # Generalized extreme value distribution, it has 3 parameters
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
        ax.plot([rv.ppf(1.-p_val), rv.ppf(1.-p_val)], [ymin, ymax], '--', linewidth=4, alpha=0.5, label=str(p_val))
    ax.set_ylim([ymin, ymax])
    plt.xlabel('Periodogram value'); plt.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(freq, per)
    ymin, ymax = ax.get_ylim()
    ax.plot([fbest[0], fbest[0]], [ymin, ymax], '-', linewidth=8, alpha=0.2)
    # Print confidence bars
    xmin, xmax = ax.get_xlim()
    for p_val in [1e-2, 1e-1]:
        ax.plot([xmin, xmax], [rv.ppf(1.-p_val), rv.ppf(1.-p_val)], '--', linewidth=4, alpha=0.5, label=str(p_val))
    ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Frequency [1/d]'); ax.set_ylabel('Periodogram')
    plt.grid(); plt.legend();
    plt.tight_layout()
    plt.save('P4J_bootstrap.png', bbox_inches='tight')