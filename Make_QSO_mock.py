import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from astropy.table import Table

from scipy.integrate import quad, dblquad
from scipy.interpolate import interp2d

from my_utilities import *


w_central = central_wavelength()
tcurves = np.load('../LAEs/npy/tcurves.npy', allow_pickle=True).item()


def add_errors(pm_flx, apply_err=True, survey_name='minijpasAEGIS001', use_5s_lims=True):
    pm_SEDs = np.copy(pm_flx)

    if survey_name == 'jnep':
        err_fit_params_jnep = np.load('../LAEs/npy/err_fit_params_jnep.npy')
    elif survey_name[:8] == 'minijpas':
        err_fit_params_001 = np.load(
            '../LAEs/npy/err_fit_params_minijpas_AEGIS001.npy')
        err_fit_params_002 = np.load(
            '../LAEs/npy/err_fit_params_minijpas_AEGIS002.npy')
        err_fit_params_003 = np.load(
            '../LAEs/npy/err_fit_params_minijpas_AEGIS003.npy')
        err_fit_params_004 = np.load(
            '../LAEs/npy/err_fit_params_minijpas_AEGIS004.npy')
    else:
        raise ValueError('Survey name not known')

    if survey_name[:8] == 'minijpas':
        detec_lim_001 = pd.read_csv('../LAEs/csv/depth3arc5s_minijpas_2241.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_002 = pd.read_csv('../LAEs/csv/depth3arc5s_minijpas_2243.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_003 = pd.read_csv('../LAEs/csv/depth3arc5s_minijpas_2406.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim_004 = pd.read_csv('../LAEs/csv/depth3arc5s_minijpas_2470.csv',
                                    sep=',', header=0, usecols=[1]).to_numpy()
        detec_lim = np.hstack(
            (
                detec_lim_001,
                detec_lim_002,
                detec_lim_003,
                detec_lim_004
            )
        )
        detec_lim.shape
    elif survey_name == 'jnep':
        detec_lim_i = pd.read_csv('../LAEs/csv/depth3arc5s_jnep_2520.csv',
                                  sep=',', header=0, usecols=[1]).to_numpy()

    if survey_name == 'jnep':
        a = err_fit_params_jnep[:, 0].reshape(-1, 1)
        b = err_fit_params_jnep[:, 1].reshape(-1, 1)
        c = err_fit_params_jnep[:, 2].reshape(-1, 1)
        def expfit(x): return a * np.exp(b * x + c)

        w_central = central_wavelength().reshape(-1, 1)

        mags = flux_to_mag(pm_SEDs, w_central)
        mags[~np.isfinite(mags)] = 99.

        # Zero point error
        # zpt_err = Zero_point_error(np.ones(mags.shape[1]) * 2520, 'jnep')

        # mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
        mag_err = expfit(mags)

        if use_5s_lims:
            where_himag = np.where(mags > detec_lim_i)
            mag_err[where_himag] = expfit(
                detec_lim_i)[where_himag[0]].reshape(-1,)
        else:
            where_himag = np.where(mags > 30)
            mag_err[where_himag] = expfit(
                detec_lim_i)[where_himag[0]].reshape(-1,)

        mags[where_himag] = detec_lim_i[where_himag[0]].reshape(-1,)

        pm_SEDs_err = mag_to_flux(
            mags - mag_err, w_central) - mag_to_flux(mags, w_central)
    elif survey_name[:8] == 'minijpas':
        i = int(survey_name[-1])

        detec_lim_i = detec_lim[:, i - 1].reshape(-1, 1)

        if i == 1:
            a = err_fit_params_001[:, 0].reshape(-1, 1)
            b = err_fit_params_001[:, 1].reshape(-1, 1)
            c = err_fit_params_001[:, 2].reshape(-1, 1)
        if i == 2:
            a = err_fit_params_002[:, 0].reshape(-1, 1)
            b = err_fit_params_002[:, 1].reshape(-1, 1)
            c = err_fit_params_002[:, 2].reshape(-1, 1)
        if i == 3:
            a = err_fit_params_003[:, 0].reshape(-1, 1)
            b = err_fit_params_003[:, 1].reshape(-1, 1)
            c = err_fit_params_003[:, 2].reshape(-1, 1)
        if i == 4:
            a = err_fit_params_004[:, 0].reshape(-1, 1)
            b = err_fit_params_004[:, 1].reshape(-1, 1)
            c = err_fit_params_004[:, 2].reshape(-1, 1)

        def expfit(x): return a * np.exp(b * x + c)

        w_central = central_wavelength().reshape(-1, 1)

        mags = flux_to_mag(pm_SEDs, w_central)
        mags[~np.isfinite(mags)] = 99.

        # Zero point error
        # tile_id = tile_id_Arr[i - 1]
        # zpt_err = Zero_point_error(
        #     np.ones(mags.shape[1]) * tile_id, 'minijpas')

        # mag_err = (expfit(mags) ** 2 + zpt_err ** 2) ** 0.5
        mag_err = expfit(mags)
        if use_5s_lims:
            where_himag = np.where(mags > detec_lim_i)
            mag_err[where_himag] = expfit(
                detec_lim_i)[where_himag[0]].reshape(-1,)
        else:
            where_himag = np.where(mags > 30)
            mag_err[where_himag] = expfit(
                detec_lim_i)[where_himag[0]].reshape(-1,)

        mags[where_himag] = detec_lim_i[where_himag[0]].reshape(-1,)

        pm_SEDs_err = mag_to_flux(
            mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    else:
        raise ValueError('Survey name not known')

    # Perturb according to the error
    if apply_err:
        pm_SEDs += np.random.normal(size=mags.shape) * pm_SEDs_err

    # Re-compute the error
    mags = flux_to_mag(pm_SEDs, w_central)
    mags[~np.isfinite(mags)] = 99.

    mag_err = expfit(mags)
    where_himag = np.where(mags > detec_lim_i)

    mag_err[where_himag] = expfit(detec_lim_i)[where_himag[0]].reshape(-1,)

    mags[where_himag] = detec_lim_i[where_himag[0]].reshape(-1,)

    pm_SEDs_err = mag_to_flux(
        mags - mag_err, w_central) - mag_to_flux(mags, w_central)

    return pm_SEDs, pm_SEDs_err


def lya_band_z(plate, mjd, fiber):
    '''
    Computes correct Arr and saves it to a .csv if it dont exist
    '''
    lya_band_res = 1000  # Resolution of the Lya band
    lya_band_hw = 150  # Half width of the Lya band in Angstroms

    correct_dir = 'csv/QSO_mock_correct_files/'
    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/DR16/QSO/'
    try:
        z = np.load(f'{correct_dir}z_arr_dr16.npy')
        lya_band = np.load(f'{correct_dir}lya_band_arr_dr16.npy')
        print('Correct arr loaded')

        return z, lya_band, lya_band_hw
    except:
        print('Computing correct arr...')
        pass

    N_sources = len(fiber)

    # Declare some arrays
    z = np.empty(N_sources)
    lya_band = np.zeros(N_sources)

    # Do the integrated photometry
    # print('Extracting band fluxes from the spectra...')
    print('Making lya_band Arr')
    plate = plate.astype(int)
    mjd = mjd.astype(int)
    fiber = fiber.astype(int)
    for src in range(N_sources):
        if src % 500 == 0:
            print(f'{src} / {N_sources}', end='\r')

        spec_name = fits_dir + \
            f'spec-{plate[src]:04d}-{mjd[src]:05d}-{fiber[src]:04d}.fits'

        spec = Table.read(spec_name, hdu=1, format='fits')
        spzline = Table.read(spec_name, hdu=3, format='fits')

        # Select the source's z as the z from any line not being Lya.
        # Lya z is biased because is taken from the position of the peak of the line,
        # and in general Lya is assymmetrical.
        z_Arr = spzline['LINEZ'][spzline['LINENAME'] != 'Ly_alpha']
        z_Arr = np.atleast_1d(z_Arr[z_Arr != 0.])
        if len(z_Arr) > 0:
            z[src] = z_Arr[-1]
        else:
            z[src] = 0.

        if z[src] < 2:
            continue

        # Synthetic band in Ly-alpha wavelength +- 200 Angstroms
        w_lya = 1215.67
        w_lya_obs = w_lya * (1 + z[src])

        lya_band_tcurves = {
            'tag': ['lyaband'],
            't': [np.ones(lya_band_res)],
            'w': [np.linspace(
                w_lya_obs - lya_band_hw, w_lya_obs + lya_band_hw, lya_band_res
            )]
        }
        # Extract the photometry of Ly-alpha (L_Arr)
        if z[src] > 0:
            lya_band[src] = JPAS_synth_phot(
                spec['FLUX'] * 1e-17, 10 ** spec['LOGLAM'], lya_band_tcurves
            )
        if ~np.isfinite(lya_band[src]):
            lya_band[src] = 0

    os.makedirs(correct_dir, exist_ok=True)
    np.save(f'{correct_dir}z_arr_dr16', z)
    np.save(f'{correct_dir}lya_band_arr_dr16', lya_band)

    return z, lya_band, lya_band_hw

def source_f_cont(mjd, plate, fiber):
    try:
        f_cont = np.load('../LAEs/MyMocks/npy/f_cont_DR16.npy')
        print('f_cont Arr loaded')
        return f_cont
    except:
        pass
    print('Computing f_cont Arr')

    Lya_fts = pd.read_csv('../csv/Lya_fts_DR16.csv')

    N_sources = len(mjd)
    EW = np.empty(N_sources)
    Flambda = np.empty(N_sources)

    for src in range(N_sources):
        if src % 1000 == 0:
            print(f'{src} / {N_sources}', end='\r')

        where = np.where(
            (int(mjd[src]) == Lya_fts['mjd'].to_numpy().flatten())
            & (int(plate[src]) == Lya_fts['plate'].to_numpy().flatten())
            & (int(fiber[src]) == Lya_fts['fiberid'].to_numpy().flatten())
        )

        # Some sources are repeated, so we take the first occurence
        where = where[0][0]

        EW[src] = np.abs(Lya_fts['LyaEW'][where])  # Obs frame EW by now
        Flambda[src] = Lya_fts['LyaF'][where]

    Flambda *= 1e-17  # Correct units & apply correction

    # From the EW formula:
    f_cont = Flambda / EW

    np.save('npy/f_cont_DR16.npy', f_cont)

    return f_cont


def fit_dist_to_profile(L_Arr, f, L_min, L_max, L_step, volume):
    '''
    L_Arr is the array of the distribution to fit
    f is a function of L

    This function trims the L_Arr profile to fit f.
    '''
    N_steps = int((L_max - L_min) / L_step)

    # This is the output mask of sources to include
    out_mask = np.ones_like(L_Arr).astype(bool)

    for j in range(N_steps):
        L_step_min = L_min + j * L_step
        L_step_max = L_min + (j + 1) * L_step
        this_mask = (L_Arr > L_step_min) & (L_Arr <= L_step_max)

        N_src_in = sum(this_mask)
        N_src_out = quad(f, L_step_min, L_step_max)[0] * volume
        N_diff = int(N_src_in - N_src_out)

        if N_diff <= 0:
            continue

        # N_diff of sources have to be removed
        to_remove = np.random.choice(np.where(this_mask)[0], N_diff)
        out_mask[to_remove] = False
    
    return out_mask

def LF_f(L_lya):
    phistar = 3.33e-6
    Lstar = 10 ** 44.65
    alpha = -1.35

    L_lya = 10 ** L_lya
    return schechter(L_lya, phistar, Lstar, alpha) * L_lya * np.log(10)


def main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs, surname=''):
    # Load the SDSS catalog
    filename_pm_DR16 = ('../LAEs/csv/J-SPECTRA_QSO_Superset_DR16_v2.csv')

    pm_SEDs_DR16 = pd.read_csv(
        filename_pm_DR16, usecols=np.arange(1, 64)).to_numpy()[:, 0:60]

    def format_string4(x): return '{:04d}'.format(int(x))
    def format_string5(x): return '{:05d}'.format(int(x))
    convert_dict = {
        122: format_string4,
        123: format_string5,
        124: format_string4
    }
    plate_mjd_fiber = pd.read_csv(
        filename_pm_DR16, sep=',', usecols=[61, 62, 63],
        converters=convert_dict
    ).to_numpy().T

    plate_mjd_fiber = plate_mjd_fiber[np.array([1, 0, 2])]
    plate = plate_mjd_fiber[0].astype(int)
    mjd = plate_mjd_fiber[1].astype(int)
    fiber = plate_mjd_fiber[2].astype(int)

    # z_Arr of SDSS sources

    Lya_fts = pd.read_csv('../LAEs/csv/Lya_fts_DR16_v2.csv')
    z_Arr = Lya_fts['Lya_z'].to_numpy().flatten()
    z_Arr[z_Arr == 0] = -1

    F_line = np.array(Lya_fts['LyaF']) * 1e-17
    F_line_err = np.array(Lya_fts['LyaF_err']) * 1e-17
    EW0 = np.array(Lya_fts['LyaEW']) / (1 + z_Arr)
    EW_err = np.array(Lya_fts['LyaEW_err'])
    dL = cosmo.luminosity_distance(z_Arr).to(u.cm).value
    L = np.log10(F_line * 4*np.pi * dL ** 2)

    F_line_NV = np.array(Lya_fts['NVF']) * 1e-17
    F_line_NV_err = np.array(Lya_fts['NVF_err']) * 1e-17
    EW0_NV = np.array(Lya_fts['NVEW']) / (1 + z_Arr)
    # EW_NV_err = np.array(Lya_fts['NVFEW_err'])
    L_NV = np.log10(F_line_NV * 4*np.pi * dL ** 2)

    # Mask poorly measured EWs & not useful objects for this mock
    mask_neg_EW0 = (EW0 <= 0) | ~np.isfinite(EW0)
    L[mask_neg_EW0] = -1
    z_Arr[mask_neg_EW0] = -1

    model = pd.read_csv('../LAEs/MyMocks/csv/PD2016-QSO_LF.csv')
    counts_model_2D = model.to_numpy()[:-1, 1:-1].astype(float) * 1e-4 * area_obs
    r_yy = np.arange(15.75, 24.25, 0.5)
    z_xx = np.arange(0.5, 6, 1)
    f_counts = interp2d(z_xx, r_yy, counts_model_2D)

    volume = z_volume(z_min, z_max, area_obs)

    Lx = np.logspace(L_min, L_max, 10000)
    log_Lx = np.log10(Lx)

    # Daniele's LF
    phistar1 = 3.33e-6
    Lstar1 = 44.65
    alpha1 = -1.35
    Phi = schechter(Lx, phistar1, 10 ** Lstar1, alpha1) * Lx * np.log(10)

    LF_p_cum_x = np.linspace(44, 47, 1000)
    N_src = int(
        simpson(
            np.interp(LF_p_cum_x, log_Lx, Phi), LF_p_cum_x
        ) * volume
    )

    # Re-bin distribution
    z_xx_new, r_yy_new = (np.linspace(z_min, z_max, 100), np.linspace(15.75, 24.25, 100))
    model_2D_interpolated = f_counts(z_xx_new, r_yy_new).flatten()
    model_2D_interpolated /= np.sum(model_2D_interpolated) # Normalize
    idx_sample = np.random.choice(np.arange(len(model_2D_interpolated)), N_src * 10,
                                  p=model_2D_interpolated)
    idx_sample = np.unravel_index(idx_sample, (len(z_xx_new), len(r_yy_new)))
    out_z_Arr = z_xx_new[idx_sample[1]]
    out_r_Arr = r_yy_new[idx_sample[0]]

    r_flx_Arr = pm_SEDs_DR16[:, -2]
    out_r_flx_Arr = mag_to_flux(out_r_Arr, w_central[-2])

    # Look for the closest source of SDSS in redshift
    out_sdss_idx_list = np.zeros(out_z_Arr.shape).astype(int)
    print('Looking for the sources in SDSS catalog')
    general_mask = (z_Arr > 1) & (L > 43) & (EW0 > 0)
    for src in range(N_src):
        if src % 500 == 0:
            print(f'{src} / {N_src}')
        # Select sources with a redshift closer than 0.06
        closest_z_Arr = np.where((np.abs(z_Arr - out_z_Arr[src]) < 0.06)
                                 & general_mask)[0]
        # If less than 10 objects found with that z_diff, then select the 10 closer
        if len(closest_z_Arr) < 10:
            closest_z_Arr = np.abs(z_Arr - out_z_Arr[src]).argsort()[:10]

        # Select one random source from those
        out_sdss_idx_list[src] = np.random.choice(closest_z_Arr, 1)

    # Correction factor to match iSDSS
    r_corr_factor = out_r_flx_Arr / r_flx_Arr[out_sdss_idx_list]

    # Output PM array
    pm_flx_0 = pm_SEDs_DR16[out_sdss_idx_list] * r_corr_factor.reshape(-1, 1)
    out_EW = EW0[out_sdss_idx_list] * (1 + z_Arr[out_sdss_idx_list]) / (1 + out_z_Arr)
    out_L = L[out_sdss_idx_list] + np.log10(r_corr_factor)
    out_Flambda = F_line[out_sdss_idx_list] * r_corr_factor
    out_Flambda_err = F_line_err[out_sdss_idx_list] * r_corr_factor

    out_EW_NV = EW0_NV[out_sdss_idx_list] * (1 + z_Arr[out_sdss_idx_list]) / (1 + out_z_Arr)
    out_L_NV = L_NV[out_sdss_idx_list] + np.log10(r_corr_factor)
    out_Flambda_NV = F_line_NV[out_sdss_idx_list] * r_corr_factor
    out_Flambda_NV_err = F_line_NV_err[out_sdss_idx_list] * r_corr_factor

    out_mjd = mjd[out_sdss_idx_list]
    out_fiber = fiber[out_sdss_idx_list]
    out_plate = plate[out_sdss_idx_list]

    # Make the pandas df
    print('Saving files')
    cat_name = f'QSO_flat_z{z_min}-{z_max}_r{r_min}-{r_max}_{surname}'
    dirname = f'/home/alberto/almacen/Source_cats/{cat_name}'
    os.makedirs(dirname, exist_ok=True)

    # Withour errors
    filename = f'{dirname}/data0.csv'
    hdr = (
        tcurves['tag']
        + [s + '_e' for s in tcurves['tag']]
        + ['z', 'EW0', 'L_lya', 'F_line', 'F_line_err']
        + ['EW0_NV', 'L_NV', 'F_line_NV', 'F_line_NV_err']
        + ['mjd', 'fiber', 'plate']
    )

    # Mask according to L lims
    L_mask = (out_L >= L_min) & (out_L < L_max)
    print(f'Final N_sources = {sum(L_mask)}')

    df = pd.DataFrame(
        data=np.hstack(
            (
                pm_flx_0[L_mask], pm_flx_0[L_mask] * 0,
                out_z_Arr[L_mask].reshape(-1, 1),
                out_EW[L_mask].reshape(-1, 1),
                out_L[L_mask].reshape(-1, 1),
                out_Flambda[L_mask].reshape(-1, 1),
                out_Flambda_err[L_mask].reshape(-1, 1),
                out_EW_NV[L_mask].reshape(-1, 1),
                out_L_NV[L_mask].reshape(-1, 1),
                out_Flambda_NV[L_mask].reshape(-1, 1),
                out_Flambda_NV_err[L_mask].reshape(-1, 1),
                out_mjd[L_mask].reshape(-1, 1),
                out_fiber[L_mask].reshape(-1, 1),
                out_plate[L_mask].reshape(-1, 1),
            )
        )
    )
    df.to_csv(filename, header=hdr)

if __name__ == '__main__':
    z_min = 1.9
    z_max = 4.2
    r_min = 18
    r_max = 23
    L_min = 40
    L_max = 47
    area_obs = 200

    surname = 'LAES'
    main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs, surname)

    z_min = 1.9
    z_max = 4.2
    r_min = 18
    r_max = 23
    L_min = 44
    L_max = 47
    area_obs = 2000

    surname = 'LAES_hiL'
    main(z_min, z_max, r_min, r_max, L_min, L_max, area_obs, surname)