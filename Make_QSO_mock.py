import os

import numpy as np
import pandas as pd

from astropy.table import Table

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


def SDSS_z_Arr(mjd, plate, fiber):
    correct_dir = 'csv/QSO_mock_build_files/'
    fits_dir = '/home/alberto/almacen/SDSS_spectra_fits/DR16/QSO/'
    try:
        z_Arr = np.load(f'{correct_dir}z_arr_dr16.npy')
        print('Correct arr loaded')

        return z_Arr
    except:
        print('Computing correct arr...')
        pass

    N_sources = len(mjd)
    z_Arr = np.empty(N_sources)

    for src in range(N_sources):
        if src % 500 == 0:
            print(f'{src} / {N_sources}', end='\r')

        spec_name = fits_dir + \
            f'spec-{plate[src]:04d}-{mjd[src]:05d}-{fiber[src]:04d}.fits'

        spzline = Table.read(spec_name, hdu=3, format='fits')

        # Select the source's z as the z from any line not being Lya.
        # Lya z is biased because is taken from the position of the peak of the line,
        # and in general Lya is assymmetrical.
        this_z_Arr = spzline['LINEZ'][spzline['LINENAME'] != 'Ly_alpha']
        this_z_Arr = np.atleast_1d(this_z_Arr[this_z_Arr != 0.])

        if len(this_z_Arr) > 0:
            z_Arr[src] = this_z_Arr[-1]
        else:
            z_Arr[src] = 0.

    os.makedirs(correct_dir, exist_ok=True)
    np.save(f'{correct_dir}z_arr_dr16.npy', z_Arr)

    return z_Arr


def main(z_min, z_max, r_min, r_max, use_5s_lims=True, surname=''):
    # Load the SDSS catalog
    filename_pm_DR16 = ('../LAEs/csv/J-SPECTRA_QSO_Superset_DR16.csv')

    pm_SEDs_DR16 = pd.read_csv(
        filename_pm_DR16, usecols=np.arange(1, 64)).to_numpy()[:, :60]

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
    z_Arr = SDSS_z_Arr(mjd, plate, fiber)
    r_flx_Arr = pm_SEDs_DR16[:, -2]

    # Output distribution
    # Flat z and i
    PD_z_Arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    PD_counts_Arr = np.array([975471, 2247522, 1282573, 280401, 31368, 4322])
    PD_z_cum_x = np.linspace(z_min, z_max, 1000)
    PD_counts_cum = np.cumsum(np.interp(PD_z_cum_x, PD_z_Arr, PD_counts_Arr))
    PD_counts_cum /= PD_counts_cum.max()

    # According to P-D et al. 2016
    N_src = int(5_222_556 / 1e4 * 400) # In 400 deg^2

    out_z_Arr = np.interp(np.random.rand(N_src),
                         PD_counts_cum, PD_z_cum_x)
    
    # r distribution
    PD_r_Arr = np.arange(15.75, 24, 0.5)
    PD_counts_Arr = np.array([88, 227, 630, 1830, 5404, 15314, 38401, 80354,
                              139413, 209407, 286455, 373434, 475434, 600426,
                              754456, 948401, 1290584])
    PD_r_cum_x = np.linspace(r_min, r_max, 1000)
    PD_counts_cum = np.cumsum(np.interp(PD_r_cum_x, PD_r_Arr, PD_counts_Arr))
    PD_counts_cum /= PD_counts_cum.max()

    out_r_Arr = np.interp(np.random.rand(N_src), PD_counts_cum, PD_r_cum_x)
    out_r_flx_Arr = mag_to_flux(out_r_Arr, w_central[-1])

    # Look for the closest source of SDSS in redshift
    out_sdss_idx_list = np.zeros(out_z_Arr.shape).astype(int)
    print('Looking for the sources in SDSS catalog')
    for src in range(N_src):
        if src % 100 == 0:
            print(f'{src} / {N_src}', end='\r')
        # Select sources with a redshift closer than 0.02
        closest_z_Arr = np.where(np.abs(z_Arr - out_z_Arr[src]) < 0.02)[0]
        # If less than 10 objects found with that z_diff, then select the 10 closer
        if len(closest_z_Arr < 10):
            closest_z_Arr = np.abs(z_Arr - out_z_Arr[src]).argsort()[:10]

        # Select one random source from those
        out_sdss_idx_list[src] = np.random.choice(closest_z_Arr, 1)

    # Correction factor to match iSDSS
    r_corr_factor = out_r_flx_Arr / r_flx_Arr[out_sdss_idx_list]

    # Output PM array
    pm_flx_0 = pm_SEDs_DR16[out_sdss_idx_list] * r_corr_factor.reshape(-1, 1)

    # Compute errors for each field
    # print('Computing errors')
    # pm_flx_AEGIS001, pm_err_AEGIS001 = add_errors(
    #     pm_flx_0.T, survey_name='minijpasAEGIS001', use_5s_lims=use_5s_lims)
    # pm_flx_AEGIS002, pm_err_AEGIS002 = add_errors(
    #     pm_flx_0.T, survey_name='minijpasAEGIS002', use_5s_lims=use_5s_lims)
    # pm_flx_AEGIS003, pm_err_AEGIS003 = add_errors(
    #     pm_flx_0.T, survey_name='minijpasAEGIS003', use_5s_lims=use_5s_lims)
    # pm_flx_AEGIS004, pm_err_AEGIS004 = add_errors(
    #     pm_flx_0.T, survey_name='minijpasAEGIS004', use_5s_lims=use_5s_lims)
    # pm_flx_JNEP, pm_err_JNEP = add_errors(
    #     pm_flx_0.T, survey_name='jnep', use_5s_lims=use_5s_lims)

    # Make the pandas df
    print('Saving files')
    cat_name = f'QSO_flat_z{z_min}-{z_max}_r{r_min}-{r_max}_{surname}'
    dirname = f'/home/alberto/almacen/Source_cats/{cat_name}'
    os.makedirs(dirname, exist_ok=True)

    # Withour errors
    filename = f'{dirname}/QSO_no_err.csv'
    hdr = tcurves['tag'] + ['z']
    df = pd.DataFrame(
        np.hstack([pm_flx_0, out_z_Arr.reshape(-1, 1)]))
    df.to_csv(filename, header=hdr)

    # # With errors
    # hdr = tcurves['tag'] + [s + '_e' for s in tcurves['tag']] + ['z']

    # filename = f'{dirname}/QSO_AEGIS001.csv'
    # df = pd.DataFrame(
    #     np.hstack([pm_flx_AEGIS001.T, pm_err_AEGIS001.T, out_z_Arr.reshape(-1, 1)]))
    # df.to_csv(filename, header=hdr)

    # filename = f'{dirname}/QSO_AEGIS002.csv'
    # df = pd.DataFrame(
    #     np.hstack([pm_flx_AEGIS002.T, pm_err_AEGIS002.T, out_z_Arr.reshape(-1, 1)]))
    # df.to_csv(filename, header=hdr)

    # filename = f'{dirname}/QSO_AEGIS003.csv'
    # df = pd.DataFrame(
    #     np.hstack([pm_flx_AEGIS003.T, pm_err_AEGIS003.T, out_z_Arr.reshape(-1, 1)]))
    # df.to_csv(filename, header=hdr)

    # filename = f'{dirname}/QSO_AEGIS004.csv'
    # df = pd.DataFrame(
    #     np.hstack([pm_flx_AEGIS004.T, pm_err_AEGIS004.T, out_z_Arr.reshape(-1, 1)]))
    # df.to_csv(filename, header=hdr)

    # filename = f'{dirname}/QSO_JNEP.csv'
    # df = pd.DataFrame(
    #     np.hstack([pm_flx_JNEP.T, pm_err_JNEP.T, out_z_Arr.reshape(-1, 1)]))
    # df.to_csv(filename, header=hdr)


if __name__ == '__main__':
    z_min = 1.9
    z_max = 4.2
    r_min = 16
    r_max = 24

    surname = 'LAES'
    main(z_min, z_max, r_min, r_max, True, surname)
