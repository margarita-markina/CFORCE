import numpy as np
from numba import jit

# In this code, we convert actual 10m winds to neutral, and then use these to obtain exchange coefficients 
# and then use them to compute turbulent lenghthscales and stability functions iteratively
# and at the end to shift temperature and humidity from 2m to 10m 

# This is based on Hiroyuki's code 
# https://github.com/HiroyukiTsujino/JRA55-do/blob/master/jra55_org_grid_anl/src/shift_wind_real_to_neutral.F90
# I translated fortran to python, and optimised loops over i and j

@jit(nopython=True)
def convert_winds_real2neutral(wdv, sat, qar, slp, sst, ice, atexl, 
                               altu = 10, altt = 2, altq = 2, alt_target = 10, sphmin = 2.0e-5):


    '''
    This function takes original winds, specific humidity (Q) and temperature (T), and information about their original levels, 
    and provides neutral winds, as well as temperature and specific humidity at target level 
    (for raw JRA55, original winds are at 10 m, and T and Q are at 2 m)
    
    This is achieved by following the following iterative procedure (described in detail in LY04):
    
    1 - compute virtual potential temperature, and makes guess on neutral stability and Un(10m) 
      to obtain transfer coefficients and compute initial turbulent lengthscales u*, t*, q*
      
    2 - begin iteration loop to estimate stability parameters (Monin-Obukhov)
    
    3 - shift winds to 10m and neutral stability, and temperature and humidity to wind height
    
    4 - update neutral 10m transfer coefficients, shift them to measurement height and stability
    
    5 - using new transfer coefficients and Theta(Zu), Q(Zu) recompute virtual potential temperature 
      and update the turbulent scales in step 1; then start next interaction loop at step 2.

    Input:
    wdv - original winds (at 10 m)
    sat - original air temperature (at 2 m)
    qar - original specific humidity (at 2 m)
    slp - sea level pressure
    sst - sea surface temperature 
    ice - sea ice concentration
    atexl - land sea mask (1 is sea, 0 is land) 

    altu - altitude of original winds
    altt - altitude of original temperature
    altq - altitude of original specific humidity 
    alt_target - target altutude
    sphmin - minimum specific humidity 

    
    Output:
    wvntrg - neutral winds at target level (10 m)
    tmptrg - temperature at target level (10 m)
    sphtrg - specific humidity at target level (10 m)
    
    '''

    jmx = slp.shape[1]
    imx = slp.shape[0]
    
    wvntrg = np.empty(np.shape(slp))
    tmptrg = np.empty(np.shape(slp))
    sphtrg = np.empty(np.shape(slp))
    
    # Constants
    RO = 1.036  # Water density 
    grav = 981

    # Properties of moist air
    cpa_mks = 1004.67  # specific heat of air [J/Kg/K]
    rhoa_mks = 1.22  # Air density [kg/m^3]
    gasr = 287.04  # gas constant for dry air [J/kg/K]
    mwwater = 18.016  # molecular weight of water vapor
    mwair = 28.966  # molecular weight of air
    tab = 273.15

    ro0mks = RO * 1.0e3
    grav_mks = grav * 1.0e-2

    wvmin = 0.3  # floor on wind speed

    eps_air = mwwater / mwair
    tvq_air = 1.0 / eps_air - 1.0

    # Gill parameters
    agill1, agill2, agill3 = 0.7859, 0.03477, 0.00412
    bgill0, bgill1, bgill2, bgill3 = 0.98, 1.0e-6, 4.5, 0.0006
    cgill1, cgill2 = 2.5008e6, -2.3e3
    dgill1, dgill2 = 1004.6, 0.8735
    igill1 = 0.00422

    karman = 0.4  # von Karman constant

    # Iteration parameters
    n_itts = 5
    inc_ratio = 1.0e-4

    # Loop over grid points
    for j in range(jmx):
        # if j%50==0:
        #     print(str(j) + ' ' + str(jmx))
        for i in range(imx):
            slpres = slp[i, j] * 1.0e-2  # [hPa]
            qatmos = qar[i, j]
            satmos = sat[i, j] - tab

            tssurf = sst[i, j] 
            dtemp = tssurf - satmos

            # Saturation specific humidity at sea surface
            sll = cgill1 + cgill2 * tssurf

            if ice[i, j] == 0.0:
                hl2 = (agill1 + agill2 * tssurf) / (1.0 + agill3 * tssurf)
                hl1 = 10 ** hl2  # [hPa]
                if atexl[i, j] == 1.0:
                    es = hl1 * bgill0 * (1.0 + bgill1 * slpres * (bgill2 + bgill3 * tssurf**2))  # [hPa]
                else:
                    es = hl1 * (1.0 + bgill1 * slpres * (bgill2 + bgill3 * tssurf**2))  # [hPa]
            else:
                hl2 = (agill1 + agill2 * tssurf) / (1.0 + agill3 * tssurf) + igill1 * tssurf
                hl1 = 10 ** hl2  # [hPa]
                es = hl1 * (1.0 + bgill1 * slpres * (bgill2 + bgill3 * tssurf**2))  # [hPa]

            qs = eps_air * es / (slpres - (1.0 - eps_air) * es)
            dqr = qs - qatmos

            
            wv = max(wdv[i, j], wvmin)

            tv = (satmos + tab) * (1.0 + tvq_air * qatmos)
            qatmosu = qatmos

            wv10n = wv  # initial guess for 10 m neutral wind

            if atexl[i, j] == 1.0 and ice[i, j] > 0.0: # neutral ice-ocean transfer coefficients
                cdn10 = 1.63e-3  # L-Y eqn. 21
                cen10 = 1.63e-3  # L-Y eqn. 21
                ctn10 = 1.63e-3  # L-Y eqn. 21
                cdn10_rt = np.sqrt(cdn10)
            else:
                hl1 = (2.7 / wv10n + 0.142 + 0.0764 * wv10n - 3.14807e-10 * (wv10n**6)) / 1.0e3  # LY2009 eqn. 11a
                cdn10 = (0.5 - np.copysign(0.5, wv10n - 33.0)) * hl1 + \
                        (0.5 + np.copysign(0.5, wv10n - 33.0)) * 2.34e-3  # LY2009 eqn. 11b
            
                cdn10_rt = np.sqrt(cdn10)
                cen10 = 34.6 * cdn10_rt / 1.0e3  # L-Y eqn. 6b
                stab = 0.5 + np.copysign(0.5, -dtemp)
                ctn10 = (18.0 * stab + 32.7 * (1.0 - stab)) * cdn10_rt / 1.0e3  # L-Y eqn. 6c
            
            # Initialize coefficients
            cdn = cdn10  # First guess for exchange coefficients at z
            ctn = ctn10
            cen = cen10
            
            cdn_prv = cdn

            for n in range(1, n_itts + 1):  # Monin-Obukhov iteration
                cd_rt = np.sqrt(cdn)  # Square root of drag coefficient
                ustar = cd_rt * wv  # Friction velocity (L-Y eqn. 7a)
                tstar = (ctn / cd_rt) * (-dtemp)  # Temperature scale (L-Y eqn. 7b)
                qstar = (cen / cd_rt) * (-dqr)  # Humidity scale (L-Y eqn. 7c)
            
                # Buoyancy scale (L-Y)
                bstar = grav_mks * (tstar / tv + qstar / (qatmosu + 1.0 / tvq_air))  # Updated version for consistency
            
                # Velocity level
                zetau = karman * bstar * altu / (ustar ** 2)  # L-Y eqn. 8a
                zetau = np.copysign(min(abs(zetau), 10.0), zetau)  # undocumented NCAR
                
                x2 = np.sqrt(abs(1.0 - 16.0 * zetau))  # L-Y eqn. 8b
                x2 = max(x2, 1.0)  # undocumented NCAR
                x = np.sqrt(x2)
                
                if zetau > 0.0:
                    psi_mu = -5.0 * zetau  # L-Y eqn. 8c
                    psi_hu = -5.0 * zetau  # L-Y eqn. 8c
                else:
                    psi_mu = np.log((1.0 + 2.0 * x + x2) * (1.0 + x2) / 8.0) - \
                             2.0 * (np.arctan(x) - np.arctan(1.0))  # L-Y eqn. 8d
                    psi_hu = 2.0 * np.log((1.0 + x2) / 2.0)  # L-Y eqn. 8e


                # Temperature level
                zetat = karman * bstar * altt / (ustar ** 2)  # L-Y eqn. 8a
                zetat = np.copysign(min(abs(zetat), 10.0), zetat)  # undocumented NCAR
                
                x2 = np.sqrt(abs(1.0 - 16.0 * zetat))  # L-Y eqn. 8b
                x2 = max(x2, 1.0)  # undocumented NCAR
                x = np.sqrt(x2)
                
                if zetat > 0.0:
                    psi_mt = -5.0 * zetat  # L-Y eqn. 8c
                    psi_ht = -5.0 * zetat  # L-Y eqn. 8c
                else:
                    psi_mt = np.log((1.0 + 2.0 * x + x2) * (1.0 + x2) / 8.0) - \
                             2.0 * (np.arctan(x) - np.arctan(1.0))  # L-Y eqn. 8d
                    psi_ht = 2.0 * np.log((1.0 + x2) / 2.0)  # L-Y eqn. 8e


                # Water vapor level
                zetaq = karman * bstar * altq / (ustar ** 2)  # L-Y eqn. 8a
                zetaq = np.copysign(min(abs(zetaq), 10.0), zetaq)  # undocumented NCAR
                
                x2 = np.sqrt(abs(1.0 - 16.0 * zetaq))  # L-Y eqn. 8b
                x2 = max(x2, 1.0)  # undocumented NCAR
                x = np.sqrt(x2)
                
                if zetaq > 0.0:
                    psi_mq = -5.0 * zetaq  # L-Y eqn. 8c
                    psi_hq = -5.0 * zetaq  # L-Y eqn. 8c
                else:
                    psi_mq = np.log((1.0 + 2.0 * x + x2) * (1.0 + x2) / 8.0) - \
                             2.0 * (np.arctan(x) - np.arctan(1.0))  # L-Y eqn. 8d
                    psi_hq = 2.0 * np.log((1.0 + x2) / 2.0)  # L-Y eqn. 8e


                # Target level
                zetatrg = karman * bstar * alt_target / (ustar ** 2)  # L-Y eqn. 8a
                zetatrg = np.copysign(min(abs(zetatrg), 10.0), zetatrg)  # undocumented NCAR
                
                x2 = np.sqrt(abs(1.0 - 16.0 * zetatrg))  # L-Y eqn. 8b
                x2 = max(x2, 1.0)  # undocumented NCAR
                x = np.sqrt(x2)
                
                if zetatrg > 0.0:
                    psi_mtrg = -5.0 * zetatrg  # L-Y eqn. 8c
                    psi_htrg = -5.0 * zetatrg  # L-Y eqn. 8c
                else:
                    psi_mtrg = np.log((1.0 + 2.0 * x + x2) * (1.0 + x2) / 8.0) - \
                               2.0 * (np.arctan(x) - np.arctan(1.0))  # L-Y eqn. 8d
                    psi_htrg = 2.0 * np.log((1.0 + x2) / 2.0)  # L-Y eqn. 8e
                    

                # Re-evaluation
                wv10n = wv / (1.0 + cdn10_rt * (np.log(altu / 10.0) - psi_mu) / karman)  # L-Y eqn. 9a
                wv10n = max(wv10n, wvmin)  # 0.3 [m/s] floor on wind
                
                satmosu = satmos - tstar * (np.log(altt / altu) + psi_hu - psi_ht) / karman  # L-Y eqn. 9b
                qatmosu = qatmos - qstar * (np.log(altq / altu) + psi_hu - psi_hq) / karman  # L-Y eqn. 9c
                
                tv = (satmosu + tab) * (1.0 + tvq_air * qatmosu)
                
                if atexl[i, j] == 1.0 and ice[i, j] > 0.0:
                    cdn10 = 1.63e-3  # L-Y eqn. 21
                    cen10 = 1.63e-3  # L-Y eqn. 21
                    ctn10 = 1.63e-3  # L-Y eqn. 21
                    cdn10_rt = np.sqrt(cdn10)
                else:
                    hl1 = (2.7 / wv10n + 0.142 + 0.0764 * wv10n - 3.14807e-10 * (wv10n**6)) / 1.0e3  # LY2009 eqn. 11a
                    cdn10 = (0.5 - np.copysign(0.5, wv10n - 33.0)) * hl1 + \
                            (0.5 + np.copysign(0.5, wv10n - 33.0)) * 2.34e-3  # LY2009 eqn. 11b
                    cdn10_rt = np.sqrt(cdn10)
                    cen10 = 34.6 * cdn10_rt / 1.0e3  # L-Y eqn. 6b
                    stab = 0.5 + np.copysign(0.5, zetau)
                    ctn10 = (18.0 * stab + 32.7 * (1.0 - stab)) * cdn10_rt / 1.0e3  # L-Y eqn. 6c again

                xx = (np.log(altu / 10.0) - psi_mu) / karman
                cdn = cdn10 / (1.0 + cdn10_rt * xx) ** 2  # L-Y eqn. 10a
                
                xx = (np.log(altu / 10.0) - psi_hu) / karman
                ctn = ctn10 / (1.0 + ctn10 * xx / cdn10_rt) * np.sqrt(cdn / cdn10)  # L-Y eqn. 10b
                cen = cen10 / (1.0 + cen10 * xx / cdn10_rt) * np.sqrt(cdn / cdn10)  # L-Y eqn. 10c
                
                dtemp = tssurf - satmosu
                dqr = qs - qatmosu
                
                test = abs(cdn - cdn_prv) / (cdn + 1.0e-8)
                
                if test < inc_ratio:
                    break  # Exit the LOOP_ADJUST
                
                cdn_prv = cdn

            wvntrg[i, j] = wv10n
            tmptrg[i, j] = satmos + tab - tstar * (np.log(altt / alt_target) + psi_htrg - psi_ht) / karman  # L-Y eqn. 9b
            sphtrg[i, j] = qatmos - qstar * (np.log(altq / alt_target) + psi_htrg - psi_hq) / karman  # L-Y eqn. 9c
            sphtrg[i, j] = max(sphtrg[i, j], sphmin)

    # return wvntrg, tmptrg, sphtrg
    return tmptrg, sphtrg