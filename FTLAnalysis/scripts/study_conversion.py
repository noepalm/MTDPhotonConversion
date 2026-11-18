import os
import argparse
import pickle
import numpy as np
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import cycler
import ROOT
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import sys
# if "/eos/home-n/npalmeri/cmssw_versions/plotting/src" not in sys.path:
#     sys.path.append("/eos/home-n/npalmeri/cmssw_versions/plotting/src")
from plot_libraries import *

NanoAODSchema.warn_missing_crossrefs = False

# ----------------- CMS & MPLHEP STYLING -----------------
hep.style.use(hep.style.CMS)

cms_palette = [
    '#5790fc',
    '#f89c20',
    '#e42536',
    '#964a8b',
    '#9c9ca1',
    '#7a21dd',
    '#aa4499',
    '#44aa99',
    '#332288',
    '#117733',
]
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', cms_palette)
mpl.rcParams['patch.linewidth'] = 2

def add_cms_label(ax, rlabel="Phase-2", data=False, label="Preliminary", **kwargs):
    """Standard CMS Simulation Preliminary label with Phase-2 on top-right"""
    hep.cms.label(ax=ax, data=data, label=label, rlabel=rlabel, **kwargs)

def plot_tefficiency_styled(teff, ax, elinewidth=2.2, capthick=2.2, capsize=3, markersize=5, **kwargs):
    """Plot TEfficiency with thick error bars and customizable styling"""
    eff = uproot.from_pyroot(teff.CreateGraph().Clone())
    x_centers = eff.values()[0]
    x_errors_low = eff.members["fEXlow"]
    x_errors_high = eff.members["fEXhigh"]
    y_values = eff.values()[1]
    y_errors_low = eff.members["fEYlow"]
    y_errors_high = eff.members["fEYhigh"]
    x_errors = np.array([x_errors_low, x_errors_high])
    y_errors = np.array([y_errors_low, y_errors_high])
    return ax.errorbar(x_centers, y_values, xerr=x_errors, yerr=y_errors, fmt='o',
                       elinewidth=elinewidth, capthick=capthick, capsize=capsize, markersize=markersize, **kwargs)

# ----------------- UTILITIES -----------------
def ak_seconds(arr, axis=1):
    idxs = ak.ones_like(arr)
    tmp = arr[ak.local_index(arr) == idxs]
    return ak.firsts(tmp, axis=axis)

def angle_to_mpi_pi(phi):
    while ak.sum(phi >= np.pi):
        phi = ak.where(phi >= np.pi, phi - np.pi * 2, phi)
    while ak.sum(phi < -np.pi):
        phi = ak.where(phi < -np.pi, phi + np.pi * 2, phi)
    return phi

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def parabola_ak(x, pars, axis=2):
    x_arr = ak.concatenate([ak.singletons(x)**2, ak.singletons(x), ak.ones_like(ak.singletons(x))], axis=axis)
    par_arr = ak.firsts(pars, axis=axis)
    return ak.sum(x_arr * par_arr, axis=axis)


# ----------------- PROCESSING FUNCTION -----------------
def process_data(args):
    print(">>> Starting data processing from ROOT tree...")
    log_lines = []
    def log_print(mystr):
        print(mystr)
        log_lines.append(mystr)

    # Load data
    if args.use_low_pt:
        fname = "/eos/home-n/npalmeri/MTD/MTD_photonReco/CMSSW_15_1_0_pre2_fabioFix/src/MTDPhotonConversion/FTLAnalysis/test/lowPtSample_fabioFix.root"
    else:
        fname = "/eos/home-n/npalmeri/MTD/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/highPtSample.root"

    evts = NanoEventsFactory.from_root(fname,
                                       treepath="neu_tree",
                                       schemaclass=NanoAODSchema).events()

    conversion_mask_photon = evts.GENConvertedPhoton.convRadius < 116.1
    conversion_mask_photon = conversion_mask_photon * (evts.GENConvertedPhoton.nlegs >= 2)

    ele1 = evts.GENElectron[ak.firsts(evts.GENConvertedPhoton.eleIdxs, axis=2)]
    ele2 = evts.GENElectron[ak_seconds(evts.GENConvertedPhoton.eleIdxs, axis=2)]

    conv_photons = evts.GENConvertedPhoton[conversion_mask_photon]
    ele1 = ele1[conversion_mask_photon]
    ele2 = ele2[conversion_mask_photon]

    mtd_hit = evts.MTDHit[conversion_mask_photon]
    mtd_hit_sim = evts.MTDHitSim[conversion_mask_photon]

    genPV = evts.genPV

    good_cluster_photons = (ak.sum(ele1.typeClus == 0, axis=2) >= 1) * (ak.sum(ele2.typeClus == 0, axis=2) >= 1)
    good_clusters_1 = (ele1[good_cluster_photons].typeClus == 0)
    good_clusters_2 = (ele2[good_cluster_photons].typeClus == 0)

    selected1 = lambda x: ak.firsts(x[good_cluster_photons][good_clusters_1], axis=2)
    selected2 = lambda x: ak.firsts(x[good_cluster_photons][good_clusters_2], axis=2)

    zConv = conv_photons.convZ
    rho = conv_photons.convRadius
    phi = conv_photons.convPhi
    eta = np.arcsinh(zConv / rho)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    n_tracker = ak.sum(rho < 115)
    n_mtd = ak.sum(rho >= 115)
    n_total = ak.sum(ak.num(rho))

    log_print(f"Fraction of conversions in tracker = {n_tracker / n_total * 100:.2f}% ({n_tracker} out of {n_total})")
    log_print(f"Fraction of conversions in MTD = {n_mtd / n_total * 100:.2f}% ({n_mtd} out of {n_total})")
    log_print(f"Fraction of conversions with direct hits for both electrons = {ak.sum(good_cluster_photons) / ak.sum(conversion_mask_photon) * 100 : .3f}% ({ak.sum(good_cluster_photons)} over {ak.sum(conversion_mask_photon)})")

    tmtd_1 = ele1.tClus
    xmtd_1 = ele1.xClus
    ymtd_1 = ele1.yClus
    zmtd_1 = ele1.zClus
    emtd_1 = ele1.energyClus
    tmtd_2 = ele2.tClus
    xmtd_2 = ele2.xClus
    ymtd_2 = ele2.yClus
    zmtd_2 = ele2.zClus
    emtd_2 = ele2.energyClus

    # Barrel cluster selection:
    # Photon in barrel: |eta| < 1.4
    # Cluster in BTL: |zClus| < 260 cm, energyClus > 0, tClus >= 0
    barrel_photon_mask = abs(conv_photons.eta) < 1.4
    ele1_barrel = ele1[barrel_photon_mask]
    ele2_barrel = ele2[barrel_photon_mask]

    m1_barrel = (ele1_barrel.energyClus > 0) & (ele1_barrel.tClus >= 0) & (abs(ele1_barrel.zClus) < 260)
    m2_barrel = (ele2_barrel.energyClus > 0) & (ele2_barrel.tClus >= 0) & (abs(ele2_barrel.zClus) < 260)

    t_barrel_all = np.concatenate([
        ak.to_numpy(ak.ravel(ele1_barrel.tClus[m1_barrel])),
        ak.to_numpy(ak.ravel(ele2_barrel.tClus[m2_barrel]))
    ])
    e_barrel_all = np.concatenate([
        ak.to_numpy(ak.ravel(ele1_barrel.energyClus[m1_barrel])),
        ak.to_numpy(ak.ravel(ele2_barrel.energyClus[m2_barrel]))
    ])

    t_barrel_by_type = [
        np.concatenate([
            ak.to_numpy(ak.ravel(ele1_barrel.tClus[m1_barrel & (ele1_barrel.typeClus == i)])),
            ak.to_numpy(ak.ravel(ele2_barrel.tClus[m2_barrel & (ele2_barrel.typeClus == i)]))
        ])
        for i in range(4)
    ]
    e_barrel_by_type = [
        np.concatenate([
            ak.to_numpy(ak.ravel(ele1_barrel.energyClus[m1_barrel & (ele1_barrel.typeClus == i)])),
            ak.to_numpy(ak.ravel(ele2_barrel.energyClus[m2_barrel & (ele2_barrel.typeClus == i)]))
        ])
        for i in range(4)
    ]

    mtd_r = np.sqrt(xmtd_1**2 + ymtd_1**2)
    mtd_r = mtd_r[(mtd_r > 100) * (mtd_r < 120)]
    log_print(f"Average MTD radius = {ak.mean(mtd_r):.6g} cm (+- 1 std = {ak.std(mtd_r):.3g} cm)")

    is_leading_photon = ak.local_index(conv_photons.pt) == 0
    mask_leading_conv_photon = is_leading_photon * good_cluster_photons
    mask_leading_timeClus_1 = ele1.typeClus[mask_leading_conv_photon] == 0
    time1 = tmtd_1[mask_leading_conv_photon][mask_leading_timeClus_1]

    mask_leading_timeClus_2 = ele2.typeClus[mask_leading_conv_photon] == 0
    time2 = tmtd_2[mask_leading_conv_photon][mask_leading_timeClus_2]

    rho_range = np.linspace(0, 120, 5)

    dPt = (ele1.pt - ele2.pt) / ele1.pt
    dt = selected1(tmtd_1) - selected2(tmtd_2)
    phi1_sel = np.arctan2(selected1(ymtd_1), selected1(xmtd_1))
    phi2_sel = np.arctan2(selected2(ymtd_2), selected2(xmtd_2))
    dphi_sel = angle_to_mpi_pi(phi1_sel - phi2_sel)

    eta1_sel = np.arcsinh(selected1(zmtd_1) / np.sqrt(selected1(xmtd_1)**2 + selected1(ymtd_1)**2))
    eta2_sel = np.arcsinh(selected2(zmtd_2) / np.sqrt(selected2(xmtd_2)**2 + selected2(ymtd_2)**2))
    deta_sel = eta1_sel - eta2_sel

    c = 29.9792458  # cm/ns
    tof1 = np.sqrt(selected1(xmtd_1 - genPV.x)**2 + selected1(ymtd_1 - genPV.y)**2 + selected1(zmtd_1 - genPV.z)**2) / c
    tof2 = np.sqrt(selected2(xmtd_2 - genPV.x)**2 + selected2(ymtd_2 - genPV.y)**2 + selected2(zmtd_2 - genPV.z)**2) / c

    t0_e1 = selected1(tmtd_1) - tof1
    t0_e2 = selected2(tmtd_2) - tof2

    dt1 = t0_e1 - genPV.t
    dt2 = t0_e2 - genPV.t

    dphi1_photon = angle_to_mpi_pi(phi1_sel - conv_photons.phi[good_cluster_photons])
    dphi2_photon = angle_to_mpi_pi(phi2_sel - conv_photons.phi[good_cluster_photons])

    deta1_photon = eta1_sel - conv_photons.eta[good_cluster_photons]
    deta2_photon = eta2_sel - conv_photons.eta[good_cluster_photons]

    tof1_all = np.sqrt((xmtd_1 - genPV.x)**2 + (ymtd_1 - genPV.y)**2 + (zmtd_1 - genPV.z)**2) / c
    tof2_all = np.sqrt((xmtd_2 - genPV.x)**2 + (ymtd_2 - genPV.y)**2 + (zmtd_2 - genPV.z)**2) / c

    t0_e1_all = tmtd_1 - tof1_all
    t0_e2_all = tmtd_2 - tof2_all

    dt1_all = t0_e1_all - genPV.t
    dt2_all = t0_e2_all - genPV.t

    phi1_all = np.arctan2(ymtd_1, xmtd_1)
    phi2_all = np.arctan2(ymtd_2, xmtd_2)

    dphi1_photon_all = angle_to_mpi_pi(phi1_all - conv_photons.phi)
    dphi2_photon_all = angle_to_mpi_pi(phi2_all - conv_photons.phi)

    eta1_all = np.arcsinh(zmtd_1 / np.sqrt(xmtd_1**2 + ymtd_1**2))
    eta2_all = np.arcsinh(zmtd_2 / np.sqrt(xmtd_2**2 + ymtd_2**2))

    deta1_photon_all = eta1_all - conv_photons.eta
    deta2_photon_all = eta2_all - conv_photons.eta

    # Profile & Parabola Fits
    results = {}
    profile_scatter_data = []
    dynamic_x_ranges = {0: 0.5, 30: 0.4, 60: 0.3, 90: 0.07}

    for idx, (rho_min, rho_max) in enumerate(zip(rho_range[:-1], rho_range[1:])):
        x_lim = dynamic_x_ranges.get(int(round(rho_min)), [0.5, 0.4, 0.3, 0.07][idx] if idx < 4 else 0.5)
        phi_bins = np.linspace(-x_lim, x_lim, 20)
        xs, xs_err, ys, ys_err, nentries = [], [], [], [], []

        mask = (rho >= rho_min) * (rho < rho_max)
        for phi_min, phi_max in zip(phi_bins[:-1], phi_bins[1:]):
            mask1 = mask * (ele1.typeClus == 0) * (dphi1_photon_all >= phi_min) * (dphi1_photon_all < phi_max)
            mask2 = mask * (ele2.typeClus == 0) * (dphi2_photon_all >= phi_min) * (dphi2_photon_all < phi_max)

            dt1_bin = tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t
            dt2_bin = tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t

            arr = ak.concatenate([ak.ravel(dt1_bin), ak.ravel(dt2_bin)])
            arr_sel = arr[(arr > -0.1) * (arr < 0.2)]
            nentries.append(len(arr_sel))
            ys.append(np.mean(arr_sel))
            ys_err.append(np.std(arr_sel))
            xs.append((phi_min + phi_max)/2)
            xs_err.append((phi_max - phi_min)/2)

        mask1 = mask * (ele1.typeClus == 0)
        mask2 = mask * (ele2.typeClus == 0)

        dt1_fit = tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t
        dt2_fit = tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t

        dphi1_fit = angle_to_mpi_pi(np.arctan2(ymtd_1[mask1], xmtd_1[mask1]) - conv_photons.phi)
        dphi2_fit = angle_to_mpi_pi(np.arctan2(ymtd_2[mask2], xmtd_2[mask2]) - conv_photons.phi)

        arrx = ak.to_numpy(ak.concatenate([ak.ravel(dphi1_fit), ak.ravel(dphi2_fit)]))
        arry = ak.to_numpy(ak.concatenate([ak.ravel(dt1_fit), ak.ravel(dt2_fit)]))

        nan_mask = np.isnan(ys) + np.isnan(ys_err) + (np.array(nentries) < 3)
        xs_c = np.array(xs)[~nan_mask]
        xs_err_c = np.array(xs_err)[~nan_mask]
        ys_c = np.array(ys)[~nan_mask]
        ys_err_c = np.array(ys_err)[~nan_mask]

        popt, pcov = curve_fit(parabola, xs_c, ys_c, sigma=ys_err_c, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        results[(rho_min, rho_max)] = (popt, perr)

        profile_scatter_data.append({
            "rho_min": rho_min,
            "rho_max": rho_max,
            "x_lim": x_lim,
            "arrx": arrx,
            "arry": arry,
            "xs": xs_c,
            "xs_err": xs_err_c,
            "ys": ys_c,
            "ys_err": ys_err_c,
            "popt": popt,
            "perr": perr,
        })

    # Fixed 10-bin profile for plot 13 (mtd_deltaTOF_vs_dphi) across common [-0.5, 0.5] range
    profile_scatter_data_10bins = []
    for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:]):
        phi_bins_10 = np.linspace(-0.5, 0.5, 10)
        xs_10, xs_err_10, ys_10, ys_err_10 = [], [], [], []
        mask = (rho >= rho_min) * (rho < rho_max)
        for phi_min, phi_max in zip(phi_bins_10[:-1], phi_bins_10[1:]):
            mask1 = mask * (ele1.typeClus == 0) * (dphi1_photon_all >= phi_min) * (dphi1_photon_all < phi_max)
            mask2 = mask * (ele2.typeClus == 0) * (dphi2_photon_all >= phi_min) * (dphi2_photon_all < phi_max)
            dt1_bin = tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t
            dt2_bin = tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t
            arr = ak.concatenate([ak.ravel(dt1_bin), ak.ravel(dt2_bin)])
            arr_sel = arr[(arr > -0.1) * (arr < 0.2)]
            ys_10.append(np.mean(arr_sel))
            ys_err_10.append(np.std(arr_sel))
            xs_10.append((phi_min + phi_max)/2)
            xs_err_10.append((phi_max - phi_min)/2)
        profile_scatter_data_10bins.append({
            "rho_min": rho_min,
            "rho_max": rho_max,
            "xs": np.array(xs_10),
            "xs_err": np.array(xs_err_10),
            "ys": np.array(ys_10),
            "ys_err": np.array(ys_err_10),
        })

    # Resolution test
    mask1 = (ele1.typeClus == 0)
    mask2 = (ele2.typeClus == 0)
    photon_mask = ak.sum(ele1.typeClus == 0, axis=2) + ak.sum(ele2.typeClus == 0, axis=2) > 0

    t0_ele1 = ak.firsts(tmtd_1[mask1], axis=2) - np.sqrt((ak.firsts(xmtd_1[mask1], axis=2) - genPV.x)**2 + (ak.firsts(ymtd_1[mask1], axis=2) - genPV.y)**2 + (ak.firsts(zmtd_1[mask1], axis=2) - genPV.z)**2) / c
    t0_ele2 = ak.firsts(tmtd_2[mask2], axis=2) - np.sqrt((ak.firsts(xmtd_2[mask2], axis=2) - genPV.x)**2 + (ak.firsts(ymtd_2[mask2], axis=2) - genPV.y)**2 + (ak.firsts(zmtd_2[mask2], axis=2) - genPV.z)**2) / c

    dphi1_res = angle_to_mpi_pi(np.arctan2(ak.firsts(ymtd_1[mask1], axis=2), ak.firsts(xmtd_1[mask1], axis=2)) - conv_photons.phi)
    dphi2_res = angle_to_mpi_pi(np.arctan2(ak.firsts(ymtd_2[mask2], axis=2), ak.firsts(xmtd_2[mask2], axis=2)) - conv_photons.phi)

    photon_mask_1 = ak.sum(mask1, axis=2) > 0
    photon_mask_2 = ak.sum(mask2, axis=2) > 0

    rho_bins_1 = ak.unflatten(np.digitize(ak.flatten(rho[photon_mask_1]), rho_range), ak.num(rho[photon_mask_1])) - 1
    rho_bins_2 = ak.unflatten(np.digitize(ak.flatten(rho[photon_mask_2]), rho_range), ak.num(rho[photon_mask_2])) - 1

    fit_results = np.array([results[(rmin, rmax)] for rmin, rmax in zip(rho_range[:-1], rho_range[1:])])

    coeffs_1 = ak.unflatten(fit_results[ak.flatten(rho_bins_1)], ak.num(rho[photon_mask_1]))
    coeffs_2 = ak.unflatten(fit_results[ak.flatten(rho_bins_2)], ak.num(rho[photon_mask_2]))

    t0_ele1_clean = t0_ele1[~ak.is_none(t0_ele1, axis=1)]
    t0_ele2_clean = t0_ele2[~ak.is_none(t0_ele2, axis=1)]

    tof_corr_1 = parabola_ak(dphi1_res[~ak.is_none(dphi1_res, axis=1)], coeffs_1, axis=2)
    tof_corr_2 = parabola_ak(dphi2_res[~ak.is_none(dphi2_res, axis=1)], coeffs_2, axis=2)

    t0_res_1 = t0_ele1_clean - tof_corr_1 - conv_photons[photon_mask_1].t0 * 1e9
    t0_res_2 = t0_ele2_clean - tof_corr_2 - conv_photons[photon_mask_2].t0 * 1e9
    t0_res_precorr_1 = t0_ele1_clean - conv_photons[photon_mask_1].t0 * 1e9
    t0_res_precorr_2 = t0_ele2_clean - conv_photons[photon_mask_2].t0 * 1e9

    resolution_bins = []
    for i in range(len(rho_range)):
        rmin = 0 if i == 0 else rho_range[i-1]
        rmax = rho_range[-1] if i == 0 else rho_range[i]
        rho_mask_1 = (rho[photon_mask_1] > rmin) * (rho[photon_mask_1] < rmax)
        rho_mask_2 = (rho[photon_mask_2] > rmin) * (rho[photon_mask_2] < rmax)
        corr = ak.to_numpy(ak.concatenate([ak.ravel(t0_res_1[rho_mask_1]), ak.ravel(t0_res_2[rho_mask_2])]))
        precorr = ak.to_numpy(ak.concatenate([ak.ravel(t0_res_precorr_1[rho_mask_1]), ak.ravel(t0_res_precorr_2[rho_mask_2])]))
        resolution_bins.append({
            "rmin": rmin,
            "rmax": rmax,
            "is_all": (i == 0),
            "corr": corr,
            "precorr": precorr
        })

    # Debugging Section
    debugging_data = {}
    rho_mins = [80, 114]
    rho_maxs = [110, 118]

    for rho_min, rho_max in zip(rho_mins, rho_maxs):
        log_print(f"LOOKING AT EVENTS WITH {rho_min} < Rconv < {rho_max}\n")
        bad_events_1 = (rho > rho_min) * (rho < rho_max)
        bad_events_2 = (rho > rho_min) * (rho < rho_max)

        mask1_d = (ele1.typeClus == 0) * (dphi1_photon_all >= -0.05) * (dphi1_photon_all < 0.05)
        mask2_d = (ele2.typeClus == 0) * (dphi2_photon_all >= -0.05) * (dphi2_photon_all < 0.05)

        arr1_d = ak.ravel(tmtd_1[mask1_d * bad_events_1])
        arr2_d = ak.ravel(tmtd_2[mask2_d * bad_events_2])
        arr_tmtd_d = ak.to_numpy(ak.concatenate([ak.ravel(arr1_d), ak.ravel(arr2_d)]))

        tof1_full = np.sqrt((xmtd_1 - genPV.x)**2 + (ymtd_1 - genPV.y)**2 + (zmtd_1 - genPV.z)**2) / c
        tof2_full = np.sqrt((xmtd_2 - genPV.x)**2 + (ymtd_2 - genPV.y)**2 + (zmtd_2 - genPV.z)**2) / c

        suspect_events_1 = tof1_full < 4
        suspect_events_2 = tof2_full < 4

        tof1_d = np.sqrt((xmtd_1[mask1_d * bad_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1] - genPV.z)**2) / c
        tof2_d = np.sqrt((xmtd_2[mask2_d * bad_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2] - genPV.z)**2) / c
        arr_tof_d = ak.to_numpy(ak.concatenate([ak.ravel(tof1_d), ak.ravel(tof2_d)]))

        arr_genPV_t_d = ak.to_numpy(genPV.t[ak.sum(bad_events_1, axis=1) > 0])

        arr1_res = tmtd_1[mask1_d * bad_events_1] - np.sqrt((xmtd_1[mask1_d * bad_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1] - genPV.z)**2) / c - genPV.t
        arr2_res = tmtd_2[mask2_d * bad_events_2] - np.sqrt((xmtd_2[mask2_d * bad_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2] - genPV.z)**2) / c - genPV.t

        awful_events_1 = arr1_res > 0.1
        awful_events_2 = arr2_res > 0.1

        arr_res = ak.to_numpy(ak.concatenate([ak.ravel(arr1_res), ak.ravel(arr2_res)]))

        arr1_sus = tmtd_1[mask1_d * bad_events_1 * suspect_events_1] - np.sqrt((xmtd_1[mask1_d * bad_events_1 * suspect_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1 * suspect_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1 * suspect_events_1] - genPV.z)**2) / c - genPV.t
        arr2_sus = tmtd_2[mask2_d * bad_events_2 * suspect_events_2] - np.sqrt((xmtd_2[mask2_d * bad_events_2 * suspect_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2 * suspect_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2 * suspect_events_2] - genPV.z)**2) / c - genPV.t
        arr_suspect = ak.to_numpy(ak.concatenate([ak.ravel(arr1_sus), ak.ravel(arr2_sus)]))

        arr1_awful = ak.ravel(tmtd_1[mask1_d * bad_events_1][awful_events_1])
        arr2_awful = ak.ravel(tmtd_2[mask2_d * bad_events_2][awful_events_2])
        arr_tmtd_awful = ak.to_numpy(ak.concatenate([arr1_awful, arr2_awful]))

        tof1_awful = np.sqrt((xmtd_1[mask1_d * bad_events_1][awful_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1][awful_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1][awful_events_1] - genPV.z)**2) / c
        tof2_awful = np.sqrt((xmtd_2[mask2_d * bad_events_2][awful_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2][awful_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2][awful_events_2] - genPV.z)**2) / c
        arr_tof_awful = ak.to_numpy(ak.concatenate([ak.ravel(tof1_awful), ak.ravel(tof2_awful)]))

        arr1_genPV_awful = genPV.t[(ak.sum(bad_events_1, axis=1) > 0) * (ak.sum(ak.sum(awful_events_1, axis=2), axis=1) > 0)]
        arr2_genPV_awful = genPV.t[(ak.sum(bad_events_2, axis=1) > 0) * (ak.sum(ak.sum(awful_events_2, axis=2), axis=1) > 0)]
        arr_genPV_awful = ak.to_numpy(ak.concatenate([arr1_genPV_awful, arr2_genPV_awful]))

        awful_mask = (ak.sum(ak.sum(awful_events_1, axis=2), axis=1) + ak.sum(ak.sum(awful_events_2, axis=2), axis=1)) > 0
        awful_events_list = evts.event[awful_mask]
        log_print("Awful events:")
        log_print(str(awful_events_list.to_list()))

        for evt in awful_events_list[:5]:
            log_print(f"Event {evt}:")
            log_print(f"GenPV time: {genPV.t[evts.event == evt].to_list()}")
            log_print(f"MTD cluster types: ele1 : {ele1.typeClus[evts.event == evt].to_list()}, ele2 : {ele2.typeClus[evts.event == evt].to_list()}")
            all_info_1 = ak.zip({"t": tmtd_1[evts.event == evt], "x": xmtd_1[evts.event == evt], "y": ymtd_1[evts.event == evt], "z": zmtd_1[evts.event == evt]})
            all_info_2 = ak.zip({"t": tmtd_2[evts.event == evt], "x": xmtd_2[evts.event == evt], "y": ymtd_2[evts.event == evt], "z": zmtd_2[evts.event == evt]})
            log_print("\tEle1:")
            for event in all_info_1:
                for j, photon in enumerate(event):
                    log_print(f"\tPhoton {j}:")
                    for idx, cluster in enumerate(photon):
                        log_print(f"\t\tCluster {idx}: {cluster}")
            log_print("\tEle2:")
            for event in all_info_2:
                for j, photon in enumerate(event):
                    log_print(f"\tPhoton {j}:")
                    for idx, cluster in enumerate(photon):
                        log_print(f"\t\tCluster {idx}: {cluster}")
            arr1_evt = tmtd_1[mask1_d * bad_events_1] - np.sqrt((xmtd_1[mask1_d * bad_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1] - genPV.z)**2) / c - genPV.t
            arr2_evt = tmtd_2[mask2_d * bad_events_2] - np.sqrt((xmtd_2[mask2_d * bad_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2] - genPV.z)**2) / c - genPV.t
            log_print(f"Uncorrected resolution: ele1 : {arr1_evt[evts.event == evt].to_list()}, ele2 : {arr2_evt[evts.event == evt].to_list()}")
            log_print(f"TOF: ele1 = {tof1_full[evts.event == evt].to_list()}, ele2 = {tof2_full[evts.event == evt].to_list()}")

        arr_x_diff = ak.to_numpy(ak.concatenate([ak.ravel(xmtd_1[mask1_d * bad_events_1] - genPV.x), ak.ravel(xmtd_2[mask2_d * bad_events_2] - genPV.x)]))
        arr_x_sus = ak.to_numpy(ak.concatenate([ak.ravel(xmtd_1[mask1_d * bad_events_1][awful_events_1]), ak.ravel(xmtd_2[mask2_d * bad_events_2][awful_events_2])]))

        arr_y_diff = ak.to_numpy(ak.concatenate([ak.ravel(ymtd_1[mask1_d * bad_events_1] - genPV.y), ak.ravel(ymtd_2[mask2_d * bad_events_2] - genPV.y)]))
        arr_y_sus = ak.to_numpy(ak.concatenate([ak.ravel(ymtd_1[mask1_d * bad_events_1][awful_events_1]), ak.ravel(ymtd_2[mask2_d * bad_events_2][awful_events_2])]))

        arr_z_diff = ak.to_numpy(ak.concatenate([ak.ravel(zmtd_1[mask1_d * bad_events_1] - genPV.z), ak.ravel(zmtd_2[mask2_d * bad_events_2] - genPV.z)]))
        arr_z_sus = ak.to_numpy(ak.concatenate([ak.ravel(zmtd_1[mask1_d * bad_events_1][awful_events_1]), ak.ravel(zmtd_2[mask2_d * bad_events_2][awful_events_2])]))

        x1_m = ak.ravel(xmtd_1[mask1_d * bad_events_1])
        y1_m = ak.ravel(ymtd_1[mask1_d * bad_events_1])
        z1_m = ak.ravel(zmtd_1[mask1_d * bad_events_1])
        eta1_m = np.arcsinh(z1_m / np.sqrt(x1_m**2 + y1_m**2))
        x2_m = ak.ravel(xmtd_2[mask2_d * bad_events_2])
        y2_m = ak.ravel(ymtd_2[mask2_d * bad_events_2])
        z2_m = ak.ravel(zmtd_2[mask2_d * bad_events_2])
        eta2_m = np.arcsinh(z2_m / np.sqrt(x2_m**2 + y2_m**2))
        arr_eta_m = ak.to_numpy(ak.concatenate([eta1_m, eta2_m]))

        x1_sus_m = ak.ravel(xmtd_1[mask1_d * bad_events_1][awful_events_1])
        y1_sus_m = ak.ravel(ymtd_1[mask1_d * bad_events_1][awful_events_1])
        z1_sus_m = ak.ravel(zmtd_1[mask1_d * bad_events_1][awful_events_1])
        eta1_sus_m = np.arcsinh(z1_sus_m / np.sqrt(x1_sus_m**2 + y1_sus_m**2))
        x2_sus_m = ak.ravel(xmtd_2[mask2_d * bad_events_2][awful_events_2])
        y2_sus_m = ak.ravel(ymtd_2[mask2_d * bad_events_2][awful_events_2])
        z2_sus_m = ak.ravel(zmtd_2[mask2_d * bad_events_2][awful_events_2])
        eta2_sus_m = np.arcsinh(z2_sus_m / np.sqrt(x2_sus_m**2 + y2_sus_m**2))
        arr_eta_sus = ak.to_numpy(ak.concatenate([eta1_sus_m, eta2_sus_m]))

        arr_rho_m = ak.to_numpy(ak.concatenate([ak.ravel(np.sqrt(x1_m**2 + y1_m**2)), ak.ravel(np.sqrt(x2_m**2 + y2_m**2))]))
        arr_rho_sus = ak.to_numpy(ak.concatenate([ak.ravel(np.sqrt(x1_sus_m**2 + y1_sus_m**2)), ak.ravel(np.sqrt(x2_sus_m**2 + y2_sus_m**2))]))

        arr_phi_m = ak.to_numpy(ak.concatenate([ak.ravel(np.arctan2(y1_m, x1_m)), ak.ravel(np.arctan2(y2_m, x2_m))]))
        arr_phi_sus = ak.to_numpy(ak.concatenate([ak.ravel(np.arctan2(y1_sus_m, x1_sus_m)), ak.ravel(np.arctan2(y2_sus_m, x2_sus_m))]))

        conv_x = conv_photons.convRadius * np.cos(conv_photons.convPhi)
        conv_y = conv_photons.convRadius * np.sin(conv_photons.convPhi)
        conv_z = conv_photons.convZ

        dist_1 = np.sqrt((xmtd_1[mask1_d * bad_events_1] - conv_x)**2 + (ymtd_1[mask1_d * bad_events_1] - conv_y)**2 + (zmtd_1[mask1_d * bad_events_1] - conv_z)**2)
        dist_2 = np.sqrt((xmtd_2[mask2_d * bad_events_2] - conv_x)**2 + (ymtd_2[mask2_d * bad_events_2] - conv_y)**2 + (zmtd_2[mask2_d * bad_events_2] - conv_z)**2)
        arr_dist = ak.to_numpy(ak.concatenate([ak.ravel(dist_1), ak.ravel(dist_2)]))

        dist_1_awful = np.sqrt((xmtd_1[mask1_d * bad_events_1][awful_events_1] - conv_x)**2 + (ymtd_1[mask1_d * bad_events_1][awful_events_1] - conv_y)**2 + (zmtd_1[mask1_d * bad_events_1][awful_events_1] - conv_z)**2)
        dist_2_awful = np.sqrt((xmtd_2[mask2_d * bad_events_2][awful_events_2] - conv_x)**2 + (ymtd_2[mask2_d * bad_events_2][awful_events_2] - conv_y)**2 + (zmtd_2[mask2_d * bad_events_2][awful_events_2] - conv_z)**2)
        arr_dist_awful = ak.to_numpy(ak.concatenate([ak.ravel(dist_1_awful), ak.ravel(dist_2_awful)]))

        far_events_1 = dist_1 > 3
        far_events_2 = dist_2 > 3

        arr1_far = tmtd_1[mask1_d * bad_events_1][far_events_1] - np.sqrt((xmtd_1[mask1_d * bad_events_1][far_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1][far_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1][far_events_1] - genPV.z)**2) / c - genPV.t
        arr2_far = tmtd_2[mask2_d * bad_events_2][far_events_2] - np.sqrt((xmtd_2[mask2_d * bad_events_2][far_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2][far_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2][far_events_2] - genPV.z)**2) / c - genPV.t
        arr_far_res = ak.to_numpy(ak.concatenate([ak.ravel(arr1_far), ak.ravel(arr2_far)]))

        arr_tmtd_far = ak.to_numpy(ak.concatenate([ak.ravel(tmtd_1[mask1_d * bad_events_1][far_events_1]), ak.ravel(tmtd_2[mask2_d * bad_events_2][far_events_2])]))

        tof1_far = np.sqrt((xmtd_1[mask1_d * bad_events_1][far_events_1] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1][far_events_1] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1][far_events_1] - genPV.z)**2) / c
        tof2_far = np.sqrt((xmtd_2[mask2_d * bad_events_2][far_events_2] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2][far_events_2] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2][far_events_2] - genPV.z)**2) / c
        arr_tof_far = ak.to_numpy(ak.concatenate([ak.ravel(tof1_far), ak.ravel(tof2_far)]))

        deltaPhi1_m = angle_to_mpi_pi(np.arctan2(ymtd_1[mask1_d * bad_events_1], xmtd_1[mask1_d * bad_events_1]) - np.arctan2(conv_y, conv_x))
        deltaPhi2_m = angle_to_mpi_pi(np.arctan2(ymtd_2[mask2_d * bad_events_2], xmtd_2[mask2_d * bad_events_2]) - np.arctan2(conv_y, conv_x))
        arr_dphi_conv = ak.to_numpy(ak.concatenate([ak.ravel(deltaPhi1_m), ak.ravel(deltaPhi2_m)]))

        deltaPhi1_awful = angle_to_mpi_pi(np.arctan2(ymtd_1[mask1_d * bad_events_1][awful_events_1], xmtd_1[mask1_d * bad_events_1][awful_events_1]) - np.arctan2(conv_y, conv_x))
        deltaPhi2_awful = angle_to_mpi_pi(np.arctan2(ymtd_2[mask2_d * bad_events_2][awful_events_2], xmtd_2[mask2_d * bad_events_2][awful_events_2]) - np.arctan2(conv_y, conv_x))
        arr_dphi_conv_awful = ak.to_numpy(ak.concatenate([ak.ravel(deltaPhi1_awful), ak.ravel(deltaPhi2_awful)]))

        deltaR1_m = np.sqrt((xmtd_1[mask1_d * bad_events_1] - conv_x)**2 + (ymtd_1[mask1_d * bad_events_1] - conv_y)**2)
        deltaR2_m = np.sqrt((xmtd_2[mask2_d * bad_events_2] - conv_x)**2 + (ymtd_2[mask2_d * bad_events_2] - conv_y)**2)
        arr_dr_conv = ak.to_numpy(ak.concatenate([ak.ravel(deltaR1_m), ak.ravel(deltaR2_m)]))

        deltaR1_awful = np.sqrt((xmtd_1[mask1_d * bad_events_1][awful_events_1] - conv_x)**2 + (ymtd_1[mask1_d * bad_events_1][awful_events_1] - conv_y)**2)
        deltaR2_awful = np.sqrt((xmtd_2[mask2_d * bad_events_2][awful_events_2] - conv_x)**2 + (ymtd_2[mask2_d * bad_events_2][awful_events_2] - conv_y)**2)
        arr_dr_conv_awful = ak.to_numpy(ak.concatenate([ak.ravel(deltaR1_awful), ak.ravel(deltaR2_awful)]))

        deltaZ1_m = np.abs(zmtd_1[mask1_d * bad_events_1] - conv_z)
        deltaZ2_m = np.abs(zmtd_2[mask2_d * bad_events_2] - conv_z)
        arr_dz_conv = ak.to_numpy(ak.concatenate([ak.ravel(deltaZ1_m), ak.ravel(deltaZ2_m)]))

        deltaZ1_awful = np.abs(zmtd_1[mask1_d * bad_events_1][awful_events_1] - conv_z)
        deltaZ2_awful = np.abs(zmtd_2[mask2_d * bad_events_2][awful_events_2] - conv_z)
        arr_dz_conv_awful = ak.to_numpy(ak.concatenate([ak.ravel(deltaZ1_awful), ak.ravel(deltaZ2_awful)]))

        deltaEta1_m = np.arcsinh(zmtd_1[mask1_d * bad_events_1] / np.sqrt(xmtd_1[mask1_d * bad_events_1]**2 + ymtd_1[mask1_d * bad_events_1]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
        deltaEta2_m = np.arcsinh(zmtd_2[mask2_d * bad_events_2] / np.sqrt(xmtd_2[mask2_d * bad_events_2]**2 + ymtd_2[mask2_d * bad_events_2]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
        arr_deta_conv = ak.to_numpy(ak.concatenate([ak.ravel(deltaEta1_m), ak.ravel(deltaEta2_m)]))

        deltaEta1_awful = np.arcsinh(zmtd_1[mask1_d * bad_events_1][awful_events_1] / np.sqrt(xmtd_1[mask1_d * bad_events_1][awful_events_1]**2 + ymtd_1[mask1_d * bad_events_1][awful_events_1]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
        deltaEta2_awful = np.arcsinh(zmtd_2[mask2_d * bad_events_2][awful_events_2] / np.sqrt(xmtd_2[mask2_d * bad_events_2][awful_events_2]**2 + ymtd_2[mask2_d * bad_events_2][awful_events_2]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
        arr_deta_conv_awful = ak.to_numpy(ak.concatenate([ak.ravel(deltaEta1_awful), ak.ravel(deltaEta2_awful)]))

        awful_events_geo = []
        for i_evt, evt in enumerate(awful_events_list[:4]):
            arr1_loc = tmtd_1[mask1_d * bad_events_1 * (evts.event == evt)] - np.sqrt((xmtd_1[mask1_d * bad_events_1 * (evts.event == evt)] - genPV.x)**2 + (ymtd_1[mask1_d * bad_events_1 * (evts.event == evt)] - genPV.y)**2 + (zmtd_1[mask1_d * bad_events_1 * (evts.event == evt)] - genPV.z)**2) / c - genPV.t
            arr2_loc = tmtd_2[mask2_d * bad_events_2 * (evts.event == evt)] - np.sqrt((xmtd_2[mask2_d * bad_events_2 * (evts.event == evt)] - genPV.x)**2 + (ymtd_2[mask2_d * bad_events_2 * (evts.event == evt)] - genPV.y)**2 + (zmtd_2[mask2_d * bad_events_2 * (evts.event == evt)] - genPV.z)**2) / c - genPV.t
            awful_1_loc = arr1_loc > 0.1
            awful_2_loc = arr2_loc > 0.1

            bad_photons_1 = (evts.event == evt) * (ak.sum(awful_1_loc, axis=2) > 0)
            bad_photons_2 = (evts.event == evt) * (ak.sum(awful_2_loc, axis=2) > 0)

            c_x_1 = conv_photons.convRadius[bad_photons_1] * np.cos(conv_photons.convPhi[bad_photons_1])
            c_y_1 = conv_photons.convRadius[bad_photons_1] * np.sin(conv_photons.convPhi[bad_photons_1])
            c_z_1 = conv_photons.convZ[bad_photons_1]
            c_r_1 = np.sqrt(c_x_1**2 + c_y_1**2)

            c_x_2 = conv_photons.convRadius[bad_photons_2] * np.cos(conv_photons.convPhi[bad_photons_2])
            c_y_2 = conv_photons.convRadius[bad_photons_2] * np.sin(conv_photons.convPhi[bad_photons_2])
            c_z_2 = conv_photons.convZ[bad_photons_2]
            c_r_2 = np.sqrt(c_x_2**2 + c_y_2**2)

            conv_x_plot = ak.to_numpy(ak.ravel(ak.concatenate([c_x_1, c_x_2])))
            conv_y_plot = ak.to_numpy(ak.ravel(ak.concatenate([c_y_1, c_y_2])))
            conv_z_plot = ak.to_numpy(ak.ravel(ak.concatenate([c_z_1, c_z_2])))
            conv_r_plot = ak.to_numpy(ak.ravel(ak.concatenate([c_r_1, c_r_2])))

            m_x_1 = xmtd_1[mask1_d * bad_events_1 * (evts.event == evt)][awful_1_loc]
            m_x_2 = xmtd_2[mask2_d * bad_events_2 * (evts.event == evt)][awful_2_loc]
            m_y_1 = ymtd_1[mask1_d * bad_events_1 * (evts.event == evt)][awful_1_loc]
            m_y_2 = ymtd_2[mask2_d * bad_events_2 * (evts.event == evt)][awful_2_loc]
            m_z_1 = zmtd_1[mask1_d * bad_events_1 * (evts.event == evt)][awful_1_loc]
            m_z_2 = zmtd_2[mask2_d * bad_events_2 * (evts.event == evt)][awful_2_loc]

            mtd_x_plot = ak.to_numpy(ak.ravel(ak.concatenate([m_x_1, m_x_2])))
            mtd_y_plot = ak.to_numpy(ak.ravel(ak.concatenate([m_y_1, m_y_2])))
            mtd_z_plot = ak.to_numpy(ak.ravel(ak.concatenate([m_z_1, m_z_2])))
            mtd_r_plot = np.sqrt(mtd_x_plot**2 + mtd_y_plot**2)

            awful_events_geo.append({
                "evt": evt,
                "conv_x": conv_x_plot,
                "conv_y": conv_y_plot,
                "conv_z": conv_z_plot,
                "conv_r": conv_r_plot,
                "mtd_x": mtd_x_plot,
                "mtd_y": mtd_y_plot,
                "mtd_z": mtd_z_plot,
                "mtd_r": mtd_r_plot
            })

        debugging_data[(rho_min, rho_max)] = {
            "arr_tmtd": arr_tmtd_d,
            "arr_tof": arr_tof_d,
            "arr_genPV_t": arr_genPV_t_d,
            "arr_res": arr_res,
            "arr_suspect": arr_suspect,
            "arr_tmtd_awful": arr_tmtd_awful,
            "arr_tof_awful": arr_tof_awful,
            "arr_genPV_awful": arr_genPV_awful,
            "arr_x_diff": arr_x_diff,
            "arr_x_sus": arr_x_sus,
            "arr_y_diff": arr_y_diff,
            "arr_y_sus": arr_y_sus,
            "arr_z_diff": arr_z_diff,
            "arr_z_sus": arr_z_sus,
            "arr_eta_m": arr_eta_m,
            "arr_eta_sus": arr_eta_sus,
            "arr_rho_m": arr_rho_m,
            "arr_rho_sus": arr_rho_sus,
            "arr_phi_m": arr_phi_m,
            "arr_phi_sus": arr_phi_sus,
            "arr_dist": arr_dist,
            "arr_dist_awful": arr_dist_awful,
            "arr_far_res": arr_far_res,
            "arr_tmtd_far": arr_tmtd_far,
            "arr_tof_far": arr_tof_far,
            "arr_dphi_conv": arr_dphi_conv,
            "arr_dphi_conv_awful": arr_dphi_conv_awful,
            "arr_dr_conv": arr_dr_conv,
            "arr_dr_conv_awful": arr_dr_conv_awful,
            "arr_dz_conv": arr_dz_conv,
            "arr_dz_conv_awful": arr_dz_conv_awful,
            "arr_deta_conv": arr_deta_conv,
            "arr_deta_conv_awful": arr_deta_conv_awful,
            "awful_events_geo": awful_events_geo,
        }

    # Conversion Rate TEfficiency
    photon_pt_sort = ak.argsort(evts.GENConvertedPhoton.pt, ascending=False)
    pt_all = evts.GENConvertedPhoton.pt[photon_pt_sort][:,:2]
    eta_all = evts.GENConvertedPhoton.eta[photon_pt_sort][:,:2]
    nlegs_all = evts.GENConvertedPhoton.nlegs[photon_pt_sort][:,:2]
    convradius_all = evts.GENConvertedPhoton.convRadius[photon_pt_sort][:,:2]

    are_conversions_old = (nlegs_all >= 2) * (convradius_all < 118) * (pt_all > 1)
    are_conversions = (nlegs_all >= 2) * (convradius_all < 116.1)
    are_conversions_MTD = (nlegs_all >= 2) * (convradius_all < 116.1) * (convradius_all > 115.4)

    pt_max = 12 if args.use_low_pt else 100
    conversion_rate_pt = ROOT.TEfficiency("conversion_rate_pt", "Conversion rate vs pt", 20, 0, pt_max)
    conversion_rate_eta = ROOT.TEfficiency("conversion_rate_eta", "Conversion rate vs eta", 20, 0, 1.5)
    conversion_rate_pt_MTD = ROOT.TEfficiency("conversion_rate_pt_MTD", "Conversion rate in MTD vs pt ", 20, 0, pt_max)
    conversion_rate_eta_MTD = ROOT.TEfficiency("conversion_rate_eta_MTD", "Conversion rate in MTD vs eta ", 20, 0, 1.5)
    conversion_rate_pt_old = ROOT.TEfficiency("conversion_rate_pt_old", "Conversion rate vs pt", 20, 0, pt_max)
    conversion_rate_eta_old = ROOT.TEfficiency("conversion_rate_eta_old", "Conversion rate vs eta", 20, 0, 1.5)

    for pt_val, eta_val, is_c, is_c_old, is_c_mtd in zip(ak.flatten(pt_all), ak.flatten(eta_all), ak.flatten(are_conversions), ak.flatten(are_conversions_old), ak.flatten(are_conversions_MTD)):
        conversion_rate_pt.Fill(1 * is_c, pt_val)
        conversion_rate_eta.Fill(1 * is_c, abs(eta_val))
        conversion_rate_pt_old.Fill(1 * is_c_old, pt_val)
        conversion_rate_eta_old.Fill(1 * is_c_old, abs(eta_val))
        conversion_rate_pt_MTD.Fill(1 * is_c_mtd, pt_val)
        conversion_rate_eta_MTD.Fill(1 * is_c_mtd, abs(eta_val))

    match_mask = mtd_hit.drMatch < 1e5
    matched_reco_hit = ak.sum(match_mask) / ak.sum(ak.num(mtd_hit))
    log_print(f"For converted photons, rate of MTD hit matching: {matched_reco_hit * 100:.2f}%")

    drMatch_all = ak.to_numpy(ak.ravel(mtd_hit[match_mask].drMatch))
    drMatch_trk = ak.to_numpy(ak.ravel(mtd_hit[match_mask * rho < 114].drMatch))
    drMatch_mtd = ak.to_numpy(ak.ravel(mtd_hit[match_mask * rho > 114].drMatch))

    res1 = ak.to_numpy(ak.ravel(mtd_hit[match_mask * good_cluster_photons].time - selected1(tmtd_1)[match_mask[good_cluster_photons]]))
    res2 = ak.to_numpy(ak.ravel(mtd_hit[match_mask * good_cluster_photons].time - selected2(tmtd_2)[match_mask[good_cluster_photons]]))

    # Package into dictionary
    data = {
        "log_lines": log_lines,
        "kinematics_photon": {
            "pt": ak.to_numpy(ak.flatten(conv_photons.pt, axis=None)),
            "eta": ak.to_numpy(ak.flatten(conv_photons.eta, axis=None)),
            "phi": ak.to_numpy(ak.flatten(conv_photons.phi, axis=None)),
        },
        "kinematics_electron": {
            "ele1_pt": ak.to_numpy(ak.flatten(ele1.pt, axis=None)),
            "ele1_eta": ak.to_numpy(ak.flatten(ele1.eta, axis=None)),
            "ele1_phi": ak.to_numpy(ak.flatten(ele1.phi, axis=None)),
            "ele2_pt": ak.to_numpy(ak.flatten(ele2.pt, axis=None)),
            "ele2_eta": ak.to_numpy(ak.flatten(ele2.eta, axis=None)),
            "ele2_phi": ak.to_numpy(ak.flatten(ele2.phi, axis=None)),
        },
        "vertex_pos": {
            "x": ak.to_numpy(ak.flatten(x, axis=None)),
            "y": ak.to_numpy(ak.flatten(y, axis=None)),
            "rho": ak.to_numpy(ak.flatten(rho, axis=None)),
            "z": ak.to_numpy(ak.flatten(zConv, axis=None)),
        },
        "mtd_cluster_positions": {
            "x1_all": ak.to_numpy(ak.flatten(xmtd_1, axis=None)),
            "y1_all": ak.to_numpy(ak.flatten(ymtd_1, axis=None)),
            "x2_all": ak.to_numpy(ak.flatten(xmtd_2, axis=None)),
            "y2_all": ak.to_numpy(ak.flatten(ymtd_2, axis=None)),
            "x1_dir": ak.to_numpy(ak.flatten(xmtd_1[ele1.typeClus == 0], axis=None)),
            "y1_dir": ak.to_numpy(ak.flatten(ymtd_1[ele1.typeClus == 0], axis=None)),
            "x2_dir": ak.to_numpy(ak.flatten(xmtd_2[ele2.typeClus == 0], axis=None)),
            "y2_dir": ak.to_numpy(ak.flatten(ymtd_2[ele2.typeClus == 0], axis=None)),
            "mtd_r": ak.to_numpy(ak.ravel(mtd_r)),
        },
        "genPV": {
            "x": ak.to_numpy(genPV.x),
            "y": ak.to_numpy(genPV.y),
            "z": ak.to_numpy(genPV.z),
            "t": ak.to_numpy(genPV.t),
        },
        "mtd_cluster_info": {
            "t_all": ak.to_numpy(ak.concatenate([ak.ravel(tmtd_1), ak.ravel(tmtd_2)])),
            "t_by_type": [
                ak.to_numpy(ak.concatenate([ak.ravel(tmtd_1[ele1.typeClus == i]), ak.ravel(tmtd_2[ele2.typeClus == i])]))
                for i in range(4)
            ],
            "t_2direct_all": ak.to_numpy(ak.concatenate([ak.ravel(selected1(tmtd_1)), ak.ravel(selected2(tmtd_2))])),
            "t_2direct_leading": ak.to_numpy(ak.concatenate([ak.ravel(time1), ak.ravel(time2)])),
            "t_direct_vs_rho": [
                (rho_min, rho_max, ak.to_numpy(ak.ravel(tmtd_1[(rho >= rho_min) * (rho < rho_max)][ele1[(rho >= rho_min) * (rho < rho_max)].typeClus == 0])))
                for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:])
            ],
            "t_direct_2dir_vs_rho": [
                (rho_min, rho_max, ak.to_numpy(ak.ravel(ak.firsts(tmtd_1[(rho >= rho_min) * (rho < rho_max) * good_cluster_photons][ele1[(rho >= rho_min) * (rho < rho_max) * good_cluster_photons].typeClus == 0], axis=2))))
                for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:])
            ],
            "clus_type_all": ak.to_numpy(ak.concatenate([ak.ravel(ele2.typeClus), ak.ravel(ele1.typeClus)])),
            "clus_type_mtd": ak.to_numpy(ak.concatenate([ak.ravel(ele2.typeClus[rho > 115]), ak.ravel(ele1.typeClus[rho > 115])])),
            "xClus_all": ak.to_numpy(ak.concatenate([ak.ravel(ele1.xClus), ak.ravel(ele2.xClus)])),
            "yClus_all": ak.to_numpy(ak.concatenate([ak.ravel(ele1.yClus), ak.ravel(ele2.yClus)])),
            "zClus_all": ak.to_numpy(ak.concatenate([ak.ravel(ele1.zClus), ak.ravel(ele2.zClus)])),
            "phi_clus": ak.to_numpy(ak.ravel(ak.concatenate([
                np.arctan2(ymtd_1[abs(ymtd_1) < 130], xmtd_1[abs(xmtd_1) < 130]),
                np.arctan2(ymtd_2[abs(ymtd_2) < 130], xmtd_2[abs(xmtd_2) < 130])
            ]))),
            "rho_clus": ak.to_numpy(ak.ravel(ak.concatenate([
                np.sqrt(xmtd_1[abs(xmtd_1) < 130]**2 + ymtd_1[abs(ymtd_1) < 130]**2),
                np.sqrt(xmtd_2[abs(xmtd_2) < 130]**2 + ymtd_2[abs(ymtd_2) < 130]**2)
            ]))),
            "barrel_t_all": t_barrel_all,
            "barrel_t_by_type": t_barrel_by_type,
            "barrel_energy_all": e_barrel_all,
            "barrel_energy_by_type": e_barrel_by_type,
        },
        "mtd_cluster_barrel": {
            "t_all": t_barrel_all,
            "t_by_type": t_barrel_by_type,
            "energy_all": e_barrel_all,
            "energy_by_type": e_barrel_by_type,
        },
        "conversion_kinematics": {
            "dPt": ak.to_numpy(ak.flatten(dPt, axis=None)),
            "dt_all": ak.to_numpy(ak.flatten(dt, axis=None)),
            "dt_mtd": ak.to_numpy(ak.flatten(dt[rho[good_cluster_photons] >= 115], axis=None)),
            "dphi_all": ak.to_numpy(ak.flatten(dphi_sel, axis=None)),
            "dphi_mtd": ak.to_numpy(ak.flatten(dphi_sel[rho[good_cluster_photons] >= 115], axis=None)),
            "rho_good": ak.to_numpy(ak.ravel(rho[good_cluster_photons])),
            "dphi_good": ak.to_numpy(ak.ravel(dphi_sel)),
            "deta_all": ak.to_numpy(ak.flatten(deta_sel, axis=None)),
            "deta_mtd": ak.to_numpy(ak.flatten(deta_sel[rho[good_cluster_photons] >= 115], axis=None)),
            "deta_good": ak.to_numpy(ak.ravel(deta_sel)),
            "dt1_all": ak.to_numpy(ak.flatten(dt1, axis=None)),
            "dt2_all": ak.to_numpy(ak.flatten(dt2, axis=None)),
            "dt1_mtd": ak.to_numpy(ak.flatten(dt1[rho[good_cluster_photons] >= 115], axis=None)),
            "dt2_mtd": ak.to_numpy(ak.flatten(dt2[rho[good_cluster_photons] >= 115], axis=None)),
            "dphi1_photon": ak.to_numpy(ak.flatten(dphi1_photon, axis=None)),
            "dphi2_photon": ak.to_numpy(ak.flatten(dphi2_photon, axis=None)),
            "dphi1_photon_mtd": ak.to_numpy(ak.flatten(dphi1_photon[rho[good_cluster_photons] >= 115], axis=None)),
            "dphi2_photon_mtd": ak.to_numpy(ak.flatten(dphi2_photon[rho[good_cluster_photons] >= 115], axis=None)),
            "pt_good": ak.to_numpy(ak.flatten(conv_photons.pt[good_cluster_photons], axis=None)),
            "deta1_photon": ak.to_numpy(ak.flatten(deta1_photon, axis=None)),
            "deta2_photon": ak.to_numpy(ak.flatten(deta2_photon, axis=None)),
            "rho_all": ak.to_numpy(ak.ravel(rho)),
        },
        "deltaTOF_vs_dphi_scatter": {
            "direct": (
                ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon), ak.ravel(dphi2_photon)])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)]))
            ),
            "direct_mtd": (
                ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[rho[good_cluster_photons] >= 115]), ak.ravel(dphi2_photon[rho[good_cluster_photons] >= 115])])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] >= 115]), ak.ravel(dt2[rho[good_cluster_photons] >= 115])]))
            ),
            "direct_trk": (
                ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[rho[good_cluster_photons] < 90]), ak.ravel(dphi2_photon[rho[good_cluster_photons] < 90])])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] < 90]), ak.ravel(dt2[rho[good_cluster_photons] < 90])]))
            ),
            "all": (
                ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon_all), ak.ravel(dphi2_photon_all)])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1_all), ak.ravel(dt2_all)]))
            ),
            "all_mtd": (
                ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon_all[rho >= 115]), ak.ravel(dphi2_photon_all[rho >= 115])])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1_all[rho >= 115]), ak.ravel(dt2_all[rho >= 115])]))
            ),
            "type_trk": {
                i: (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon_all[(ele1.typeClus == i) * (rho < 90)]), ak.ravel(dphi2_photon_all[(ele2.typeClus == i) * (rho < 90)])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1_all[(ele1.typeClus == i) * (rho < 90)]), ak.ravel(dt2_all[(ele2.typeClus == i) * (rho < 90)])]))
                )
                for i in range(2)
            },
            "eta_direct": {
                "inner": (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[abs(eta[good_cluster_photons]) < 0.6]), ak.ravel(dphi2_photon[abs(eta[good_cluster_photons]) < 0.6])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1[abs(eta[good_cluster_photons]) < 0.6]), ak.ravel(dt2[abs(eta[good_cluster_photons]) < 0.6])]))
                ),
                "outer": (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[abs(eta[good_cluster_photons]) >= 0.6]), ak.ravel(dphi2_photon[abs(eta[good_cluster_photons]) >= 0.6])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1[abs(eta[good_cluster_photons]) >= 0.6]), ak.ravel(dt2[abs(eta[good_cluster_photons]) >= 0.6])]))
                ),
            },
            "eta_mtd": {
                "inner": (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) < 0.6)]), ak.ravel(dphi2_photon[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) < 0.6)])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) < 0.6)]), ak.ravel(dt2[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) < 0.6)])]))
                ),
                "outer": (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) >= 0.6)]), ak.ravel(dphi2_photon[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) >= 0.6)])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) >= 0.6)]), ak.ravel(dt2[(rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) >= 0.6)])]))
                ),
            },
            "eta_trk": {
                "inner": (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) < 0.6)]), ak.ravel(dphi2_photon[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) < 0.6)])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) < 0.6)]), ak.ravel(dt2[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) < 0.6)])]))
                ),
                "outer": (
                    ak.to_numpy(ak.concatenate([ak.ravel(dphi1_photon[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) >= 0.6)]), ak.ravel(dphi2_photon[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) >= 0.6)])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) >= 0.6)]), ak.ravel(dt2[(rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) >= 0.6)])]))
                ),
            }
        },
        "deltaTOF_vs_deta_scatter": {
            "direct": (
                ak.to_numpy(ak.concatenate([ak.ravel(deta1_photon), ak.ravel(deta2_photon)])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)]))
            ),
            "direct_mtd": (
                ak.to_numpy(ak.concatenate([ak.ravel(deta1_photon[rho[good_cluster_photons] >= 115]), ak.ravel(deta2_photon[rho[good_cluster_photons] >= 115])])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] >= 115]), ak.ravel(dt2[rho[good_cluster_photons] >= 115])]))
            ),
            "direct_trk": (
                ak.to_numpy(ak.concatenate([ak.ravel(deta1_photon[rho[good_cluster_photons] < 90]), ak.ravel(deta2_photon[rho[good_cluster_photons] < 90])])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] < 90]), ak.ravel(dt2[rho[good_cluster_photons] < 90])]))
            ),
            "all": (
                ak.to_numpy(ak.concatenate([ak.ravel(deta1_photon_all), ak.ravel(deta2_photon_all)])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1_all), ak.ravel(dt2_all)]))
            ),
            "all_mtd": (
                ak.to_numpy(ak.concatenate([ak.ravel(deta1_photon_all[rho >= 115]), ak.ravel(deta2_photon_all[rho >= 115])])),
                ak.to_numpy(ak.concatenate([ak.ravel(dt1_all[rho >= 115]), ak.ravel(dt2_all[rho >= 115])]))
            ),
            "type_trk": {
                i: (
                    ak.to_numpy(ak.concatenate([ak.ravel(deta1_photon_all[(ele1.typeClus == i) * (rho < 90)]), ak.ravel(deta2_photon_all[(ele2.typeClus == i) * (rho < 90)])])),
                    ak.to_numpy(ak.concatenate([ak.ravel(dt1_all[(ele1.typeClus == i) * (rho < 90)]), ak.ravel(dt2_all[(ele2.typeClus == i) * (rho < 90)])]))
                )
                for i in range(2)
            }
        },
        "profile_scatter_data": profile_scatter_data,
        "profile_scatter_data_10bins": profile_scatter_data_10bins,
        "deltaTOF_1d_by_rho": [
            (
                rho_min,
                rho_max,
                ak.to_numpy(ak.concatenate([
                    ak.ravel(tmtd_1[(rho >= rho_min) * (rho < rho_max) * (ele1.typeClus == 0)] - np.sqrt((xmtd_1[(rho >= rho_min) * (rho < rho_max) * (ele1.typeClus == 0)] - genPV.x)**2 + (ymtd_1[(rho >= rho_min) * (rho < rho_max) * (ele1.typeClus == 0)] - genPV.y)**2 + (zmtd_1[(rho >= rho_min) * (rho < rho_max) * (ele1.typeClus == 0)] - genPV.z)**2) / c - genPV.t),
                    ak.ravel(tmtd_2[(rho >= rho_min) * (rho < rho_max) * (ele2.typeClus == 0)] - np.sqrt((xmtd_2[(rho >= rho_min) * (rho < rho_max) * (ele2.typeClus == 0)] - genPV.x)**2 + (ymtd_2[(rho >= rho_min) * (rho < rho_max) * (ele2.typeClus == 0)] - genPV.y)**2 + (zmtd_2[(rho >= rho_min) * (rho < rho_max) * (ele2.typeClus == 0)] - genPV.z)**2) / c - genPV.t)
                ]))
            )
            for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:])
        ],
        "resolution_bins": resolution_bins,
        "all_tof_corr": ak.to_numpy(ak.concatenate([ak.ravel(tof_corr_1), ak.ravel(tof_corr_2)])),
        "mtd_tof_corr": ak.to_numpy(ak.concatenate([ak.ravel(tof_corr_1[rho[photon_mask_1] > 112]), ak.ravel(tof_corr_2[rho[photon_mask_2] > 112])])),
        "debugging": debugging_data,
        "conversion_rate": {
            "pt": conversion_rate_pt,
            "eta": conversion_rate_eta,
            "pt_mtd": conversion_rate_pt_MTD,
            "eta_mtd": conversion_rate_eta_MTD,
            "pt_old": conversion_rate_pt_old,
            "eta_old": conversion_rate_eta_old,
        },
        "hit_matching": {
            "drMatch_all": drMatch_all,
            "drMatch_trk": drMatch_trk,
            "drMatch_mtd": drMatch_mtd,
            "res1": res1,
            "res2": res2,
        }
    }
    print(">>> Data processing completed successfully!")
    return data


# ----------------- PLOTTING FUNCTION -----------------
def make_plots(data, args):
    print(">>> Starting plotting with mplhep CMS style...")

    # Write log file
    with open("conversions.log", "w") as f_log:
        for line in data["log_lines"]:
            f_log.write(line + "\n")

    # 1. kinematics_photon
    vars_photon = [
        ("pt", r"$p_\mathrm{T}\ [\mathrm{GeV}]$", data["kinematics_photon"]["pt"]),
        ("eta", r"$\eta$", data["kinematics_photon"]["eta"]),
        ("phi", r"$\phi\ [\mathrm{rad}]$", data["kinematics_photon"]["phi"]),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(36, 12))
    for ax, (var_name, var_label, arr) in zip(axs, vars_photon):
        counts, bins = np.histogram(arr, bins=50)
        hep.histplot(counts, bins, ax=ax, histtype="step", color=cms_palette[0], linewidth=2)
        ax.set_xlabel(var_label)
        ax.set_ylabel("Entries")
        add_cms_label(ax)
    fig.tight_layout()
    fig.savefig("kinematics_photon.png")
    fig.savefig("kinematics_photon.pdf")
    plt.close(fig)

    # 2. kinematics_electron
    vars_ele = [
        ("pt", r"$p_\mathrm{T}\ [\mathrm{GeV}]$", data["kinematics_electron"]["ele1_pt"], data["kinematics_electron"]["ele2_pt"]),
        ("eta", r"$\eta$", data["kinematics_electron"]["ele1_eta"], data["kinematics_electron"]["ele2_eta"]),
        ("phi", r"$\phi\ [\mathrm{rad}]$", data["kinematics_electron"]["ele1_phi"], data["kinematics_electron"]["ele2_phi"]),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(36, 12))
    for ax, (var_name, var_label, arr1, arr2) in zip(axs, vars_ele):
        c1, b1 = np.histogram(arr1, bins=50)
        c2, b2 = np.histogram(arr2, bins=50)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="Leading", linewidth=2)
        hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="Subleading", linewidth=2)
        ax.set_xlabel(var_label)
        ax.set_ylabel("Entries")
        ax.legend()
        add_cms_label(ax)
    fig.tight_layout()
    fig.savefig("kinematics_electron.png")
    fig.savefig("kinematics_electron.pdf")
    plt.close(fig)

    # 3. conversion_vertex_xy (scatter - square)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(data["vertex_pos"]["x"], data["vertex_pos"]["y"], s=1, color=cms_palette[0])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-125, 125)
    ax.set_ylim(-125, 125)
    ax.set_xlabel(r"$x\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$y\ [\mathrm{cm}]$")
    add_cms_label(ax)
    fig.tight_layout()
    fig.savefig("conversion_vertex_xy.png")
    fig.savefig("conversion_vertex_xy.pdf")
    plt.close(fig)

    # 4. conversion_vertex_xy_hist (2D hist - square)
    fig, ax = plt.subplots(figsize=(12, 10))
    h = ax.hist2d(data["vertex_pos"]["x"], data["vertex_pos"]["y"], bins=30, cmap="viridis")
    ax.set_box_aspect(1)
    ax.set_xlim(-125, 125)
    ax.set_ylim(-125, 125)
    ax.set_xlabel(r"$x\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$y\ [\mathrm{cm}]$")
    fig.colorbar(h[3], ax=ax)
    add_cms_label(ax)
    fig.tight_layout()
    fig.savefig("conversion_vertex_xy_hist.png")
    fig.savefig("conversion_vertex_xy_hist.pdf")
    plt.close(fig)

    # 5. conversion_vertex_z_rho (scatter - square)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(data["vertex_pos"]["rho"], data["vertex_pos"]["z"], s=1, color=cms_palette[0])
    ax.set_box_aspect(1)
    ax.set_xlim(0, 125)
    ax.set_ylim(-200, 200)
    ax.set_xlabel(r"$\rho\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$z\ [\mathrm{cm}]$")
    add_cms_label(ax)
    fig.tight_layout()
    fig.savefig("conversion_vertex_z_rho.png")
    fig.savefig("conversion_vertex_z_rho.pdf")
    plt.close(fig)

    # 6. conversion_vertex_z_rho_hist (2D hist - square)
    fig, ax = plt.subplots(figsize=(12, 10))
    h = ax.hist2d(data["vertex_pos"]["rho"], data["vertex_pos"]["z"], bins=30, cmap="viridis")
    ax.set_box_aspect(1)
    ax.set_xlim(0, 125)
    ax.set_ylim(-200, 200)
    ax.set_xlabel(r"$\rho\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$z\ [\mathrm{cm}]$")
    fig.colorbar(h[3], ax=ax)
    add_cms_label(ax)
    fig.tight_layout()
    fig.savefig("conversion_vertex_z_rho_hist.png")
    fig.savefig("conversion_vertex_z_rho_hist.pdf")
    plt.close(fig)

    # 7. mtd_cluster_positions
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    ax = axs[0]
    ax.scatter(data["mtd_cluster_positions"]["x1_all"], data["mtd_cluster_positions"]["y1_all"], s=1, color=cms_palette[0], label="Lead")
    ax.scatter(data["mtd_cluster_positions"]["x2_all"], data["mtd_cluster_positions"]["y2_all"], s=1, color=cms_palette[1], label="Sublead")
    ax.scatter(data["mtd_cluster_positions"]["x1_dir"], data["mtd_cluster_positions"]["y1_dir"], s=1, color=cms_palette[2], label="ele1 direct")
    ax.scatter(data["mtd_cluster_positions"]["x2_dir"], data["mtd_cluster_positions"]["y2_dir"], s=1, color=cms_palette[3], label="ele2 direct")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-125, 125)
    ax.set_ylim(-125, 125)
    ax.set_xlabel(r"$x\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$y\ [\mathrm{cm}]$")
    ax.legend()
    ax.grid(True)
    add_cms_label(ax)

    ax = axs[1]
    c, b = np.histogram(data["mtd_cluster_positions"]["mtd_r"], bins=50)
    hep.histplot(c, b, ax=ax, histtype="step", color=cms_palette[0], linewidth=2)
    ax.set_xlabel(r"$r\ [\mathrm{cm}]$")
    ax.set_ylabel("Entries")
    ax.grid(True)
    add_cms_label(ax)

    fig.tight_layout()
    fig.savefig("mtd_cluster_positions.png")
    fig.savefig("mtd_cluster_positions.pdf")
    plt.close(fig)

    # 8. genPV
    fig, axs = plt.subplots(2, 3, figsize=(36, 24))
    gen_vars = [
        ("x", r"$x_\mathrm{PV}\ [\mathrm{cm}]$", data["genPV"]["x"]),
        ("y", r"$y_\mathrm{PV}\ [\mathrm{cm}]$", data["genPV"]["y"]),
        ("z", r"$z_\mathrm{PV}\ [\mathrm{cm}]$", data["genPV"]["z"]),
        ("t", r"$t_\mathrm{PV}\ [\mathrm{ns}]$", data["genPV"]["t"]),
    ]
    for i, (name, label, arr) in enumerate(gen_vars):
        ax = axs.flatten()[i]
        c, b = np.histogram(arr, bins=50)
        hep.histplot(c, b, ax=ax, histtype="step", color=cms_palette[i % len(cms_palette)], label=name, linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Entries")
        ax.legend()
        add_cms_label(ax)

    ax = axs[1, 1]
    ax.scatter(data["genPV"]["x"], data["genPV"]["y"], s=1, alpha=0.5, color=cms_palette[0])
    ax.set_xlabel(r"$x_\mathrm{PV}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$y_\mathrm{PV}\ [\mathrm{cm}]$")
    ax.grid(True)
    add_cms_label(ax)

    # hide the unused panel [1, 2]
    axs[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig("genPV.png")
    fig.savefig("genPV.pdf")
    plt.close(fig)

    # 9. mtd_cluster_info
    fig, axs = plt.subplots(4, 3, figsize=(36, 48))
    type_dict = {0: "Direct", 1: "Secondary", 2: "Looper", 3: "Calo backscatter"}

    # [0, 0] MTD time
    ax = axs[0, 0]
    c_all, b_all = np.histogram(data["mtd_cluster_info"]["t_all"], bins=100, range=[2.5, 25])
    hep.histplot(c_all, b_all, ax=ax, histtype="fill", color="gray", alpha=0.3, label="All clusters")
    for i in range(4):
        c_t, b_t = np.histogram(data["mtd_cluster_info"]["t_by_type"][i], bins=100, range=[2.5, 25])
        hep.histplot(c_t, b_t, ax=ax, histtype="step", color=cms_palette[i], label=type_dict[i], linewidth=2)
    ax.set_xlabel(r"$t\ [\mathrm{ns}]$")
    ax.set_ylabel("Entries")
    ax.set_xlim(2.5, 25)
    ax.legend()
    add_cms_label(ax)

    # [1, 0] 2 direct hits
    ax = axs[1, 0]
    c1, b1 = np.histogram(data["mtd_cluster_info"]["t_2direct_all"], bins=100)
    hep.histplot(c1, b1, ax=ax, histtype="fill", color="gray", alpha=0.3, label="All photons")
    c2, b2 = np.histogram(data["mtd_cluster_info"]["t_2direct_leading"], bins=100)
    hep.histplot(c2, b2, ax=ax, histtype="fill", color=cms_palette[0], alpha=0.3, label="Leading photons")
    ax.set_xlabel(r"$t\ [\mathrm{ns}]$")
    ax.set_ylabel("Entries")
    ax.legend()
    add_cms_label(ax)

    # [0, 1] MTD time vs conversion radius
    ax = axs[0, 1]
    for idx, (rmin, rmax, arr) in enumerate(data["mtd_cluster_info"]["t_direct_vs_rho"]):
        c_r, b_r = np.histogram(arr, bins=50, range=[2.5, 25], density=True)
        hep.histplot(c_r, b_r, ax=ax, histtype="step", color=cms_palette[idx % len(cms_palette)],
                     label=rf"${rmin:.0f} < R_\mathrm{{conv}} < {rmax:.0f}\ \mathrm{{cm}}$", linewidth=2)
    ax.set_xlabel(r"$t\ [\mathrm{ns}]$")
    ax.set_ylabel("Density")
    ax.legend()
    add_cms_label(ax)

    # [1, 1] MTD time vs conversion radius (photons with 2 DIRECT clus)
    ax = axs[1, 1]
    for idx, (rmin, rmax, arr) in enumerate(data["mtd_cluster_info"]["t_direct_2dir_vs_rho"]):
        c_r, b_r = np.histogram(arr, bins=50, range=[2.5, 25], density=True)
        hep.histplot(c_r, b_r, ax=ax, histtype="step", color=cms_palette[idx % len(cms_palette)],
                     label=rf"${rmin:.0f} < R_\mathrm{{conv}} < {rmax:.0f}\ \mathrm{{cm}}$", linewidth=2)
    ax.set_xlabel(r"$t\ [\mathrm{ns}]$")
    ax.set_ylabel("Density")
    ax.legend()
    add_cms_label(ax)

    # [0, 2] cluster type distribution
    ax = axs[0, 2]
    c_all, b_all = np.histogram(data["mtd_cluster_info"]["clus_type_all"], bins=4, range=[-0.5, 3.5])
    hep.histplot(c_all, b_all, ax=ax, histtype="step", color=cms_palette[0], label="All conversions", linewidth=2)
    c_mtd, b_mtd = np.histogram(data["mtd_cluster_info"]["clus_type_mtd"], bins=4, range=[-0.5, 3.5])
    hep.histplot(c_mtd, b_mtd, ax=ax, histtype="step", color=cms_palette[1], label="MTD conversions", linewidth=2)
    ax.set_xlabel("Cluster type")
    ax.set_ylabel("Entries")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Direct", "Secondary", "Looper", "Calo backscatter"])
    ax.legend()
    add_cms_label(ax)

    # [1, 2] Free slot
    axs[1, 2].axis("off")

    # [2, 0], [2, 1], [2, 2] x, y, z positions
    clus_pos = [
        ("x", r"$x\ [\mathrm{cm}]$", data["mtd_cluster_info"]["xClus_all"], [-120, 120]),
        ("y", r"$y\ [\mathrm{cm}]$", data["mtd_cluster_info"]["yClus_all"], [-120, 120]),
        ("z", r"$z\ [\mathrm{cm}]$", data["mtd_cluster_info"]["zClus_all"], [-300, 300]),
    ]
    for ax, (dim, dim_label, arr, p_range) in zip(axs[2], clus_pos):
        c_p, b_p = np.histogram(arr, bins=500, range=p_range)
        hep.histplot(c_p, b_p, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
        ax.set_xlabel(dim_label)
        ax.set_ylabel("Entries")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

    # [3, 0] phi
    ax = axs[3, 0]
    c_phi, b_phi = np.histogram(data["mtd_cluster_info"]["phi_clus"], bins=400)
    hep.histplot(c_phi, b_phi, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    ax.grid(True)
    ax.set_xlabel(r"$\phi\ [\mathrm{rad}]$")
    ax.set_ylabel("Entries")
    ax.legend()
    add_cms_label(ax)

    # [3, 1] rho
    ax = axs[3, 1]
    c_rho, b_rho = np.histogram(data["mtd_cluster_info"]["rho_clus"], bins=400, range=[115.4, 116.4])
    hep.histplot(c_rho, b_rho, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    ax.set_xlabel(r"$R\ [\mathrm{cm}]$")
    ax.set_ylabel("Entries")
    ax.legend()
    add_cms_label(ax)

    # [3, 2] Free slot
    axs[3, 2].axis("off")

    fig.tight_layout()
    fig.savefig("mtd_cluster_info.png")
    fig.savefig("mtd_cluster_info.pdf")
    plt.close(fig)

    # 9b. Barrel MTD clusters: Time and Energy by category
    type_dict = {0: "Direct", 1: "Secondary", 2: "Looper", 3: "Calo backscatter"}
    bclus = data.get("mtd_cluster_barrel", data.get("mtd_cluster_info", {}))
    t_barrel_all = bclus.get("t_all", bclus.get("barrel_t_all"))
    t_barrel_by_type = bclus.get("t_by_type", bclus.get("barrel_t_by_type"))
    e_barrel_all = bclus.get("energy_all", bclus.get("barrel_energy_all"))
    e_barrel_by_type = bclus.get("energy_by_type", bclus.get("barrel_energy_by_type"))

    if t_barrel_all is not None and e_barrel_all is not None:
        # Combined Figure (2 square panels side-by-side, smaller canvas for nice font sizing)
        fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(16, 8))
        ax_t.set_box_aspect(1)
        ax_e.set_box_aspect(1)

        # Left panel: Time in [0, 25] ns
        c_all_t, b_all_t = np.histogram(t_barrel_all, bins=100, range=[0, 25])
        hep.histplot(c_all_t, b_all_t, ax=ax_t, histtype="fill", color="gray", alpha=0.3, label="All clusters")
        for i in range(4):
            c_t, b_t = np.histogram(t_barrel_by_type[i], bins=100, range=[0, 25])
            hep.histplot(c_t, b_t, ax=ax_t, histtype="step", color=cms_palette[i], label=type_dict[i], linewidth=2)
        ax_t.set_xlabel(r"$t\ [\mathrm{ns}]$")
        ax_t.set_ylabel("Entries")
        ax_t.set_xlim(0, 25)
        ax_t.set_ylim(0, max(c_all_t) * 1.25)
        ax_t.legend(fontsize=14)
        add_cms_label(ax_t)

        # Right panel: Energy in [10^-4, 10^3] GeV (log-scale x)
        bins_e = np.logspace(-4, 3, 71)
        c_all_e, b_all_e = np.histogram(e_barrel_all, bins=bins_e)
        hep.histplot(c_all_e, b_all_e, ax=ax_e, histtype="fill", color="gray", alpha=0.3, label="All clusters")
        for i in range(4):
            c_e, b_e = np.histogram(e_barrel_by_type[i], bins=bins_e)
            hep.histplot(c_e, b_e, ax=ax_e, histtype="step", color=cms_palette[i], label=type_dict[i], linewidth=2)
        ax_e.set_xscale("log")
        ax_e.set_xlabel(r"$E\ [\mathrm{GeV}]$")
        ax_e.set_ylabel("Entries")
        ax_e.set_xlim(1e-4, 1e3)
        ax_e.set_ylim(0, max(c_all_e) * 1.25)
        ax_e.legend(fontsize=14)
        add_cms_label(ax_e)

        fig.tight_layout()
        fig.savefig("mtd_clusters_barrel_time_energy.png")
        fig.savefig("mtd_clusters_barrel_time_energy.pdf")
        plt.close(fig)

        # Standalone square Time figure
        fig_t, ax_t_s = plt.subplots(figsize=(8, 8))
        ax_t_s.set_box_aspect(1)
        hep.histplot(c_all_t, b_all_t, ax=ax_t_s, histtype="fill", color="gray", alpha=0.3, label="All clusters")
        for i in range(4):
            c_t, b_t = np.histogram(t_barrel_by_type[i], bins=100, range=[0, 25])
            hep.histplot(c_t, b_t, ax=ax_t_s, histtype="step", color=cms_palette[i], label=type_dict[i], linewidth=2)
        ax_t_s.set_xlabel(r"$t\ [\mathrm{ns}]$")
        ax_t_s.set_ylabel("Entries")
        ax_t_s.set_xlim(0, 25)
        ax_t_s.set_ylim(0, max(c_all_t) * 1.25)
        ax_t_s.legend(fontsize=14)
        add_cms_label(ax_t_s, fontsize=18)
        fig_t.tight_layout()
        fig_t.savefig("mtd_clusters_barrel_time.png")
        fig_t.savefig("mtd_clusters_barrel_time.pdf")
        plt.close(fig_t)

        # Standalone square Energy figure
        fig_e, ax_e_s = plt.subplots(figsize=(8, 8))
        ax_e_s.set_box_aspect(1)
        hep.histplot(c_all_e, b_all_e, ax=ax_e_s, histtype="fill", color="gray", alpha=0.3, label="All clusters")
        for i in range(4):
            c_e, b_e = np.histogram(e_barrel_by_type[i], bins=bins_e)
            hep.histplot(c_e, b_e, ax=ax_e_s, histtype="step", color=cms_palette[i], label=type_dict[i], linewidth=2)
        ax_e_s.set_xscale("log")
        ax_e_s.set_xlabel(r"$E\ [\mathrm{GeV}]$")
        ax_e_s.set_ylabel("Entries")
        ax_e_s.set_xlim(1e-4, 1e3)
        ax_e_s.set_ylim(0, max(c_all_e) * 1.25)
        ax_e_s.legend(fontsize=14)
        add_cms_label(ax_e_s, fontsize=18)
        fig_e.tight_layout()
        fig_e.savefig("mtd_clusters_barrel_energy.png")
        fig_e.savefig("mtd_clusters_barrel_energy.pdf")
        plt.close(fig_e)

        # Standalone square Energy figure (log y-scale for inspecting tails)
        fig_e_logy, ax_e_logy = plt.subplots(figsize=(8, 8))
        ax_e_logy.set_box_aspect(1)
        hep.histplot(c_all_e, b_all_e, ax=ax_e_logy, histtype="fill", color="gray", alpha=0.3, label="All clusters")
        for i in range(4):
            c_e, b_e = np.histogram(e_barrel_by_type[i], bins=bins_e)
            hep.histplot(c_e, b_e, ax=ax_e_logy, histtype="step", color=cms_palette[i], label=type_dict[i], linewidth=2)
        ax_e_logy.set_xscale("log")
        ax_e_logy.set_yscale("log")
        ax_e_logy.set_xlabel(r"$E\ [\mathrm{GeV}]$")
        ax_e_logy.set_ylabel("Entries")
        ax_e_logy.set_xlim(1e-4, 1e3)
        ax_e_logy.set_ylim(0.5, max(c_all_e) * 5)
        ax_e_logy.legend(fontsize=14)
        add_cms_label(ax_e_logy, fontsize=18)
        fig_e_logy.tight_layout()
        fig_e_logy.savefig("mtd_clusters_barrel_energy_logy.png")
        fig_e_logy.savefig("mtd_clusters_barrel_energy_logy.pdf")
        plt.close(fig_e_logy)

    # 10. conversion_kinematics
    fig, axs = plt.subplots(3, 6, figsize=(72, 36))
    ck = data["conversion_kinematics"]

    # [0, 0] deltaPt
    ax = axs[0, 0]
    c, b = np.histogram(ck["dPt"], bins=50)
    hep.histplot(c, b, ax=ax, histtype="step", color=cms_palette[0], linewidth=2)
    ax.set_xlabel(r"$\Delta p_\mathrm{T} / p_\mathrm{T}^\mathrm{lead}$")
    ax.set_ylabel("Entries")
    add_cms_label(ax)

    # [0, 1] dt
    ax = axs[0, 1]
    c1, b1 = np.histogram(ck["dt_all"], bins=100, range=[-0.3, 0.3])
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    c2, b2 = np.histogram(ck["dt_mtd"], bins=100, range=[-0.3, 0.3])
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="MTD conversions", linewidth=2)
    ax.set_xlabel(r"$\Delta t\ [\mathrm{ns}]$")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend()
    add_cms_label(ax)

    # [0, 2] deltaPhi
    ax = axs[0, 2]
    c1, b1 = np.histogram(ck["dphi_all"], bins=100, range=[-0.3, 0.3])
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    c2, b2 = np.histogram(ck["dphi_mtd"], bins=100, range=[-0.3, 0.3])
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="MTD conversions", linewidth=2)
    ax.set_xlabel(r"$\Delta \phi(e_1, e_2)$")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend()
    add_cms_label(ax)

    # [0, 3] deltaPhi vs conversion radius
    ax = axs[0, 3]
    ax.scatter(ck["rho_good"], ck["dphi_good"], s=1, color=cms_palette[0])
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$\Delta \phi(e_1, e_2)$")
    add_cms_label(ax)

    # [0, 4] deltaEta
    ax = axs[0, 4]
    c1, b1 = np.histogram(ck["deta_all"], bins=100, range=[-0.3, 0.3])
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    c2, b2 = np.histogram(ck["deta_mtd"], bins=100, range=[-0.3, 0.3])
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="MTD conversions", linewidth=2)
    ax.set_xlabel(r"$\Delta \eta(e_1, e_2)$")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend()
    add_cms_label(ax)

    # [0, 5] deltaEta vs conversion radius
    ax = axs[0, 5]
    ax.scatter(ck["rho_good"], ck["deta_good"], s=1, color=cms_palette[0])
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$\Delta \eta(e_1, e_2)$")
    add_cms_label(ax)

    # [1, 0] photon-ele delta t
    ax = axs[1, 0]
    p_range = [-0.02, 20]
    c1, b1 = np.histogram(ck["dt1_all"], bins=50, range=p_range)
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All, leading", linewidth=2)
    c2, b2 = np.histogram(ck["dt2_all"], bins=50, range=p_range)
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="All, subleading", linewidth=2)
    c3, b3 = np.histogram(ck["dt1_mtd"], bins=50, range=p_range)
    hep.histplot(c3, b3, ax=ax, histtype="fill", alpha=0.5, color=cms_palette[0], label="MTD conversions, leading")
    c4, b4 = np.histogram(ck["dt2_mtd"], bins=50, range=p_range)
    hep.histplot(c4, b4, ax=ax, histtype="fill", alpha=0.5, color=cms_palette[1], label="MTD conversions, subleading")
    ax.set_xlabel(r"$\Delta \mathrm{TOF}\ [\mathrm{ns}]$")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend()
    add_cms_label(ax)

    # [1, 1] photon-ele deltaPhi
    ax = axs[1, 1]
    p_range = [-0.5, 0.5]
    c1, b1 = np.histogram(ck["dphi1_photon"], bins=50, range=p_range)
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All, leading", linewidth=2)
    c2, b2 = np.histogram(ck["dphi2_photon"], bins=50, range=p_range)
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="All, subleading", linewidth=2)
    c3, b3 = np.histogram(ck["dphi1_photon_mtd"], bins=50, range=p_range)
    hep.histplot(c3, b3, ax=ax, histtype="fill", alpha=0.5, color=cms_palette[0], label="MTD conversions, leading")
    c4, b4 = np.histogram(ck["dphi2_photon_mtd"], bins=50, range=p_range)
    hep.histplot(c4, b4, ax=ax, histtype="fill", alpha=0.5, color=cms_palette[1], label="MTD conversions, subleading")
    ax.set_xlabel(r"$\Delta \phi(e, \gamma)$")
    ax.set_ylabel("Entries")
    ax.legend()
    add_cms_label(ax)

    # [1, 2] deltaPhi(ele, photon) vs photon pT (scatter)
    ax = axs[1, 2]
    ax.scatter(ck["pt_good"], ck["dphi1_photon"], s=1, color=cms_palette[0], label="Lead")
    ax.scatter(ck["pt_good"], ck["dphi2_photon"], s=1, color=cms_palette[1], label="Sublead")
    ax.set_xlabel(r"$p_\mathrm{T}\ [\mathrm{GeV}]$")
    ax.set_ylabel(r"$\Delta \phi(e, \gamma)$")
    ax.legend()
    add_cms_label(ax)

    # [2, 2] deltaPhi(ele, photon) vs photon pT (2D hist)
    ax = axs[2, 2]
    arrx = np.concatenate([ck["pt_good"], ck["pt_good"]])
    arry = np.concatenate([ck["dphi1_photon"], ck["dphi2_photon"]])
    h = ax.hist2d(arrx, arry, bins=30, norm=mpl.colors.LogNorm(), range=[[0, 10], [-1, 1]], cmap="viridis")
    fig.colorbar(h[3], ax=ax, fraction=0.05)
    ax.set_xlabel(r"$p_\mathrm{T}\ [\mathrm{GeV}]$")
    ax.set_ylabel(r"$\Delta \phi(e, \gamma)$")
    add_cms_label(ax)

    # [1, 3] deltaPhi(ele, photon) vs conversion radius (scatter)
    ax = axs[1, 3]
    ax.scatter(ck["rho_good"], ck["dphi1_photon"], s=1, color=cms_palette[0], label="Lead")
    ax.scatter(ck["rho_good"], ck["dphi2_photon"], s=1, color=cms_palette[1], label="Sublead")
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$\Delta \phi(e, \gamma)$")
    ax.legend()
    add_cms_label(ax)

    # [2, 3] deltaPhi(ele, photon) vs conversion radius (2D hist)
    ax = axs[2, 3]
    arrx = np.concatenate([ck["rho_good"], ck["rho_good"]])
    arry = np.concatenate([ck["dphi1_photon"], ck["dphi2_photon"]])
    h = ax.hist2d(arrx, arry, bins=30, norm=mpl.colors.LogNorm(), range=[[0, 120], [-1, 1]], cmap="viridis")
    fig.colorbar(h[3], ax=ax, fraction=0.05)
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$\Delta \phi(e, \gamma)$")
    add_cms_label(ax)

    # [1, 4] deltaEta(ele, photon) vs conversion radius (scatter)
    ax = axs[1, 4]
    ax.scatter(ck["rho_good"], ck["deta1_photon"], s=1, color=cms_palette[0], label="Lead")
    ax.scatter(ck["rho_good"], ck["deta2_photon"], s=1, color=cms_palette[1], label="Sublead")
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$\Delta \eta(e, \gamma)$")
    ax.legend()
    add_cms_label(ax)

    # [2, 4] deltaEta(ele, photon) vs conversion radius (2D hist)
    ax = axs[2, 4]
    arrx = np.concatenate([ck["rho_good"], ck["rho_good"]])
    arry = np.concatenate([ck["deta1_photon"], ck["deta2_photon"]])
    h = ax.hist2d(arrx, arry, bins=30, norm=mpl.colors.LogNorm(), range=[[0, 120], [-1, 1]], cmap="viridis")
    fig.colorbar(h[3], ax=ax, fraction=0.05)
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel(r"$\Delta \eta(e, \gamma)$")
    add_cms_label(ax)

    # [1, 5] conversion radius
    ax = axs[1, 5]
    c1, b1 = np.histogram(ck["rho_all"], bins=50, density=True)
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    c2, b2 = np.histogram(ck["rho_good"], bins=50, density=True)
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="2 direct hits", linewidth=2)
    ax.set_xlabel(r"$R_\mathrm{conv}\ [\mathrm{cm}]$")
    ax.set_ylabel("Density")
    ax.legend()
    add_cms_label(ax)

    # hide empty subplots
    axs[2, 0].axis("off")
    axs[2, 1].axis("off")
    axs[2, 5].axis("off")

    fig.tight_layout()
    fig.savefig("conversion_kinematics.png")
    fig.savefig("conversion_kinematics.pdf")
    plt.close(fig)

    # 11. mtd_deltaTOF_vs_dphi_scatter
    fig, axs = plt.subplots(3, 3, figsize=(36, 36))
    s_dphi = data["deltaTOF_vs_dphi_scatter"]

    # [0, 0] Direct
    ax = axs[0, 0]
    ax.scatter(s_dphi["direct"][0], s_dphi["direct"][1], s=1, color=cms_palette[0], label="Direct")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [0, 1] Direct MTD
    ax = axs[0, 1]
    ax.scatter(s_dphi["direct_mtd"][0], s_dphi["direct_mtd"][1], s=1, color=cms_palette[0], label=r"Direct, $R_\mathrm{conv} \geq 115\ \mathrm{cm}$")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [0, 2] Direct Tracker
    ax = axs[0, 2]
    ax.scatter(s_dphi["direct_trk"][0], s_dphi["direct_trk"][1], s=1, color=cms_palette[0], label=r"Direct, $R_\mathrm{conv} < 90\ \mathrm{cm}$")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [1, 0] All clusters
    ax = axs[1, 0]
    ax.scatter(s_dphi["all"][0], s_dphi["all"][1], s=1, color=cms_palette[0], label="All clusters")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [1, 1] All clusters MTD
    ax = axs[1, 1]
    ax.scatter(s_dphi["all_mtd"][0], s_dphi["all_mtd"][1], s=1, color=cms_palette[0], label=r"All, $R_\mathrm{conv} \geq 115\ \mathrm{cm}$")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [1, 2] Tracker by cluster type
    ax = axs[1, 2]
    for i in range(2):
        ax.scatter(s_dphi["type_trk"][i][0], s_dphi["type_trk"][i][1], s=1, color=cms_palette[i], label=f"{type_dict[i]} (trk)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend(markerscale=5)
    add_cms_label(ax)

    # [2, 0] Eta direct
    ax = axs[2, 0]
    ax.scatter(s_dphi["eta_direct"]["inner"][0], s_dphi["eta_direct"]["inner"][1], s=1, color=cms_palette[0], label=r"$|\eta| < 0.6$")
    ax.scatter(s_dphi["eta_direct"]["outer"][0], s_dphi["eta_direct"]["outer"][1], s=1, color=cms_palette[1], label=r"$|\eta| \geq 0.6$")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [2, 1] Eta MTD
    ax = axs[2, 1]
    ax.scatter(s_dphi["eta_mtd"]["inner"][0], s_dphi["eta_mtd"]["inner"][1], s=1, color=cms_palette[0], label=r"$|\eta| < 0.6$ (MTD)")
    ax.scatter(s_dphi["eta_mtd"]["outer"][0], s_dphi["eta_mtd"]["outer"][1], s=1, color=cms_palette[1], label=r"$|\eta| \geq 0.6$ (MTD)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    # [2, 2] Eta Tracker
    ax = axs[2, 2]
    ax.scatter(s_dphi["eta_trk"]["inner"][0], s_dphi["eta_trk"]["inner"][1], s=1, color=cms_palette[0], label=r"$|\eta| < 0.6$ (trk)")
    ax.scatter(s_dphi["eta_trk"]["outer"][0], s_dphi["eta_trk"]["outer"][1], s=1, color=cms_palette[1], label=r"$|\eta| \geq 0.6$ (trk)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    ax.grid(True)
    add_cms_label(ax)

    for ax in axs.flatten():
        ax.set_xlabel(r"$\Delta \phi(e, \gamma)$")
        ax.set_ylabel(r"$\Delta \mathrm{TOF}\ [\mathrm{ns}]$")

    fig.tight_layout()
    fig.savefig("mtd_deltaTOF_vs_dphi_scatter.png")
    fig.savefig("mtd_deltaTOF_vs_dphi_scatter.pdf")
    plt.close(fig)

    # 12. mtd_deltaTOF_vs_deta_scatter
    fig, axs = plt.subplots(2, 3, figsize=(36, 24))
    s_deta = data["deltaTOF_vs_deta_scatter"]

    ax = axs[0, 0]
    ax.scatter(s_deta["direct"][0], s_deta["direct"][1], s=1, color=cms_palette[0], label="Direct")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    ax = axs[0, 1]
    ax.scatter(s_deta["direct_mtd"][0], s_deta["direct_mtd"][1], s=1, color=cms_palette[0], label=r"Direct, $R_\mathrm{conv} \geq 115\ \mathrm{cm}$")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    ax = axs[0, 2]
    ax.scatter(s_deta["direct_trk"][0], s_deta["direct_trk"][1], s=1, color=cms_palette[0], label=r"Direct, $R_\mathrm{conv} < 90\ \mathrm{cm}$")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    ax = axs[1, 0]
    ax.scatter(s_deta["all"][0], s_deta["all"][1], s=1, color=cms_palette[0], label="All")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    ax = axs[1, 1]
    ax.scatter(s_deta["all_mtd"][0], s_deta["all_mtd"][1], s=1, color=cms_palette[0], label=r"All, $R_\mathrm{conv} \geq 115\ \mathrm{cm}$")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()
    add_cms_label(ax)

    ax = axs[1, 2]
    for i in range(2):
        ax.scatter(s_deta["type_trk"][i][0], s_deta["type_trk"][i][1], s=1, color=cms_palette[i], label=f"{type_dict[i]} (trk)")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.01, 0.2)
    ax.legend(markerscale=5)
    add_cms_label(ax)

    for ax in axs.flatten():
        ax.set_xlabel(r"$\Delta \eta(e, \gamma)$")
        ax.set_ylabel(r"$\Delta \mathrm{TOF}\ [\mathrm{ns}]$")

    fig.tight_layout()
    fig.savefig("mtd_deltaTOF_vs_deta_scatter.png")
    fig.savefig("mtd_deltaTOF_vs_deta_scatter.pdf")
    plt.close(fig)

    # 13. mtd_deltaTOF_vs_dphi (hist & profile)
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    ax = axs[0]
    for idx, (rmin, rmax, arr) in enumerate(data["deltaTOF_1d_by_rho"]):
        c, b = np.histogram(arr, bins=30, range=[-0.01, 0.1], density=True)
        hep.histplot(c, b, ax=ax, histtype="step", color=cms_palette[idx % len(cms_palette)],
                     label=rf"${rmin:.0f} < R_\mathrm{{conv}} < {rmax:.0f}\ \mathrm{{cm}}$", linewidth=2)
    ax.set_xlabel(r"$\Delta \mathrm{TOF}\ [\mathrm{ns}]$")
    ax.set_ylabel("Density")
    ax.legend()
    add_cms_label(ax)

    ax = axs[1]
    p_10bins = data.get("profile_scatter_data_10bins", data["profile_scatter_data"])
    for idx, p in enumerate(p_10bins):
        ax.errorbar(p["xs"], p["ys"], xerr=p["xs_err"], yerr=p["ys_err"], fmt="o",
                    color=cms_palette[idx % len(cms_palette)],
                    label=rf"${p['rho_min']:.0f} < R_\mathrm{{conv}} < {p['rho_max']:.0f}\ \mathrm{{cm}}$",
                    markersize=5, capsize=5)
    ax.set_xlabel(r"$\Delta \phi(e, \gamma)$")
    ax.set_ylabel(r"$\Delta \mathrm{TOF}\ [\mathrm{ns}]$")
    ax.legend()
    add_cms_label(ax)

    fig.tight_layout()
    fig.savefig("mtd_deltaTOF_vs_dphi.png")
    fig.savefig("mtd_deltaTOF_vs_dphi.pdf")
    plt.close(fig)

    # 14. mtd_deltaTOF_vs_dphi_scatter_profile
    ncols = len(data["profile_scatter_data"])
    fig, axs = plt.subplots(1, ncols, figsize=(12 * ncols, 12))
    dynamic_x_ranges = {0: 0.5, 30: 0.4, 60: 0.3, 90: 0.07}
    for idx, (ax, p) in enumerate(zip(axs, data["profile_scatter_data"])):
        x_lim = p.get("x_lim", dynamic_x_ranges.get(int(round(p["rho_min"])), [0.5, 0.4, 0.3, 0.07][idx] if idx < 4 else 0.5))
        ax.scatter(p["arrx"], p["arry"], s=2, alpha=0.1, color=cms_palette[0], label="Direct")
        ax.errorbar(p["xs"], p["ys"], xerr=p["xs_err"], yerr=p["ys_err"], fmt="o",
                    color=cms_palette[0],
                    label=rf"${p['rho_min']:.0f} < R_\mathrm{{conv}} < {p['rho_max']:.0f}\ \mathrm{{cm}}$",
                    markersize=5, capsize=5)
        xfit = np.linspace(-x_lim, x_lim, 1000)
        yfit = parabola(xfit, *p["popt"])
        ax.plot(xfit, yfit, color=cms_palette[2],
                label=f"Fit: {p['popt'][0]:.2g}$x^2$ + {p['popt'][1]:.2g}$x$ + {p['popt'][2]:.2g}")
        ax.set_xlabel(r"$\Delta \phi(e, \gamma)$")
        ax.set_ylabel(r"$\Delta \mathrm{TOF}\ [\mathrm{ns}]$")
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-0.01, 0.2)
        ax.legend(loc="upper right")
        add_cms_label(ax)

    fig.tight_layout()
    fig.savefig("mtd_deltaTOF_vs_dphi_scatter_profile.png")
    fig.savefig("mtd_deltaTOF_vs_dphi_scatter_profile.pdf")
    plt.close(fig)

    # 15. t0_resolution
    ncols_res = len(data["resolution_bins"])
    fig, axs = plt.subplots(3, ncols_res, figsize=(12 * ncols_res, 36))

    ax = axs[0, 0]
    p_range = [-0.05, 0.2]
    c1, b1 = np.histogram(data["all_tof_corr"], bins=100, range=p_range)
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All conversions", linewidth=2)
    c2, b2 = np.histogram(data["mtd_tof_corr"], bins=100, range=p_range)
    hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label="MTD conversions", linewidth=2)
    ax.set_xlabel(r"Time correction [ns]")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    add_cms_label(ax)

    for i in range(1, ncols_res):
        axs[0, i].axis("off")

    for i, rbin in enumerate(data["resolution_bins"]):
        for j, log_scale in enumerate([False, True], start=1):
            ax = axs[j, i]
            c_c, b_c = np.histogram(rbin["corr"], bins=100, range=[-0.05, 0.05])
            hep.histplot(c_c, b_c, ax=ax, histtype="step", color=cms_palette[0], label="Corrected", linewidth=2)
            c_p, b_p = np.histogram(rbin["precorr"], bins=100, range=[-0.05, 0.05])
            hep.histplot(c_p, b_p, ax=ax, histtype="step", color=cms_palette[1], label="NO corrections", linewidth=2)
            ax.axvline(0, color=cms_palette[2], linestyle="dashed", label=r"$\Delta t = 0$")
            ax.set_xlabel(r"$\Delta t\ [\mathrm{ns}]$")
            ax.set_ylabel("Entries")
            if log_scale:
                ax.set_yscale("log")
            range_str = "All conversions" if rbin["is_all"] else rf"${rbin['rmin']:.0f} < R_\mathrm{{conv}} < {rbin['rmax']:.0f}\ \mathrm{{cm}}$"
            ax.legend(title=range_str)
            ax.grid(True)
            add_cms_label(ax)

    fig.tight_layout()
    fig.savefig("t0_resolution.png")
    fig.savefig("t0_resolution.pdf")
    plt.close(fig)

    # 16 & 17. debugging
    for (rho_min, rho_max), d_dbg in data["debugging"].items():
        fig, axs = plt.subplots(7, 3, figsize=(36, 84))

        # [0, 0]
        ax = axs[0, 0]
        c1, b1 = np.histogram(d_dbg["arr_tmtd"], bins=100, range=(3, 8))
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_tmtd_awful"], bins=100, range=(3, 8))
        hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        c3, b3 = np.histogram(d_dbg["arr_tmtd_far"], bins=100, range=(3, 8))
        hep.histplot(c3, b3, ax=ax, histtype="step", color=cms_palette[2], label=r"+ $d(\mathrm{conv}, \mathrm{MTD}) > 3\ \mathrm{cm}$", linewidth=2)
        ax.set_xlabel(r"$t_\mathrm{MTD}\ [\mathrm{ns}]$")
        ax.set_ylabel("Entries")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [0, 1]
        ax = axs[0, 1]
        c1, b1 = np.histogram(d_dbg["arr_tof"], bins=100, range=(3, 8))
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        bin_centers = 0.5 * (b1[:-1] + b1[1:])
        bin_widths = b1[1:] - b1[:-1]
        mask_4 = bin_centers < 4
        ax.bar(bin_centers[mask_4], c1[mask_4], width=bin_widths[mask_4], align="center", alpha=0.5, color=cms_palette[0], label=r"TOF $< 4\ \mathrm{ns}$")
        c2, b2 = np.histogram(d_dbg["arr_tof_awful"], bins=100, range=(3, 8))
        hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        c3, b3 = np.histogram(d_dbg["arr_tof_far"], bins=100, range=(3, 8))
        hep.histplot(c3, b3, ax=ax, histtype="step", color=cms_palette[2], label=r"+ $d(\mathrm{conv}, \mathrm{MTD}) > 3\ \mathrm{cm}$", linewidth=2)
        ax.set_xlabel(r"$\overline{\mathrm{TOF}}\ [\mathrm{ns}]$")
        ax.set_ylabel("Entries")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [0, 2]
        ax = axs[0, 2]
        c1, b1 = np.histogram(d_dbg["arr_genPV_t"], bins=100)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_genPV_awful"], bins=100)
        hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$t_\mathrm{PV}\ [\mathrm{ns}]$")
        ax.set_ylabel("Entries")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [1, 0]
        ax = axs[1, 0]
        h = ax.hist2d(d_dbg["arr_tmtd"], d_dbg["arr_tof"], bins=100, range=((3, 8), (3, 8)), cmap="viridis")
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(r"$t_\mathrm{MTD}\ [\mathrm{ns}]$")
        ax.set_ylabel(r"$\overline{\mathrm{TOF}}\ [\mathrm{ns}]$")
        add_cms_label(ax)

        # [1, 1]
        ax = axs[1, 1]
        c1, b1 = np.histogram(d_dbg["arr_res"], bins=30, range=(-0.1, 1))
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_suspect"], bins=30, range=(-0.1, 1))
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ TOF $< 4\ \mathrm{ns}$", linewidth=2)
        c3, b3 = np.histogram(d_dbg["arr_far_res"], bins=30, range=(-0.1, 1))
        hep.histplot(c3, b3, ax=ax, histtype="step", color=cms_palette[2], label=r"+ $d(\mathrm{conv}, \mathrm{MTD}) > 3\ \mathrm{cm}$", linewidth=2)
        ax.set_xlabel(r"$\Delta t\ [\mathrm{ns}]$")
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [1, 2]
        ax = axs[1, 2]
        arr_diff = d_dbg["arr_tmtd"] - d_dbg["arr_tof"]
        c1, b1 = np.histogram(arr_diff, bins=100, range=(-0.6, 0.6))
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        arr_diff_awful = d_dbg["arr_tmtd_awful"] - d_dbg["arr_tof_awful"]
        c2, b2 = np.histogram(arr_diff_awful, bins=100, range=(-0.6, 0.6))
        hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$t_\mathrm{MTD} - \overline{\mathrm{TOF}}\ [\mathrm{ns}]$")
        ax.set_ylabel("Entries")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [2, 0]
        ax = axs[2, 0]
        c1, b1 = np.histogram(d_dbg["arr_x_diff"], bins=50, density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_x_sus"], bins=50, density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$x_\mathrm{MTD} - x_\mathrm{PV}\ [\mathrm{cm}]$")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [2, 1]
        ax = axs[2, 1]
        c1, b1 = np.histogram(d_dbg["arr_y_diff"], bins=50, density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_y_sus"], bins=50, density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$y_\mathrm{MTD} - y_\mathrm{PV}\ [\mathrm{cm}]$")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [2, 2]
        ax = axs[2, 2]
        c1, b1 = np.histogram(d_dbg["arr_z_diff"], bins=50, range=[-200, 200], density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_z_sus"], bins=50, density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$z_\mathrm{MTD} - z_\mathrm{PV}\ [\mathrm{cm}]$")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [3, 0]
        ax = axs[3, 0]
        c1, b1 = np.histogram(d_dbg["arr_eta_m"], bins=50, density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_eta_sus"], bins=50, density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\eta_\mathrm{MTD}$")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [3, 1]
        ax = axs[3, 1]
        c1, b1 = np.histogram(d_dbg["arr_rho_m"], bins=50, range=(115.4, 116.2), density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_rho_sus"], bins=50, range=(115.4, 116.2), density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\rho_\mathrm{MTD}\ [\mathrm{cm}]$")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [3, 2]
        ax = axs[3, 2]
        c1, b1 = np.histogram(d_dbg["arr_phi_m"], bins=50, range=(-np.pi, np.pi), density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_phi_sus"], bins=50, range=(-np.pi, np.pi), density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\phi_\mathrm{MTD}\ [\mathrm{rad}]$")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [4, 0]
        ax = axs[4, 0]
        plot_rng = (0, 60) if rho_max == 110 else (0, 20)
        c1, b1 = np.histogram(d_dbg["arr_dist"], bins=50, range=plot_rng, density=True)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_dist_awful"], bins=50, range=plot_rng, density=True)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        if rho_min < 115 and rho_max > 115:
            ax.axvline(3, color=cms_palette[2], linestyle="dashed", label=r"$d(\mathrm{conv}, \mathrm{MTD}) = 3\ \mathrm{cm}$")
        ax.set_xlabel(r"$d(\mathrm{conv}, \mathrm{MTD})\ [\mathrm{cm}]$")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [4, 1] MTD ring + 4 awful events
        ax = axs[4, 1]
        mtd_ring = mpl.patches.Annulus((0, 0), 116.2, 116.2 - 115.4, color=cms_palette[0], alpha=0.3, label="MTD")
        ax.add_patch(mtd_ring)
        for i_geo, geo in enumerate(d_dbg["awful_events_geo"]):
            ax.plot(geo["conv_x"], geo["conv_y"], "x", color=cms_palette[(i_geo + 1) % len(cms_palette)], label=f"Event {geo['evt']}")
            ax.plot(geo["mtd_x"], geo["mtd_y"], "o", color=cms_palette[(i_geo + 1) % len(cms_palette)])
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(r"$x\ [\mathrm{cm}]$")
        ax.set_ylabel(r"$y\ [\mathrm{cm}]$")
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        add_cms_label(ax)

        # [4, 2] z, rho view
        ax = axs[4, 2]
        mtd_rect = mpl.patches.Rectangle((-200, 115.4), 400, 0.8, color=cms_palette[0], alpha=0.3, label="MTD")
        ax.add_patch(mtd_rect)
        for i_geo, geo in enumerate(d_dbg["awful_events_geo"]):
            ax.plot(geo["conv_z"], geo["conv_r"], "x", color=cms_palette[(i_geo + 1) % len(cms_palette)], label=f"Event {geo['evt']}")
            ax.plot(geo["mtd_z"], geo["mtd_r"], "o", color=cms_palette[(i_geo + 1) % len(cms_palette)])
        ax.grid(True)
        ax.legend()
        ax.set_xlabel(r"$z\ [\mathrm{cm}]$")
        ax.set_ylabel(r"$R\ [\mathrm{cm}]$")
        ax.set_xlim(-200, 200)
        ax.set_ylim(115, 116.8)
        add_cms_label(ax)

        # [5, 0] deltaPhi conv
        ax = axs[5, 0]
        c1, b1 = np.histogram(d_dbg["arr_dphi_conv"], bins=50)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_dphi_conv_awful"], bins=50)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\Delta\phi(\mathrm{conv}, \mathrm{MTD})$")
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [5, 1] deltaR conv
        ax = axs[5, 1]
        c1, b1 = np.histogram(d_dbg["arr_dr_conv"], bins=50)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_dr_conv_awful"], bins=50)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\Delta R(\mathrm{conv}, \mathrm{MTD})\ [\mathrm{cm}]$")
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [5, 2] deltaZ conv
        ax = axs[5, 2]
        c1, b1 = np.histogram(d_dbg["arr_dz_conv"], bins=50, range=(0, 20))
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_dz_conv_awful"], bins=50, range=(0, 20))
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\Delta z(\mathrm{conv}, \mathrm{MTD})\ [\mathrm{cm}]$")
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # [6, 0] deltaEta conv
        ax = axs[6, 0]
        c1, b1 = np.histogram(d_dbg["arr_deta_conv"], bins=50)
        hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label=r"$|\Delta\phi| < 0.05$", linewidth=2)
        c2, b2 = np.histogram(d_dbg["arr_deta_conv_awful"], bins=50)
        hep.histplot(c2, b2, ax=ax, histtype="step", linestyle="dashed", color=cms_palette[1], alpha=0.8, label=r"+ $\Delta\mathrm{TOF} > 0.1\ \mathrm{ns}$", linewidth=2)
        ax.set_xlabel(r"$\Delta\eta(\mathrm{conv}, \mathrm{MTD})$")
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        add_cms_label(ax)

        # hide remaining subplots [6, 1], [6, 2]
        axs[6, 1].axis("off")
        axs[6, 2].axis("off")

        fig.tight_layout()
        fig.savefig(f"debugging_{rho_min}To{rho_max}.png")
        fig.savefig(f"debugging_{rho_min}To{rho_max}.pdf")
        plt.close(fig)

    # 18. conversion_rate
    fig, axs = plt.subplots(2, 2, figsize=(14, 14), layout="constrained")
    cr = data["conversion_rate"]

    for ax in axs.flatten():
        ax.tick_params(labelsize=16)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(0, 1)

    ax = axs[0, 0]
    plot_tefficiency_styled(cr["pt"], ax=ax, color=cms_palette[0], label="All conversions")
    plot_tefficiency_styled(cr["pt_mtd"], ax=ax, color=cms_palette[1], label="MTD conversions")
    ax.set_xlabel(r"$p_\mathrm{T}\ [\mathrm{GeV}]$", fontsize=18)
    ax.set_ylabel("Conversion efficiency", fontsize=18)
    ax.legend(fontsize=14, loc="upper right")
    add_cms_label(ax, fontsize=16)

    ax = axs[0, 1]
    plot_tefficiency_styled(cr["eta"], ax=ax, color=cms_palette[0], label="All conversions")
    plot_tefficiency_styled(cr["eta_mtd"], ax=ax, color=cms_palette[1], label="MTD conversions")
    ax.set_xlabel(r"$|\eta|$", fontsize=18)
    ax.set_ylabel("Conversion efficiency", fontsize=18)
    ax.legend(fontsize=14, loc="upper right")
    add_cms_label(ax, fontsize=16)

    ax = axs[1, 0]
    plot_tefficiency_styled(cr["pt_old"], ax=ax, color=cms_palette[2], label=r"Old: $R_\mathrm{conv} < 118\ \mathrm{cm},\ p_\mathrm{T} > 1\ \mathrm{GeV}$")
    ax.set_xlabel(r"$p_\mathrm{T}\ [\mathrm{GeV}]$", fontsize=18)
    ax.set_ylabel("Conversion efficiency", fontsize=18)
    ax.legend(fontsize=13, loc="upper right")
    add_cms_label(ax, fontsize=16)

    ax = axs[1, 1]
    plot_tefficiency_styled(cr["eta_old"], ax=ax, color=cms_palette[2], label=r"Old: $R_\mathrm{conv} < 118\ \mathrm{cm},\ p_\mathrm{T} > 1\ \mathrm{GeV}$")
    ax.set_xlabel(r"$|\eta|$", fontsize=18)
    ax.set_ylabel("Conversion efficiency", fontsize=18)
    ax.legend(fontsize=13, loc="upper right")
    add_cms_label(ax, fontsize=16)

    fig.savefig("conversion_rate.png")
    fig.savefig("conversion_rate.pdf")
    plt.close(fig)

    # 19. mtd_hit_matching
    fig, axs = plt.subplots(2, 2, figsize=(24, 20))
    hm = data["hit_matching"]

    ax = axs[0, 0]
    c1, b1 = np.histogram(hm["drMatch_all"], bins=100, range=[0, 0.5])
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="All", linewidth=2)
    c2, b2 = np.histogram(hm["drMatch_trk"], bins=100, range=[0, 0.5])
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label=r"Tracker conversions ($\rho < 114\ \mathrm{cm}$)", linewidth=2)
    c3, b3 = np.histogram(hm["drMatch_mtd"], bins=100, range=[0, 0.5])
    hep.histplot(c3, b3, ax=ax, histtype="step", color=cms_palette[2], label=r"MTD conversions ($\rho \geq 114\ \mathrm{cm}$)", linewidth=2)
    ax.set_xlabel(r"$\Delta R(\gamma, \mathrm{hit})$")
    ax.set_ylabel("Entries")
    ax.legend()
    ax.grid(True)
    add_cms_label(ax)

    ax = axs[0, 1]
    c1, b1 = np.histogram(hm["res1"], bins=100, range=[-0.5, 0.5])
    hep.histplot(c1, b1, ax=ax, histtype="step", color=cms_palette[0], label="Res. wrt leading", linewidth=2)
    c2, b2 = np.histogram(hm["res2"], bins=100, range=[-0.5, 0.5])
    hep.histplot(c2, b2, ax=ax, histtype="step", color=cms_palette[1], label="Res. wrt subleading", linewidth=2)
    ax.set_xlabel(r"$\Delta t(\mathrm{RECO}, \mathrm{SIM})\ [\mathrm{ns}]$")
    ax.set_ylabel("Entries")
    ax.legend()
    ax.grid(True)
    add_cms_label(ax)

    axs[1, 0].axis("off")
    axs[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig("mtd_hit_matching.png")
    fig.savefig("mtd_hit_matching.pdf")
    plt.close(fig)

    plt.close("all")

    # Copy plots to EOS area
    outfolder = "0p1To10/replot" if args.use_low_pt else "8To150"
    eosfolder = f"/eos/home-n/npalmeri/www/MTD/PhotonReco/GEN_level/{outfolder}"
    print(f">>> Copying plots and logs to {eosfolder}...")
    os.system(f"cp *.png {eosfolder}")
    os.system(f"cp *.pdf {eosfolder}")
    os.system(f"cp conversions.log {eosfolder}")
    print(">>> Finished copying to EOS.")


# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_low_pt", help="Use low-pT sample (0.1 to 10 GeV) or high pT (8 to 150 GeV) for single photon gun", action="store_true", default=True)
    parser.add_argument("--recompute", help="Force recomputing processed data and rebuilding cache", action="store_true", default=False)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_tag = "low_pt" if args.use_low_pt else "high_pt"
    cache_file = os.path.join(script_dir, f"conversion_cache_{sample_tag}.pkl")

    if (not args.recompute) and os.path.exists(cache_file):
        print(f">>> Loading cached processed data from {cache_file}...")
        with open(cache_file, "rb") as f_in:
            data = pickle.load(f_in)
        print(f">>> Cache loaded successfully!")
    else:
        if args.recompute:
            print(">>> --recompute requested. Bypassing cache...")
        else:
            print(f">>> Cache file {cache_file} not found. Running data processing...")
        data = process_data(args)
        print(f">>> Saving processed data to cache file {cache_file}...")
        with open(cache_file, "wb") as f_out:
            pickle.dump(data, f_out)
        print(">>> Cached successfully!")

    make_plots(data, args)
    print(">>> All tasks completed!")
