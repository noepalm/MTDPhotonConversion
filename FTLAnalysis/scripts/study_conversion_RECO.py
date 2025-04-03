from plot_libraries import *
import os
import argparse
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import curve_fit

NanoAODSchema.warn_missing_crossrefs = False

# --- UTILS ----
def ak_seconds(arr, axis = 1):
    idxs = ak.ones_like(arr)
    tmp = arr[ak.local_index(arr) == idxs]
    return ak.firsts(tmp, axis = axis)

def angle_to_mpi_pi(phi):
    while ak.sum(phi >= np.pi):
        phi = ak.where(phi >= np.pi, phi - np.pi*2, phi)
    while ak.sum(phi < -np.pi):
        phi = ak.where(phi < -np.pi, phi + np.pi *2, phi)
    return phi

def printlog(mystr, f):
    print(mystr)
    f.write(mystr)
    f.write("\n")

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 19
plt.rcParams['ytick.labelsize'] = 19

# --------------

# OPTIONS
parser = argparse.ArgumentParser()
# TODO: CHANGE DEFAULT WHEN RUNNING ON HIGH PT SAMPLE TOO
parser.add_argument("--use_low_pt", help="Use low-pT sample (0.1 to 10 GeV) or high pT (8 to 150 GeV) for single photon gun", action="store_true", default=True)
args = parser.parse_args()

# create log file
os.system("touch conversions.log")
f = open("conversions.log", "w")

# Load data
if args.use_low_pt:
    # fname = "/eos/home-n/npalmeri/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/lowPtSample.root"
    fname = "/eos/home-n/npalmeri/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/test.root"
else:
    fname = "/eos/home-n/npalmeri/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/highPtSample.root"

# Create output plot folder (if it does not exist)
local_folder = "plots/RECO"
# if args.use_low_pt:
#     local_folder = f"{local_folder}_0p1To10"
# else:
#     local_folder = f"{local_folder}_8To150"

os.system("mkdir -p %s" % local_folder)

# Awkward Arrays using NANOAOD schema
evts = NanoEventsFactory.from_root(fname,
                                   treepath="neu_tree",
                                   schemaclass = NanoAODSchema).events()

# GEN INFO
conversion_mask_photon = evts.GENConvertedPhoton.convRadius < 116.1 # photon decay vertex has occured before or at MTD
conversion_mask_photon = conversion_mask_photon * (evts.GENConvertedPhoton.nlegs >= 2) # photon has converted

# NB: already pT ordered in ntuplizer
ele1 = evts.GENElectron[ak.firsts(evts.GENConvertedPhoton.eleIdxs, axis = 2)]
ele2 = evts.GENElectron[ak_seconds(evts.GENConvertedPhoton.eleIdxs, axis = 2)]

conv_photons = evts.GENConvertedPhoton[conversion_mask_photon]
ele1 = ele1[conversion_mask_photon]
ele2 = ele2[conversion_mask_photon]

mtd_hit = evts.MTDHit[conversion_mask_photon]
mtd_hit_sim = evts.MTDHitSim[conversion_mask_photon]

# event-level info
genPV = evts.genPV

# RECO info

#TOFILL

# --------------------------- #

# PF Photon kinematics

### statistics on PF photon matching

all_matched_mask = ak.sum(evts.Photon.drMatch < 1e4, axis = 1) == ak.num(evts.Photon, axis = 1)

ak.num(evts.Photon, axis = 1) == ak.num(evts.GENConvertedPhoton, axis = 1)
printlog(f"Events with all gen photons matched: {ak.sum(all_matched_mask)}", f)

# for these events, plot kinematics and resolutions
vars = ["pt", "eta", "phi"]
nrows = 3
ncols = len(vars)

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 12, nrows * 12))

for i in range(2):
    for var, ax in zip(vars, axs[i]):
        arr = evts.Photon[var][all_matched_mask]
        if var == "pt":
            arr = arr * 10
        if i == 1:
            arr = arr - evts.GENConvertedPhoton[var][all_matched_mask]
        if var == "phi":
            arr = angle_to_mpi_pi(arr)
        ax.hist(ak.ravel(arr).to_numpy(), bins=100, histtype='step', label='RECO Photon')
        ax.set_xlabel(f'{var} [GeV]')
        ax.set_ylabel('Entries')
        title = f"{var} RECO - GEN resolution" if i == 1 else f'{var} RECO distribution'
        ax.set_title(title, pad = 40)
        ax.legend()
        ax.grid()

# plot deltaR, deltaPtOverPt distributions
ax = axs[1, 0]
arr = evts.Photon.drMatch
ax.hist(ak.ravel(arr).to_numpy(), bins=100, histtype='step', label='RECO Photon')
ax.set_xlabel(f'Match $\Delta R$')
ax.set_ylabel('Entries')
ax.legend()
ax.grid()

ax = axs[1, 1]
arr = (evts.Photon.pt[all_matched_mask] - evts.GENConvertedPhoton.pt[all_matched_mask]) / evts.GENConvertedPhoton.pt[all_matched_mask]
ax.hist(ak.ravel(arr).to_numpy(), bins=100, histtype='step', label='RECO Photon')
ax.set_xlabel(f'$\Delta p_T / p_T(GEN)$')
ax.set_ylabel('Entries')
ax.legend()
ax.grid()

fig.savefig(f"{local_folder}/photon_kinematics.png")
fig.savefig(f"{local_folder}/photon_kinematics.pdf")

### statistics on MTD reco hit matching

# note: one MTD reco hit per photon, but multiple sim cluster times per electron
# ==> take the first, direct hit for each electron to define a single SIM MTD time per photon

simtime_1 = ak.firsts(ele1.tClus[ele1.typeClus == 0], axis = 2)
simtime_2 = ak.firsts(ele2.tClus[ele2.typeClus == 0], axis = 2)

recotime = evts.MTDHit[conversion_mask_photon].time

nrows = 2
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 12, nrows * 12))

ax = axs[0, 0]
# plot distirbution of reco time
ax.hist(ak.ravel(recotime).to_numpy(), bins=100, range = [3, 11], histtype='step', label='RECO MTD hit')
ax.set_xlabel(f'MTD reco hit time [ns]')
ax.set_ylabel('Entries')
ax.legend()
ax.grid()

ax = axs[0, 1]
# plot difference wrt sim time 1
ax.hist(ak.ravel(recotime - simtime_1).to_numpy(), bins=100, range = [-2, 2], histtype='step', label=r"$t_\text{reco} - t_\text{sim}(1)$")
ax.hist(ak.ravel(recotime - simtime_2).to_numpy(), bins=100, range = [-2, 2], histtype='step', label=r"$t_\text{reco} - t_\text{sim}(2)$")
ax.set_xlabel(f'MTD time resolution (RECO - SIM) [ns]')
ax.set_ylabel('Entries')
ax.legend()
ax.grid()
ax.set_yscale("log")

fig.savefig(f"{local_folder}/mtd_time_resolution.png")
fig.savefig(f"{local_folder}/mtd_time_resolution.pdf")

# --------------------------- #

# close all figs
plt.close("all")

# copy plots to EOS area
outfolder = "0p1To10" if args.use_low_pt else "8To150"
eosfolder = f"/eos/home-n/npalmeri/www/MTD/PhotonReco/RECO/{outfolder}"
os.system(f"cp {local_folder}/*.png {eosfolder}")
os.system(f"cp {local_folder}/*.pdf {eosfolder}")

# also copy log
f.close()
os.system(f"cp conversions.log {eosfolder}")