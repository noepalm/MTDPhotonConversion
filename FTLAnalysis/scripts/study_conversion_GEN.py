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

# create local plot output folder
local_folder = "plots/GEN"
os.system("mkdir -p %s" % local_folder)

# Load data
if args.use_low_pt:
    fname = "/eos/home-n/npalmeri/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/lowPtSample.root"
    # fname = "/eos/home-n/npalmeri/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/test.root"
else:
    fname = "/eos/home-n/npalmeri/MTD_photonReco/CMSSW_15_0_0_pre3/src/PrecisionTiming/FTLAnalysis/test/highPtSample.root"

# Awkward Arrays using NANOAOD schema
evts = NanoEventsFactory.from_root(fname,
                                   treepath="neu_tree",
                                   schemaclass = NanoAODSchema).events()

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

# First approximation: consider only conversions with direct hits for both electrons
good_cluster_photons = (ak.sum(ele1.typeClus == 0, axis = 2) >= 1) * (ak.sum(ele2.typeClus == 0, axis = 2) >= 1)
good_clusters_1 = (ele1[good_cluster_photons].typeClus == 0)
good_clusters_2 = (ele2[good_cluster_photons].typeClus == 0)
## CHECK: deta cut
# good_cluster_photons = (ak.sum((ele1.typeClus == 0) * (abs(deta1_photon_all) < 0.02), axis = 2) >= 1) * (ak.sum((ele2.typeClus == 0) * (abs(deta2_photon_all) < 0.02), axis = 2) >= 1)
# good_clusters_1 = (ele1[good_cluster_photons].typeClus == 0) * (abs(deta1_photon_all[good_cluster_photons]) < 0.02)
# good_clusters_2 = (ele2[good_cluster_photons].typeClus == 0) * (abs(deta2_photon_all[good_cluster_photons]) < 0.02)
selected1 = lambda x: ak.firsts(x[good_cluster_photons][good_clusters_1], axis = 2)
selected2 = lambda x: ak.firsts(x[good_cluster_photons][good_clusters_2], axis = 2)

### MC validation
# plot PHOTON kinematics
vars = ["pt", "eta", "phi"]
nrows = 1
ncols = len(vars)
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

for var, ax in zip(vars, axs):
    plot_hist_from_array(conv_photons[var], ax = ax, bins = 50)
    ax.set_xlabel(var)
    ax.set_ylabel("Entries")
    ax.set_title(f"Photon {var}", pad = 40)

fig.savefig(f"{local_folder}/kinematics_photon.png")
fig.savefig(f"{local_folder}/kinematics_photon.pdf")

# plot ELECTRON kinematics
nrows = 1
ncols = len(vars)
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

for var, ax in zip(vars, axs):
    plot_hist_from_array(ele1[var], ax = ax, bins = 50, label = "Leading")
    plot_hist_from_array(ele2[var], ax = ax, bins = 50, label = "Subleading")
    ax.legend()
    ax.set_xlabel(var)
    ax.set_ylabel("Entries")
    ax.set_title(f"Electron {var}", pad = 40)

fig.savefig(f"{local_folder}/kinematics_electron.png")
fig.savefig(f"{local_folder}/kinematics_electron.pdf")

# 1. scatter plot of conversion vertex position
### a. xy plane from rho, phi
zConv = conv_photons.convZ
rho = conv_photons.convRadius
phi = conv_photons.convPhi
eta = np.arcsinh(zConv / rho)
x = rho*np.cos(phi)
y = rho*np.sin(phi)

# first, print fraction of conversions occurring in tracker vs MTD
n_tracker = ak.sum(rho < 115)
n_mtd = ak.sum(rho >= 115)
n_total = ak.sum(ak.num(rho))

# print to log file
printlog(f"Fraction of conversions in tracker = {n_tracker / n_total * 100:.2f}% ({n_tracker} out of {n_total})", f)
printlog(f"Fraction of conversions in MTD = {n_mtd / n_total * 100:.2f}% ({n_mtd} out of {n_total})", f)
printlog(f"Fraction of conversions with direct hits for both electrons = {ak.sum(good_cluster_photons) / ak.sum(conversion_mask_photon) * 100 : .3f}% ({ak.sum(good_cluster_photons)} over {ak.sum(conversion_mask_photon)})", f)

fig, ax = plt.subplots()
ax.scatter(ak.flatten(x), ak.flatten(y), s=1)
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_title("Conversion vertex position in xy plane", pad = 40)
fig.savefig(f"{local_folder}/conversion_vertex_xy.png")
fig.savefig(f"{local_folder}/conversion_vertex_xy.pdf")

# do the same but with 2D hist
fig, ax = plt.subplots(figsize = (10, 8))
h = ax.hist2d(ak.flatten(x), ak.flatten(y), bins = 30)
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
fig.colorbar(h[3], ax = ax)
ax.set_title("Conversion vertex position in xy plane", pad = 40)
fig.savefig(f"{local_folder}/conversion_vertex_xy_hist.png")
fig.savefig(f"{local_folder}/conversion_vertex_xy_hist.pdf")

### b. z, rho plane
z = conv_photons.convZ

fig, ax = plt.subplots()
ax.scatter(ak.flatten(rho), ak.flatten(z), s=1)
ax.set_xlabel("rho [cm]")
ax.set_ylabel("z [cm]")
ax.set_xlim(0, 120)
ax.set_ylim(-200, 200)
ax.set_title("Conversion vertex position in z, rho plane", pad = 40)
fig.savefig(f"{local_folder}/conversion_vertex_z_rho.png")
fig.savefig(f"{local_folder}/conversion_vertex_z_rho.pdf")

# do the same but with 2D hist
fig, ax = plt.subplots(figsize = (10, 8))
h = ax.hist2d(ak.flatten(rho).to_numpy(), ak.flatten(z).to_numpy(), bins = 30, range = [[0, 120], [-200, 200]])
ax.set_xlabel("rho [cm]")
ax.set_ylabel("z [cm]")
fig.colorbar(h[3], ax = ax)
ax.set_title("Conversion vertex position in z, rho plane", pad = 40)
fig.savefig(f"{local_folder}/conversion_vertex_z_rho_hist.png")
fig.savefig(f"{local_folder}/conversion_vertex_z_rho_hist.pdf")

# 2. scatter plot of MTD position
tmtd_1 = ele1.tClus
xmtd_1 = ele1.xClus
ymtd_1 = ele1.yClus
zmtd_1 = ele1.zClus
tmtd_2 = ele2.tClus
xmtd_2 = ele2.xClus
ymtd_2 = ele2.yClus
zmtd_2 = ele2.zClus

ncols = 2
nrows = 1
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

ax = axs[0]
ax.scatter(ak.flatten(xmtd_1, axis = None), ak.flatten(ymtd_1, axis = None), s=1, label = "Lead")
ax.scatter(ak.flatten(xmtd_2, axis = None), ak.flatten(ymtd_2, axis = None), s=1, label = "Sublead")

# plot direct hit position only
ax.scatter(ak.flatten(xmtd_1[ele1.typeClus == 0], axis = None), ak.flatten(ymtd_1[ele1.typeClus == 0], axis = None), s=1, label = "ele1 direct")
ax.scatter(ak.flatten(xmtd_2[ele2.typeClus == 0], axis = None), ak.flatten(ymtd_2[ele2.typeClus == 0], axis = None), s=1, label = "ele2 direct")

ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_title("MTD position", pad = 40)
ax.legend()
ax.grid()

ax = axs[1]

mtd_r = np.sqrt(xmtd_1**2 + ymtd_1**2)
mtd_r = mtd_r[(mtd_r > 100) * (mtd_r < 120)]
# print to log file
printlog(f"Average MTD radius = {ak.mean(mtd_r):.6g} cm (+- 1 std = {ak.std(mtd_r):.3g} cm)", f)

ax.hist(ak.ravel(mtd_r), bins = 50, histtype = "step")
ax.set_xlabel("r [cm]")
ax.set_ylabel("Entries")
ax.set_title("MTD cluster radius distribution", pad = 40)
ax.grid()

fig.savefig(f"{local_folder}/mtd_cluster_positions.png")
fig.savefig(f"{local_folder}/mtd_cluster_positions.pdf")

# -------- GEN INFO ------#

# scatter plot of gen vertex position in xy
nrows = 2
ncols = 3
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

# coordinate histograms
var_names = {0 : "x", 1 : "y", 2 : "z", 3 : "t"}
for i, var in enumerate(var_names.values()):
    ax = axs.flatten()[i]
    ax.hist(genPV[var], bins = 50, histtype = "step", label = var)
    unit = "cm" if var != "t" else "ns"
    ax.set_xlabel(f"{var} [{unit}]")
    ax.set_ylabel("Entries")
    ax.set_title(f"GenPV {var} position", pad = 40)
    ax.legend()

# x vs y
ax = axs[1, 1]
ax.scatter(genPV.x, genPV.y, s=1, alpha = 0.5)
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_title("GenPV position in xy plane", pad = 40)
ax.grid()

fig.savefig(f"{local_folder}/genPV.png")
fig.savefig(f"{local_folder}/genPV.pdf")

# -------- MTD INFO PLOTS -------- #
nrows = 8
ncols = 4

fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

# MTD time
ax = axs[0, 0]
ax.hist(ak.concatenate([ak.ravel(tmtd_1), ak.ravel(tmtd_2)]), bins = 100, range = [2.5, 25], histtype = "bar", color = "gray", alpha = 0.3, label = "All clusters")
# plot each cluster type separately
type_dict = {0 : "Direct", 1 : "Secondary", 2 : "Looper", 3 : "Calo backscatter"}
for i in range(4):
    ax.hist(ak.concatenate([ak.ravel(tmtd_1[ele1.typeClus == i]), ak.ravel(tmtd_2[ele2.typeClus == i])]).to_numpy(), bins = 100, range = [2.5, 25], histtype = "step", label = type_dict[i])
ax.set_xlabel("t [ns]")
ax.set_ylabel("Entries")
ax.set_xlim(2.5, 25)
ax.set_title("MTD time of ALL clusters for conversion events", pad = 40)
ax.legend()

### same but only for events with 2 direct hits
ax = axs[1, 0]

# All conversions with 2 direct hits
ax.hist(ak.concatenate([ak.ravel(selected1(tmtd_1)), ak.ravel(selected2(tmtd_2))]), bins = 100, histtype = "bar", color = "gray", alpha = 0.3, label = "All photons")

# Converisons with 2 direct hits from LEADING photon only
is_leading_photon = ak.local_index(conv_photons.pt) == 0
mask_leading_conv_photon = is_leading_photon * good_cluster_photons
mask_leading_timeClus_1 = ele1.typeClus[mask_leading_conv_photon] == 0
time1 = tmtd_1[mask_leading_conv_photon][mask_leading_timeClus_1]

mask_leading_timeClus_2 = ele2.typeClus[mask_leading_conv_photon] == 0
time2 = tmtd_2[mask_leading_conv_photon][mask_leading_timeClus_2]

arrx = ak.concatenate([ak.ravel(time1), ak.ravel(time2)])
ax.hist(arrx, bins = 100, histtype = "bar", alpha = 0.3, label = "Leading photons")

ax.set_xlabel("t [ns]")
ax.set_ylabel("Entries")
ax.set_title("MTD time of direct clusters for conversions with 2 direct hits", pad = 40)
ax.legend()

# MTD time vs conversion radius
ax = axs[0, 1]
rho_range = np.linspace(0, 120, 5)
for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:]):
    mask = (rho >= rho_min) * (rho < rho_max)
    ax.hist(ak.ravel(tmtd_1[mask][ele1[mask].typeClus == 0]), label = f"{rho_min} < Rconv < {rho_max} cm", histtype = "step", bins = 50, density = True, range = [2.5, 25])
ax.set_xlabel("t [ns]")
ax.set_ylabel("Density")
ax.set_title("MTD time of directs vs conv. radius", pad = 40)
ax.legend()

### same, but only selected converted photons where BOTH electrons have a direct cluster
# MTD time vs conversion radius
ax = axs[1, 1]
rho_range = np.linspace(0, 120, 5)
for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:]):
    mask = (rho >= rho_min) * (rho < rho_max)
    mask1 = mask * good_cluster_photons
    mask2 = ele1.typeClus[mask1] == 0
    ax.hist(ak.ravel(ak.firsts(tmtd_1[mask1][mask2], axis = 2)), label = f"{rho_min} < Rconv < {rho_max} cm", histtype = "step", bins = 50, density = True, range = [2.5, 25])
ax.set_xlabel("t [ns]")
ax.set_ylabel("Density")
ax.set_title(f"MTD time of directs (for $\gamma$ w/ 2 DIRECT) vs conv. radius", pad = 40)
ax.legend()

# Time broken down by photon eta
ax = axs[1, 2]
mask_loweta = good_cluster_photons * (eta < 0.6)
mask1_loweta = mask_loweta * (ele1.typeClus == 0)
mask2_loweta = mask_loweta * (ele2.typeClus == 0)
ax.hist(ak.ravel(ak.firsts(tmtd_1[mask1_loweta], axis = 2)), range = [2.5, 8], bins = 50, histtype = "step", label = r"Lead, $|\eta| < 0.6$")
# ax.hist(ak.ravel(ak.firsts(tmtd_2[mask2_loweta], axis = 2)), bins = 100, histtype = "step", label = r"Sublead, $|\eta| < 0.6$")
mask_higheta = good_cluster_photons * (eta >= 0.6)
mask1_higheta = mask_higheta * (ele1.typeClus == 0)
mask2_higheta = mask_higheta * (ele2.typeClus == 0)
ax.hist(ak.ravel(ak.firsts(tmtd_1[mask1_higheta], axis = 2)), range = [2.5, 8], bins = 50, histtype = "step", label = r"Lead, $|\eta| \geq 0.6$")
# ax.hist(ak.ravel(ak.firsts(tmtd_2[mask2_higheta], axis = 2)), bins = 100, histtype = "step", label = r"Sublead, $|\eta| > 0.6$")
ax.set_xlabel("t [ns]")
ax.set_ylabel("Entries")
ax.set_title(r"MTD time of directs for conv. with 2 direct hits vs. conversion $\eta$", pad = 40)
ax.legend()

# TIME vs. ENERGY
ax = axs[1, 3]
for i in range(4)[::-1]:
    arrx1 = ak.ravel(ele1.tClus[ele1.typeClus == i])
    arrx2 = ak.ravel(ele2.tClus[ele2.typeClus == i])
    arrx = ak.concatenate([arrx1, arrx2])
    arry1 = ak.ravel(ele1.energyClus[ele1.typeClus == i])
    arry2 = ak.ravel(ele2.energyClus[ele2.typeClus == i])
    arry = ak.concatenate([arry1, arry2])
    ax.scatter(arrx, arry, s = 1, alpha = 0.5, label = type_dict[i], color = f"C{i}")
ax.set_xlabel("t [ns]")
ax.set_ylabel("E [MeV]")
ax.set_xlim(4, 10)
ax.set_ylim(0, 10)
ax.set_title("MTD time vs energy", pad = 40)
ax.legend(markerscale = 10)
ax.grid()

# cluster type distribution
# (can be 0 = direct, 1 = secondary, 2 = looper, 3 = calo backscatter)
ax = axs[0, 2]
ax.hist(ak.concatenate([ak.ravel(ele2.typeClus), ak.ravel(ele1.typeClus)]), bins = 4, range = [-0.5, 3.5], histtype = "step", label = "All conversions")
# only for MTD conversions
ax.hist(ak.concatenate([ak.ravel(ele2.typeClus[rho > 115]), ak.ravel(ele1.typeClus[rho > 115])]), bins = 4, range = [-0.5, 3.5], histtype = "step", label = "MTD conversions")
ax.set_xlabel("Cluster type")
ax.set_ylabel("Entries")
ax.set_title("MTD cluster type", pad = 40)
ax.legend()

# change tick labels
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["Direct", "Secondary", "Looper", "Calo backscatter"])

# --- ENERGY, SIZE, ETC ---
ax = axs[2, 0]
plot_range = (0, 10)
# plot energy distribution for each cluster type
ax.hist(ak.concatenate([ak.ravel(ele1.energyClus), ak.ravel(ele2.energyClus)]).to_numpy(), bins = 50, histtype = "bar", range = plot_range, label = "All", color = "gray", alpha = 0.3)
for i in range(4):
    ax.hist(ak.concatenate([ak.ravel(ele1.energyClus[ele1.typeClus == i]), ak.ravel(ele2.energyClus[ele2.typeClus == i])]).to_numpy(), bins = 50, range = plot_range, histtype = "step", label = type_dict[i])
ax.set_xlabel("E [MeV]")
ax.set_ylabel("Entries")
ax.set_title("MTD cluster energy", pad = 40)
ax.set_yscale("log")
ax.legend()

ax = axs[2, 1]
plot_range = (0, 10)
ax.hist(ak.concatenate([ak.ravel(ele1.energyClus[abs(eta) < 0.6]), ak.ravel(ele2.energyClus[abs(eta) < 0.6])]).to_numpy(), bins = 50, histtype = "step", range = plot_range, label = "$|\eta| < 0.6$", density = True)
ax.hist(ak.concatenate([ak.ravel(ele1.energyClus[abs(eta) >= 0.6]), ak.ravel(ele2.energyClus[abs(eta) < 0.6])]).to_numpy(), bins = 50, histtype = "step", range = plot_range, label = "$|\eta| \geq 0.6$", density = True)
ax.set_xlabel("E [MeV]")
ax.set_ylabel("Density")
ax.set_title("MTD cluster energy vs. eta", pad = 40)
ax.set_yscale("log")
ax.legend()

ax = axs[2, 2]
ax.hist(ak.concatenate([ak.ravel(ele1.nHitsClus), ak.ravel(ele2.nHitsClus)]).to_numpy(), bins = 11, range = (-0.5, 10.5), histtype = "step", label = "All", density = True)
# only for MTD conversions
ax.hist(ak.concatenate([ak.ravel(ele1.nHitsClus[rho > 115]), ak.ravel(ele2.nHitsClus[rho > 115])]).to_numpy(), bins = 11, range = (-0.5, 10.5), histtype = "step", label = "MTD conversions", density = True)
ax.set_xlabel("#hits")
ax.set_ylabel("Density")
ax.set_title("MTD cluster number of hits", pad = 40)
ax.set_yscale("log")
ax.legend()

ax = axs[2, 3]
# number of clusters PER ELECTRON
xmax = 40
ax.hist(ak.concatenate([ak.ravel(ak.num(ele1.tClus[ele1.tClus > -100], axis = 2)), ak.ravel(ak.num(ele2.tClus[ele2.tClus > -100], axis = 2))]), bins = xmax + 1, range = (-0.5, xmax + 0.5), histtype = "step", label = "All", density = True)
# MTD conversions only
ax.hist(ak.concatenate([ak.ravel(ak.num(ele1.tClus[ele1.tClus > -100][rho > 115], axis = 2)), ak.ravel(ak.num(ele2.tClus[ele2.tClus > -100][rho > 115], axis = 2))]), bins = xmax + 1, range = (-0.5, xmax + 0.5), histtype = "step", label = "MTD conversions", density = True)
ax.set_xlabel("#clusters")
ax.set_ylabel("Entries")
ax.set_title("Number of clusters per electron", pad = 40)
ax.set_yscale("log")
ax.legend()

# -------------------------

# x, y, z positions
vars = ["xClus", "yClus", "zClus"]
for var, ax in zip(vars, axs[3]):
    plot_range = [-120, 120] if var != "zClus" else [-300, 300]
    ax.hist(ak.concatenate([ak.ravel(ele1[var]), ak.ravel(ele2[var])]), bins = 100, range = plot_range, histtype = "step", label = "All")
    ax.set_xlabel(f"{var[:1]} [cm]")
    ax.set_ylabel("Entries")
    ax.set_title(f"MTD cluster {var[:1]} position", pad = 40)
    ax.grid()
    ax.legend()

# x, y, z positions std dev
vars = ["tStdClus", "xStdClus", "yStdClus", "zStdClus"]
for var, ax in zip(vars, axs[4]):
    plot_range = [0, 2]
    nbins = 50
    if var == "tStdClus":
        plot_range[1] = 0.3
    elif var == "yStdClus":
        plot_range[1] = 0.2
    elif var == "zStdClus":
        plot_range[1] = 0.2
    # All conversions
    plot_hist_from_array(ak.concatenate([ak.ravel(ele1[var]), ak.ravel(ele2[var])]), ax = ax, bins = nbins, range = plot_range, histtype = "step", label = "All", density = True, print_stats = False)
    # MTD conversions
    plot_hist_from_array(ak.concatenate([ak.ravel(ele1[var][rho > 115]), ak.ravel(ele2[var][rho > 115])]), ax = ax, bins = nbins, range = plot_range, histtype = "step", label = "MTD conversions", density = True, print_stats = False)
    unit = "[cm]" if var != "tStdClus" else "[ns]"
    ax.set_xlabel(f"{var[:1]} {unit}")
    ax.set_ylabel("Density")
    ax.set_title(f"MTD hit {var[:1]} std. dev. in cluster", pad = 40)
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

# plot standard deviation of clusters PER ELECTRON
vars = ["tClus", "xClus", "yClus", "zClus"]
for var, ax in zip(vars, axs[5]):
    plot_range = [0, 150]
    if var == "tClus":
        plot_range[1] = 10
    if var == "xClus" or var == "yClus":
        plot_range[1] = 120
    # All conversions
    plot_hist_from_array(ak.concatenate([ak.ravel(ak.std(ele1[var], axis = 2)), ak.ravel(ak.std(ele2[var], axis = 2))]).to_numpy(), ax = ax, bins = 50, range = plot_range, histtype = "step", label = "All", density = True, print_stats = False)
    # MTD conversions
    plot_hist_from_array(ak.concatenate([ak.ravel(ak.std(ele1[var][rho > 115], axis = 2)), ak.ravel(ak.std(ele2[var][rho > 115], axis = 2))]).to_numpy(), ax = ax, bins = 50, range = plot_range, histtype = "step", label = "MTD conversions", density = True,  print_stats = False)
    # exclude calo backscatter
    clusters1 = ele1[var][rho > 115][ele1.typeClus[rho > 115] <= 0]
    clusters2 = ele2[var][rho > 115][ele2.typeClus[rho > 115] <= 0]
    plot_hist_from_array(ak.concatenate([ak.ravel(ak.std(clusters1, axis = 2)), ak.ravel(ak.std(clusters2, axis = 2))]).to_numpy(), ax = ax, bins = 50, range = plot_range, histtype = "step", label = "All, direct only", density = True, print_stats = False)
    unit = "[cm]" if var != "tStdClus" else "[ns]"
    ax.set_xlabel(f"{var[:1]} {unit}")
    ax.set_ylabel("Density")
    ax.set_title(f"MTD cluster {var[:1]} std. dev. per electron", pad = 40)
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

# plot standard deviation of clusters PER ELECTRON for direct, detailed zoom
vars = ["tClus", "xClus", "yClus", "zClus"]
for var, ax in zip(vars, axs[6]):
    plot_range = [0, 0.3]
    if var != "tClus":
        plot_range[1] = 4
    # exclude calo backscatter
    clusters1 = ele1[var][rho > 115][ele1.typeClus[rho > 115] <= 0]
    clusters2 = ele2[var][rho > 115][ele2.typeClus[rho > 115] <= 0]
    plot_hist_from_array(ak.concatenate([ak.ravel(ak.std(clusters1, axis = 2)), ak.ravel(ak.std(clusters2, axis = 2))]).to_numpy(), ax = ax, bins = 50, range = plot_range, histtype = "step", label = "All, direct only", density = True, print_stats = False)
    unit = "[cm]" if var != "tStdClus" else "[ns]"
    ax.set_xlabel(f"{var[:1]} {unit}")
    ax.set_ylabel("Density")
    ax.set_title(f"MTD cluster {var[:1]} std. dev. per electron, directs only", pad = 40)
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

# phi
ax = axs[-1, 0]
phi1 = ak.ravel(np.arctan2(ymtd_1[abs(ymtd_1) < 130], xmtd_1[abs(xmtd_1) < 130]))
phi2 = ak.ravel(np.arctan2(ymtd_2[abs(ymtd_2) < 130], xmtd_2[abs(xmtd_2) < 130]))

ax.hist(ak.ravel(ak.concatenate([phi1, phi2])), bins = 50, histtype = "step", label = "All")
ax.grid()
ax.set_xlabel(r"$\phi$")
ax.set_ylabel("Entries")
ax.legend()

# rho
ax = axs[-1, 1]
rho1 = np.sqrt(xmtd_1[abs(xmtd_1) < 130]**2 + ymtd_1[abs(ymtd_1) < 130]**2)
rho2 = np.sqrt(xmtd_2[abs(xmtd_2) < 130]**2 + ymtd_2[abs(ymtd_2) < 130]**2)
ax.hist(ak.ravel(ak.concatenate([rho1, rho2])).to_numpy(), bins = 100, range = [115.4, 116.4], histtype = "step", label = "All")
ax.set_xlabel("R [cm]")
ax.set_ylabel("Entries")
ax.set_title("MTD cluster radius", pad = 40)
ax.legend()

fig.savefig(f"{local_folder}/mtd_cluster_info.png")
fig.savefig(f"{local_folder}/mtd_cluster_info.pdf")

# -------- CONVERSION KINEMATIC PLOTS ------- #

# misc information about dielectrons
# (ELECTRON, ELECTRON)
nrows = 3
ncols = 6
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

### deltaPt
dPt = (ele1.pt - ele2.pt)/ele1.pt

ax = axs[0, 0]
ax.hist(ak.flatten(dPt), bins = 50)
ax.set_xlabel(r"$\Delta p_T/p_T$ [GeV]")
ax.set_ylabel("Entries")
ax.set_title("Dielectron relative $\Delta p_T$ (over leading electron $p_T$)", pad = 40)
### dt

dt = selected1(tmtd_1) - selected2(tmtd_2)
ax = axs[0, 1]
plot_hist_from_array(dt, ax = ax, bins = 100, histtype = "step", range = (-0.3, 0.3), label = "All")
# same but only for mtd conversions
plot_hist_from_array(dt[rho[good_cluster_photons] >= 115], ax = ax, bins = 100, histtype = "step", range = [-0.3, 0.3], label = "MTD conversions")
ax.set_xlabel(r"$\Delta t$ [ns]")
ax.set_ylabel("Entries")
ax.set_title("Dielectron $\Delta t$", pad = 40)
ax.set_yscale("log")
ax.legend()

### deltaPhi
phi1 = np.arctan2(selected1(ymtd_1), selected1(xmtd_1))
phi2 = np.arctan2(selected2(ymtd_2), selected2(xmtd_2))

dphi = angle_to_mpi_pi(phi1 - phi2)

ax = axs[0, 2]
ax.hist(ak.flatten(dphi), bins = 100, range = (-0.3, 0.3), histtype = "step", label = "All")
# same but only for mtd conversions
ax.hist(ak.flatten(dphi[rho[good_cluster_photons] >= 115]), bins = 100, histtype = "step", range = (-0.3, 0.3), label = "MTD conversions")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel("Entries")
ax.set_title("Dielectron $\Delta \phi$ spectrum", pad = 40)
ax.set_yscale("log")
ax.legend()

### deltaPhi vs conversion radius
ax = axs[0, 3]
ax.scatter(ak.ravel(rho[good_cluster_photons]), ak.ravel(dphi), s = 1)
# h = ax.hist2d(ak.ravel(rho[good_cluster_photons]), ak.ravel(dphi), bins = 30, norm=mpl.colors.LogNorm())
# fig.colorbar(h[3], ax = ax)
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel(r"$\Delta \phi$")
ax.set_title("Dielectron $\Delta \phi$ vs conversion radius", pad = 40)


### deltaEta
ax = axs[0, 4]
# compute eta
eta1 = np.arcsinh(selected1(zmtd_1) / np.sqrt(selected1(xmtd_1)**2 + selected1(ymtd_1)**2))
eta2 = np.arcsinh(selected2(zmtd_2) / np.sqrt(selected2(xmtd_2)**2 + selected2(ymtd_2)**2))
deta = eta1 - eta2

ax.hist(ak.flatten(deta), bins = 100, range = (-0.3, 0.3), histtype = "step", label = "All")
# same but only for mtd conversions
ax.hist(ak.flatten(deta[rho[good_cluster_photons] >= 115]), bins = 100, histtype = "step", range = (-0.3, 0.3), label = "MTD conversions")

ax.set_xlabel(r"$\Delta \eta$")
ax.set_ylabel("Entries")
ax.set_title("Dielectron $\Delta \eta$ spectrum", pad = 40)
ax.set_yscale("log")
ax.legend()

### deltaEta vs conversion radius
ax = axs[0, 5]

ax.scatter(ak.ravel(rho[good_cluster_photons]), ak.ravel(deta), s = 1)
# h = ax.hist2d(ak.ravel(rho[good_cluster_photons]), ak.ravel(dphi), bins = 30, norm=mpl.colors.LogNorm())
# fig.colorbar(h[3], ax = ax)
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel(r"$\Delta \eta$")
ax.set_title("Dielectron $\Delta \eta$ vs conversion radius", pad = 40)

### (PHOTON, ELECTRON)

### delta t
# compute TOF from MTD positions with straight line approximation
c = 29.9792458 # cm/ns
tof1 = np.sqrt(selected1(xmtd_1 - genPV.x)**2 + selected1(ymtd_1 - genPV.y)**2 + selected1(zmtd_1 - genPV.z)**2) / c
tof2 = np.sqrt(selected2(xmtd_2 - genPV.x)**2 + selected2(ymtd_2 - genPV.y)**2 + selected2(zmtd_2 - genPV.z)**2) / c

t0_e1 = selected1(tmtd_1) - tof1
t0_e2 = selected2(tmtd_2) - tof2

# compare to photon t0  
t0_photon = conv_photons.t0[good_cluster_photons]
dt1 = t0_e1 - genPV.t #t0_photon
dt2 = t0_e2 - genPV.t #t0_photon

ax = axs[1, 0]
plot_range = [-0.02, 20]
plot_hist_from_array(dt1, ax = ax, bins = 50, histtype = "step", label = "All, lead", print_stats = False, range = plot_range)
plot_hist_from_array(dt2, ax = ax, bins = 50, histtype = "step", label = "All, sublead", print_stats = False, range = plot_range)
# same but only for mtd conversions
plot_hist_from_array(dt1[rho[good_cluster_photons] >= 115], ax = ax, histtype = "step", alpha = 0.8, linestyle = "dashed", bins = 50, color = "C0", label = "MTD conv., lead", print_stats = False, range = plot_range)
plot_hist_from_array(dt2[rho[good_cluster_photons] >= 115], ax = ax, histtype = "step", alpha = 0.8, linestyle = "dashed", bins = 50, color = "C1", label = "MTD conv., sublead", print_stats = False, range = plot_range)
ax.set_xlabel(r"$\Delta t$ [ns]")
ax.set_ylabel("Entries")
ax.set_yscale("log")
ax.set_title("Photon-ele $\Delta TOF$ distribution", pad = 40)
ax.legend()

### deltaPhi
# compute dphi wrt photon
phi1 = np.arctan2(selected1(ymtd_1), selected1(xmtd_1))
phi2 = np.arctan2(selected2(ymtd_2), selected2(xmtd_2))

dphi1_photon = angle_to_mpi_pi(phi1 - conv_photons.phi[good_cluster_photons])
dphi2_photon = angle_to_mpi_pi(phi2 - conv_photons.phi[good_cluster_photons])

ax = axs[1, 1]
plot_range = [-0.5, 0.5]
ax.hist(ak.flatten(dphi1_photon), bins = 50, histtype = "step", range = plot_range, label = "All, lead")
ax.hist(ak.flatten(dphi2_photon), bins = 50, histtype = "step", range = plot_range, label = "All, sublead")
# same but only for mtd conversions
ax.hist(ak.flatten(dphi1_photon[rho[good_cluster_photons] >= 115]), bins = 50, range = plot_range, color = "C0", histtype = "step", linestyle = "dashed", alpha = 0.9, label = "MTD conv., lead")
ax.hist(ak.flatten(dphi2_photon[rho[good_cluster_photons] >= 115]), bins = 50, range = plot_range, color = "C1", histtype = "step", linestyle = "dashed", alpha = 0.9, label = "MTD conv., sublead")

ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel("Entries")
ax.set_title("Photon-ele $\Delta \phi$ distribution", pad = 40)
ax.legend()

### deltaPhi(ele, photon) vs photon pT
ax = axs[1, 2]
ax.scatter(ak.flatten(conv_photons.pt[good_cluster_photons]), ak.flatten(dphi1_photon), s = 1, label = "Lead")
ax.scatter(ak.flatten(conv_photons.pt[good_cluster_photons]), ak.flatten(dphi2_photon), s = 1, label = "Sublead")
ax.set_xlabel("Photon $p_T$ [GeV]")
ax.set_ylabel(r"$\Delta \phi$")
ax.set_title("Photon-ele $\Delta \phi$ vs photon $p_T$", pad = 40)
ax.legend()

### deltaPhi(ele, photon) vs photon pT, histogram (same)
ax = axs[2, 2]
arrx = ak.concatenate([ak.ravel(conv_photons.pt[good_cluster_photons]), ak.ravel(conv_photons.pt[good_cluster_photons])])
arry = ak.concatenate([ak.ravel(dphi1_photon), ak.ravel(dphi2_photon)])
h = ax.hist2d(arrx.to_numpy(), arry.to_numpy(), bins = 30, norm = mpl.colors.LogNorm(), range = [[0, 10], [-1, 1]])
fig.colorbar(h[3], ax = ax, fraction = 0.05)
ax.set_xlabel("Photon $p_T$ [GeV]")
ax.set_ylabel(r"$\Delta \phi$")
ax.set_title("Photon-ele $\Delta \phi$ vs photon $p_T$, histogram", pad = 40)
ax.legend()

### deltaPhi(ele, photon) vs conversion radius
ax = axs[1, 3]
ax.scatter(ak.flatten(rho[good_cluster_photons]), ak.flatten(dphi1_photon), s = 1, label = "Lead")
ax.scatter(ak.flatten(rho[good_cluster_photons]), ak.flatten(dphi2_photon), s = 1, label = "Sublead")
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel(r"$\Delta \phi$")
ax.set_title("Photon-ele $\Delta \phi$ vs conversion radius", pad = 40)
ax.legend()

### deltaPhi(ele, photon) vs conversion radius, histogram (same)
ax = axs[2, 3]

arrx = ak.concatenate([ak.ravel(rho[good_cluster_photons]), ak.ravel(rho[good_cluster_photons])])
arry = ak.concatenate([ak.ravel(dphi1_photon), ak.ravel(dphi2_photon)])
h = ax.hist2d(arrx.to_numpy(), arry.to_numpy(), bins = 30, norm = mpl.colors.LogNorm(), range = [[0, 120], [-1, 1]])
fig.colorbar(h[3], ax = ax, fraction = 0.05)
ax.set_xlabel("Conversion radius [cm]")
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel(r"$\Delta \phi$")
ax.set_title("Photon-ele $\Delta \phi$ vs conversion radius, histogram", pad = 40)

### deltaEta(ele, photon) vs conversion radius
ax = axs[1, 4]

deta1_photon = eta1 - conv_photons.eta[good_cluster_photons]
deta2_photon = eta2 - conv_photons.eta[good_cluster_photons]

ax.scatter(ak.flatten(rho[good_cluster_photons]), ak.flatten(deta1_photon), s = 1, label = "Lead")
ax.scatter(ak.flatten(rho[good_cluster_photons]), ak.flatten(deta2_photon), s = 1, label = "Sublead")
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel(r"$\Delta \eta$")
ax.set_title("Photon-ele $\Delta \eta$ vs conversion radius", pad = 40)
ax.legend()

### deltaEta(ele, photon) vs conversion radius, histogram (same)
ax = axs[2, 4]

arrx = ak.concatenate([ak.ravel(rho[good_cluster_photons]), ak.ravel(rho[good_cluster_photons])])
arry = ak.concatenate([ak.ravel(deta1_photon), ak.ravel(deta2_photon)])
h = ax.hist2d(arrx.to_numpy(), arry.to_numpy(), bins = 30, norm = mpl.colors.LogNorm(), range = [[0, 120], [-1, 1]])
fig.colorbar(h[3], ax = ax, fraction = 0.05)
ax.set_xlabel("Conversion radius [cm]")
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel(r"$\Delta \eta$")
ax.set_title("Photon-ele $\Delta \eta$ vs conversion radius, histogram", pad = 40)

### conversion radius
ax = axs[1, 5]
ax.hist(ak.ravel(rho), bins = 50, histtype = "step", label = "All", density = True)
ax.hist(ak.ravel(rho[good_cluster_photons]), bins = 50, histtype = "step", label = "2 direct hits", density = True)
ax.set_xlabel("Conversion radius [cm]")
ax.set_ylabel("Entries")
ax.set_title("Conversion radius", pad = 40)
ax.legend()

##### BACK TO DIELECTRON KINEMATICS

# pt1 vs pt2
ax = axs[2, 0]
ax.scatter(ak.flatten(ele1.pt), ak.flatten(ele2.pt), s = 1, alpha = 0.5)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.set_xlabel(r"$p_T(1)$ [GeV]")
ax.set_ylabel(r"$p_T(2)$ [GeV]")
ax.set_title("Dielectron $p_T$ correlation", pad = 40)

fig.savefig(f"{local_folder}/conversion_kinematics.png")
fig.savefig(f"{local_folder}/conversion_kinematics.pdf")

# -------- CORE PLOTS --------- #
nrows = 3
ncols = 3
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

ax = axs[0, 0]
arrx = ak.concatenate([ak.ravel(dphi1_photon), ak.ravel(dphi2_photon)])
arry = ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)])
ax.scatter(arrx, arry, s = 1, label = "Direct")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
# ax.set_ylim(-0.01, 0.01)
# ax.set_xlim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison, direct clusters only", pad = 40)
ax.legend()

# same but for MTD conversions only
ax = axs[0, 1]
arrx = ak.concatenate([ak.ravel(dphi1_photon[rho[good_cluster_photons] >= 115]), ak.ravel(dphi2_photon[rho[good_cluster_photons] >= 115])])
arry = ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] >= 115]), ak.ravel(dt2[rho[good_cluster_photons] >= 115])])
ax.scatter(arrx, arry, s = 1, label = "Direct")

ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
# ax.set_ylim(-0.01, 0.01)
# ax.set_xlim(-0.5, 0.5)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for MTD conversions, direct clusters only", pad = 40)
ax.legend()

# same but for (innish) TRACKER conversions only
ax = axs[0, 2]
arrx = ak.concatenate([ak.ravel(dphi1_photon[rho[good_cluster_photons] < 90]), ak.ravel(dphi2_photon[rho[good_cluster_photons] < 90])])
arry = ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] < 90]), ak.ravel(dt2[rho[good_cluster_photons] < 90])])
ax.scatter(arrx, arry, s = 1, label = "Direct")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for tracker conversions (rho < 90 cm)", pad = 40)
ax.legend()

### SECOND ROW: same, but select all clusters instead
# plot deltaPhi vs deltaTOF for ALL clusters
tof1_all = np.sqrt((xmtd_1 - genPV.x)**2 + (ymtd_1 - genPV.y)**2 + (zmtd_1 - genPV.z)**2) / c
tof2_all = np.sqrt((xmtd_2 - genPV.x)**2 + (ymtd_2 - genPV.y)**2 + (zmtd_2 - genPV.z)**2) / c

t0_e1_all = (tmtd_1) - tof1_all
t0_e2_all = (tmtd_2) - tof2_all

# compare to photon t0
dt1_all = t0_e1_all - genPV.t # conv_photons.t0
dt2_all = t0_e2_all - genPV.t # conv_photons.t0

phi1_all = np.arctan2(ymtd_1, xmtd_1)
phi2_all = np.arctan2(ymtd_2, xmtd_2)

dphi1_photon_all = angle_to_mpi_pi(phi1_all - conv_photons.phi)
dphi2_photon_all = angle_to_mpi_pi(phi2_all - conv_photons.phi)

ax = axs[1, 0]
arrx = ak.concatenate([ak.ravel(dphi1_photon_all), ak.ravel(dphi2_photon_all)])
arry = ak.concatenate([ak.ravel(dt1_all), ak.ravel(dt2_all)])

ax.scatter(arrx, arry, s = 1, label = "All")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for ALL clusters", pad = 40)
ax.legend()

# same but for MTD conversions only
ax = axs[1, 1]
arrx = ak.concatenate([ak.ravel(dphi1_photon_all[rho >= 115]), ak.ravel(dphi2_photon_all[rho >= 115])])
arry = ak.concatenate([ak.ravel(dt1_all[rho >= 115]), ak.ravel(dt2_all[rho >= 115])])
ax.scatter(arrx, arry, s = 1, label = "All")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for MTD conversions, all clusters", pad = 40)
ax.legend()

# same but for (innish) TRACKER conversions only
ax = axs[1, 2]
# arrx = ak.concatenate([ak.ravel(dphi1_photon_all[rho < 70]), ak.ravel(dphi2_photon_all[rho < 70])])
# arry = ak.concatenate([ak.ravel(dt1_all[rho < 70]), ak.ravel(dt2_all[rho < 70])])
# ax.scatter(arrx, arry, s = 1, label = "All")

# scatter for each cluster type separately
arrx = {i : ak.concatenate([ak.ravel(dphi1_photon_all[(ele1.typeClus == i) * (rho < 90)]),
                            ak.ravel(dphi2_photon_all[(ele2.typeClus == i) * (rho < 90)])]) for i in range(2)}

arry = {i : ak.concatenate([ak.ravel(dt1_all[(ele1.typeClus == i) * (rho < 90)]),
                            ak.ravel(dt2_all[(ele2.typeClus == i) * (rho < 90)])]) for i in range(2)}
for i in range(2):
    ax.scatter(arrx[i], arry[i], s = 1, label = type_dict[i])
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.legend(markerscale = 5)

ax.set_title("TOF comparison for tracker conversions (rho < 90 cm), all categories", pad = 40)

# Test dependence on eta
ax = axs[2, 0]
m_inner = (abs(eta[good_cluster_photons]) < 0.6)
m_outer = (abs(eta[good_cluster_photons]) >= 0.6)

arrx = ak.concatenate([ak.ravel(dphi1_photon[m_inner]), ak.ravel(dphi2_photon[m_inner])])
arry = ak.concatenate([ak.ravel(dt1[m_inner]), ak.ravel(dt2[m_inner])])
ax.scatter(arrx, arry, s = 1, label = r"$|\eta| < 0.6$")

arrx = ak.concatenate([ak.ravel(dphi1_photon[m_outer]), ak.ravel(dphi2_photon[m_outer])])
arry = ak.concatenate([ak.ravel(dt1[m_outer]), ak.ravel(dt2[m_outer])])
ax.scatter(arrx, arry, s = 1, label = r"$|\eta| \geq 0.6$")

ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF of direct clusters only, inner vs outer barrel", pad = 40)
ax.legend()

# same but for MTD conversions only
ax = axs[2, 1]

m_inner = (rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) < 0.6)
m_outer = (rho[good_cluster_photons] >= 115) * (abs(eta[good_cluster_photons]) >= 0.6)
arrx = ak.concatenate([ak.ravel(dphi1_photon[m_inner]), ak.ravel(dphi2_photon[m_inner])])
arry = ak.concatenate([ak.ravel(dt1[m_inner]), ak.ravel(dt2[m_inner])])
ax.scatter(arrx, arry, s = 1, label = r"$|\eta| < 0.6$")

arrx = ak.concatenate([ak.ravel(dphi1_photon[m_outer]), ak.ravel(dphi2_photon[m_outer])])
arry = ak.concatenate([ak.ravel(dt1[m_outer]), ak.ravel(dt2[m_outer])])
ax.scatter(arrx, arry, s = 1, label = r"$|\eta| \geq 0.6$")

ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF for MTD conversions, inner vs outer barrel", pad = 40)
ax.legend()

# Tracker conversions
ax = axs[2, 2]
m_inner = (rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) < 0.6)
m_outer = (rho[good_cluster_photons] < 90) * (abs(eta[good_cluster_photons]) >= 0.6)
arrx = ak.concatenate([ak.ravel(dphi1_photon[m_inner]), ak.ravel(dphi2_photon[m_inner])])
arry = ak.concatenate([ak.ravel(dt1[m_inner]), ak.ravel(dt2[m_inner])])
ax.scatter(arrx, arry, s = 1, label = r"$|\eta| < 0.6$")

arrx = ak.concatenate([ak.ravel(dphi1_photon[m_outer]), ak.ravel(dphi2_photon[m_outer])])
arry = ak.concatenate([ak.ravel(dt1[m_outer]), ak.ravel(dt2[m_outer])])
ax.scatter(arrx, arry, s = 1, label = r"$|\eta| \geq 0.6$")

ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF for Rconv < 90 cm, direct hits, inner vs outer barrel", pad = 40)
ax.legend()
ax.grid()

fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_dphi_scatter.png")
fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_dphi_scatter.pdf")

#### IDEM BUT DELTAETA
nrows = 2
ncols = 3
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

ax = axs[0, 0]
arrx = ak.concatenate([ak.ravel(deta1_photon), ak.ravel(deta2_photon)])
arry = ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)])
ax.scatter(arrx, arry, s = 1, label = "Direct")
ax.set_xlabel(r"$\Delta \eta$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison, direct clusters only", pad = 40)
ax.legend()

# same but for MTD conversions only
ax = axs[0, 1]
arrx = ak.concatenate([ak.ravel(deta1_photon[rho[good_cluster_photons] >= 115]), ak.ravel(deta2_photon[rho[good_cluster_photons] >= 115])])
arry = ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] >= 115]), ak.ravel(dt2[rho[good_cluster_photons] >= 115])])
ax.scatter(arrx, arry, s = 1, label = "Direct")

ax.set_xlabel(r"$\Delta \eta$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for MTD conversions, direct clusters only", pad = 40)
ax.legend()

# same but for (innish) TRACKER conversions only
ax = axs[0, 2]
arrx = ak.concatenate([ak.ravel(deta1_photon[rho[good_cluster_photons] < 90]), ak.ravel(deta2_photon[rho[good_cluster_photons] < 90])])
arry = ak.concatenate([ak.ravel(dt1[rho[good_cluster_photons] < 90]), ak.ravel(dt2[rho[good_cluster_photons] < 90])])
ax.scatter(arrx, arry, s = 1, label = "Direct")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for tracker conversions (rho < 90 cm)", pad = 40)
ax.legend()

### SECOND ROW: same, but select all clusters instead
# plot deltaEta vs deltaTOF for ALL clusters
eta1_all = np.arcsinh(zmtd_1 / np.sqrt(xmtd_1**2 + ymtd_1**2))
eta2_all = np.arcsinh(zmtd_2 / np.sqrt(xmtd_2**2 + ymtd_2**2))

deta1_photon_all = eta1_all - conv_photons.eta
deta2_photon_all = eta2_all - conv_photons.eta

ax = axs[1, 0]
arrx = ak.concatenate([ak.ravel(deta1_photon_all), ak.ravel(deta2_photon_all)])
arry = ak.concatenate([ak.ravel(dt1_all), ak.ravel(dt2_all)])

ax.scatter(arrx, arry, s = 1, label = "All")
ax.set_xlabel(r"$\Delta \eta$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for ALL clusters", pad = 40)
ax.legend()

# same but for MTD conversions only
ax = axs[1, 1]
arrx = ak.concatenate([ak.ravel(deta1_photon_all[rho >= 115]), ak.ravel(deta2_photon_all[rho >= 115])])
arry = ak.concatenate([ak.ravel(dt1_all[rho >= 115]), ak.ravel(dt2_all[rho >= 115])])
ax.scatter(arrx, arry, s = 1, label = "All")
ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.01, 0.2)
ax.set_title("TOF comparison for MTD conversions, all clusters", pad = 40)
ax.legend()

# same but for (innish) TRACKER conversions only
ax = axs[1, 2]
# arrx = ak.concatenate([ak.ravel(dphi1_photon_all[rho < 70]), ak.ravel(dphi2_photon_all[rho < 70])])
# arry = ak.concatenate([ak.ravel(dt1_all[rho < 70]), ak.ravel(dt2_all[rho < 70])])
# ax.scatter(arrx, arry, s = 1, label = "All")

# scatter for each cluster type separately
arrx = {i : ak.concatenate([ak.ravel(deta1_photon_all[(ele1.typeClus == i) * (rho < 90)]),
                            ak.ravel(deta2_photon_all[(ele2.typeClus == i) * (rho < 90)])]) for i in range(2)}

arry = {i : ak.concatenate([ak.ravel(dt1_all[(ele1.typeClus == i) * (rho < 90)]),
                            ak.ravel(dt2_all[(ele2.typeClus == i) * (rho < 90)])]) for i in range(2)}
for i in range(2):
    ax.scatter(arrx[i], arry[i], s = 1, label = type_dict[i])
ax.set_xlabel(r"$\Delta \eta$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.01, 0.2)
ax.legend(markerscale = 5)

ax.set_title("TOF comparison for tracker conversions (rho < 90 cm), all categories", pad = 40)

fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_deta_scatter.png")
fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_deta_scatter.pdf")

# ----- Histograms and profiles of the same above

nrows = 1
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

ax = axs[0]
for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:]):
    mask = (rho >= rho_min) * (rho < rho_max)
    mask1 = mask * (ele1.typeClus == 0)
    mask2 = mask * (ele2.typeClus == 0)

    dt1 = tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t
    dt2 = tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t

    ax.hist(ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)]), 
            histtype = "step", bins = 30, range = [-0.01, 0.1],
            density = True, label = f"{rho_min} < Rconv < {rho_max} cm")

ax.set_xlabel(r"$\Delta TOF$ [ns]")
ax.set_ylabel("Density")
ax.set_title("TOF comparison for direct clusters vs conversion radius", pad = 40)
ax.legend()

ax = axs[1]
# do profile
for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:]):
    phi_range = np.linspace(-0.5, 0.5, 10)

    xs = []
    xs_err = []
    ys = []
    ys_err = []

    for phi_min, phi_max in zip(phi_range[:-1], phi_range[1:]):
        # TODO: fix profile; stats don't match up
        ### BOOKMARK 1: deltaTOF
        mask = (rho >= rho_min) * (rho < rho_max)
        mask1 = mask * (ele1.typeClus == 0) * (dphi1_photon_all >= phi_min) * (dphi1_photon_all < phi_max)
        mask2 = mask * (ele2.typeClus == 0) * (dphi2_photon_all >= phi_min) * (dphi2_photon_all < phi_max)

        dt1 = tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t
        dt2 = tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t

        arr = ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)])

        # ax.hist(arr, bins = 30, histtype = "step", range = [0, 0.1], density = True, label = "{:.1f} < phi < {:.1f}; {:.1f} < rho < {:.1f}".format(phi_min, phi_max, rho_min, rho_max))

        ys.append(np.mean(arr))
        ys_err.append(np.std(arr))
        xs.append((phi_min + phi_max)/2)
        xs_err.append((phi_max - phi_min)/2)

    # plot as errorbar
    ax.errorbar(xs, ys, xerr = xs_err, yerr = ys_err, fmt = "o", 
                label = f"{rho_min} < Rconv < {rho_max} cm",
                markersize = 5, capsize = 5)

ax.set_xlabel(r"$\Delta \phi$")
ax.set_ylabel(r"$\Delta TOF$ [ns]")
ax.set_title("TOF comparison for direct clusters vs conversion radius", pad = 40)
ax.legend()

fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_dphi.png")
fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_dphi.pdf")

# do each radius separately (scatter + profile overlayed)
nrows = 1
ncols = len(rho_range) - 1
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

results = {}

parabola = lambda x, a, b, c : a * x**2 + b * x + c

for ax, rho_min, rho_max in zip(axs, rho_range[:-1], rho_range[1:]):
    phi_range = np.linspace(-0.5, 0.5, 20)

    xs = []
    xs_err = []
    ys = []
    ys_err = []
    nentries = []

    mask = (rho >= rho_min) * (rho < rho_max)

    for phi_min, phi_max in zip(phi_range[:-1], phi_range[1:]):
        mask1 = mask * (ele1.typeClus == 0) * (dphi1_photon_all >= phi_min) * (dphi1_photon_all < phi_max)
        mask2 = mask * (ele2.typeClus == 0) * (dphi2_photon_all >= phi_min) * (dphi2_photon_all < phi_max)
        ## CHECK: deta cut
        # mask1 = mask * (ele1.typeClus == 0) * (dphi1_photon_all >= phi_min) * (dphi1_photon_all < phi_max) * (abs(deta1_photon_all) <= 0.02)
        # mask2 = mask * (ele2.typeClus == 0) * (dphi2_photon_all >= phi_min) * (dphi2_photon_all < phi_max) * (abs(deta2_photon_all) <= 0.02)

        dt1 = ak.firsts(tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t, axis = 2)
        dt2 = ak.firsts(tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t, axis = 2)

        arr = ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)])

        # ys.append(np.mean(arr))
        # ys_err.append(np.std(arr))
        nentries.append(len(arr[(arr > -0.1) * (arr < 0.2)]))
        ys.append(np.mean(arr[(arr > -0.1) * (arr < 0.2)]))
        ys_err.append(np.std(arr[(arr > -0.1) * (arr < 0.2)]))

        xs.append((phi_min + phi_max)/2)
        xs_err.append((phi_max - phi_min)/2)

        # print(f"phi_min = {phi_min}, phi_max = {phi_max}, rho_min = {rho_min}, rho_max = {rho_max}")
        # print(f"\tEvents = {len(arr)}, mean = {np.mean(arr):.3g}, std = {np.std(arr):.3g}")
        # print(f"\tMean in -0.1, 0.2 range = {np.mean(arr[(arr > -0.1) * (arr < 0.2)]):.3g}")
        # print(f"\tStd in -0.1, 0.2 range = {np.std(arr[(arr > -0.1) * (arr < 0.2)]):.3g}")

    mask1 = mask * (ele1.typeClus == 0)
    mask2 = mask * (ele2.typeClus == 0)

    dt1 = ak.firsts(tmtd_1[mask1] - np.sqrt((xmtd_1[mask1] - genPV.x)**2 + (ymtd_1[mask1] - genPV.y)**2 + (zmtd_1[mask1] - genPV.z)**2) / c - genPV.t, axis = 2)
    dt2 = ak.firsts(tmtd_2[mask2] - np.sqrt((xmtd_2[mask2] - genPV.x)**2 + (ymtd_2[mask2] - genPV.y)**2 + (zmtd_2[mask2] - genPV.z)**2) / c - genPV.t, axis = 2)

    dphi1 = angle_to_mpi_pi(ak.firsts(np.arctan2(ymtd_1[mask1], xmtd_1[mask1]), axis = 2) - conv_photons.phi)
    dphi2 = angle_to_mpi_pi(ak.firsts(np.arctan2(ymtd_2[mask2], xmtd_2[mask2]), axis = 2) - conv_photons.phi)

    arrx = ak.concatenate([ak.ravel(dphi1), ak.ravel(dphi2)])
    arry = ak.concatenate([ak.ravel(dt1), ak.ravel(dt2)])

    # remove nans from xs, ys
    nan_mask = np.isnan(ys) + np.isnan(ys_err) + (np.array(nentries) < 3)
    xs = np.array(xs)[~nan_mask]
    xs_err = np.array(xs_err)[~nan_mask]
    ys = np.array(ys)[~nan_mask]
    ys_err = np.array(ys_err)[~nan_mask]

    # plot scatter
    ax.scatter(arrx, arry, s = 2, alpha = 0.1, label = "Direct")

    # plot as errorbar
    ax.errorbar(xs, ys, xerr = xs_err, yerr = ys_err, fmt = "o", 
                label = f"{rho_min} < Rconv < {rho_max} cm",
                markersize = 5, capsize = 5)

    # fit xs, ys with a parabola with errors
    popt, pcov = curve_fit(parabola, xs, ys, sigma = ys_err, absolute_sigma = True)
    perr = np.sqrt(np.diag(pcov))

    results[(rho_min, rho_max)] = (popt, perr)

    # plot 
    xfit = np.linspace(-0.5, 0.5, 1000)
    yfit = parabola(xfit, *popt)
    ax.plot(xfit, yfit, label = f"Fit: {popt[0]:.3g}x^2 + {popt[1]:.3g}x + {popt[2]:.3g}", color = "red")

    ax.set_xlabel(r"$\Delta \phi$")
    ax.set_ylabel(r"$\Delta TOF$ [ns]")
    ax.set_title(f"$\Delta TOF$ vs $\Delta\phi$ ({rho_min} < Rconv < {rho_max} cm)", pad = 40)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.01, 0.2)
    ax.legend()

fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_dphi_scatter_profile.png")
fig.savefig(f"{local_folder}/mtd_deltaTOF_vs_dphi_scatter_profile.pdf")

# Resolution test: apply correction as fit above and check resolution
def parabola_ak(x, pars, axis = 2):
    x_arr = ak.concatenate([ak.singletons(x)**2, ak.singletons(x), ak.ones_like(ak.singletons(x))], axis = axis) # has (x**2, x, 2) for each element, should be directly multipliable with parabola coefficients
    par_arr = ak.firsts(pars, axis = axis) #first values are parameter values, seconds are errors
    return ak.sum(x_arr * par_arr, axis = axis)

mask1 = (ele1.typeClus == 0)
mask2 = (ele2.typeClus == 0)
photon_mask = ak.sum(ele1.typeClus == 0, axis = 2) + ak.sum(ele2.typeClus == 0, axis = 2) > 0

t0_ele1 = ak.firsts(tmtd_1[mask1], axis = 2) - np.sqrt((ak.firsts(xmtd_1[mask1], axis = 2) - genPV.x)**2 + (ak.firsts(ymtd_1[mask1], axis = 2) - genPV.y)**2 + (ak.firsts(zmtd_1[mask1], axis = 2) - genPV.z)**2) / c
t0_ele2 = ak.firsts(tmtd_2[mask2], axis = 2) - np.sqrt((ak.firsts(xmtd_2[mask2], axis = 2) - genPV.x)**2 + (ak.firsts(ymtd_2[mask2], axis = 2) - genPV.y)**2 + (ak.firsts(zmtd_2[mask2], axis = 2) - genPV.z)**2) / c

dphi1 = angle_to_mpi_pi(np.arctan2(ak.firsts(ymtd_1[mask1], axis = 2), ak.firsts(xmtd_1[mask1], axis = 2)) - conv_photons.phi)
dphi2 = angle_to_mpi_pi(np.arctan2(ak.firsts(ymtd_2[mask2], axis = 2), ak.firsts(xmtd_2[mask2], axis = 2)) - conv_photons.phi)
# PROBLEM: dimension mismatch might have to do with >1 direct cluster per photon

photon_mask_1 = ak.sum(mask1, axis = 2) > 0 #only select photons where ele1 has at least one direct clusters [will be considering the first one only anyway]
photon_mask_2 = ak.sum(mask2, axis = 2) > 0

# 1. for each cluster, determine rho bin
rho_bins_1 = ak.unflatten(np.digitize(ak.flatten(rho[photon_mask_1]), rho_range), ak.num(rho[photon_mask_1])) - 1
rho_bins_2 = ak.unflatten(np.digitize(ak.flatten(rho[photon_mask_2]), rho_range), ak.num(rho[photon_mask_2])) - 1

# 2. turn results into an array where the i-th element corresponds to the fit results for the i-th rho bin
fit_results = np.array([results[(rho_min, rho_max)] for rho_min, rho_max in zip(rho_range[:-1], rho_range[1:])]) #define in order so it can be sliced with ints

coeffs_1 = ak.unflatten(fit_results[ak.flatten(rho_bins_1)], ak.num(rho[photon_mask_1]))
coeffs_2 = ak.unflatten(fit_results[ak.flatten(rho_bins_2)], ak.num(rho[photon_mask_2]))

# remove None values from cluster quantities -- they fuck up dimension matching with singletons. They come from using ak.firsts
t0_ele1 = t0_ele1[~ak.is_none(t0_ele1, axis = 1)]
t0_ele2 = t0_ele2[~ak.is_none(t0_ele2, axis = 1)]

tof_corr_1 = parabola_ak(dphi1[~ak.is_none(dphi1, axis = 1)], coeffs_1, axis = 2)
tof_corr_2 = parabola_ak(dphi2[~ak.is_none(dphi2, axis = 1)], coeffs_2, axis = 2)

t0_res_1 = t0_ele1 - tof_corr_1 - conv_photons[photon_mask_1].t0 * 1e9
t0_res_2 = t0_ele2 - tof_corr_2 - conv_photons[photon_mask_2].t0 * 1e9
t0_res_precorr_1 = t0_ele1 - conv_photons[photon_mask_1].t0 * 1e9
t0_res_precorr_2 = t0_ele2 - conv_photons[photon_mask_2].t0 * 1e9

# Make resolution histograms
nrows = 3
ncols = len(rho_range)
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

ax = axs[0, 0]
plot_range = [-0.05, 0.2]
plot_hist_from_array(ak.concatenate([ak.ravel(tof_corr_1), ak.ravel(tof_corr_2)]), ax = ax, label = "All conversions", histtype = "step", bins = 100, range = plot_range)
plot_hist_from_array(ak.concatenate([ak.ravel(tof_corr_1[rho[photon_mask_1] > 112]), ak.ravel(tof_corr_2[rho[photon_mask_2] > 112])]), ax = ax, label = "MTD conversions", histtype = "step", linestyle = "dashed", alpha = 0.6, color = "C0", bins = 100, range = plot_range)

ax.set_xlabel(r"Time correction [ns]")
ax.set_ylabel("Entries")
ax.set_title("Time correction only")
ax.set_yscale("log")
ax.legend()
ax.grid()

# plot resolution broken down by conversion radius range
### BOOKMARK 2: resolution
for i in range(len(rho_range)):
    for j in range(1, 3):
        ax = axs[j, i]

        rho_min = 0 if i == 0 else rho_range[i-1]
        rho_max = rho_range[-1] if i == 0 else rho_range[i]

        rho_mask_1 = (rho[photon_mask_1] > rho_min) * (rho[photon_mask_1] < rho_max)
        rho_mask_2 = (rho[photon_mask_2] > rho_min) * (rho[photon_mask_2] < rho_max)

        plot_hist_from_array(ak.concatenate([ak.ravel(t0_res_1[rho_mask_1]), ak.ravel(t0_res_2[rho_mask_2])]), ax = ax, label = "Corrected", histtype = "step", bins = 100, range = (-0.05, 0.05), print_stats = False)
        plot_hist_from_array(ak.concatenate([ak.ravel(t0_res_precorr_1[rho_mask_1]), ak.ravel(t0_res_precorr_2[rho_mask_2])]), ax = ax, label = "NO corrections", histtype = "step", bins = 100, range = (-0.05, 0.05), print_stats = False)
        
        ax.axvline(0, color = "red", linestyle = "dashed", label = r"$\Delta t$ = 0")

        ax.set_xlabel(r"$\Delta t$ [ns]")
        ax.set_ylabel("Entries")
        rho_range_text = f"{rho_min} < Rconv < {rho_max} cm"
        if i == 0:
            rho_range_text = "all conversions"
        ax.set_title(f"Time resolution, {rho_range_text}")
        ax.legend()
        ax.grid()

        if j == 2:
            ax.set_yscale("log")

fig.savefig(f"{local_folder}/t0_resolution.png")
fig.savefig(f"{local_folder}/t0_resolution.pdf")

# ---- DEBUGGING RESOLUTION PROBLEM ----- #

rho_mins = [80, 114]
rho_maxs = [110, 118]

for rho_min, rho_max in zip(rho_mins, rho_maxs):

    # print(f"LOOKING AT EVENTS WITH {rho_min} < Rconv < {rho_max}\n")

    bad_events_1 = (rho > rho_min) * (rho < rho_max)
    bad_events_2 = (rho > rho_min) * (rho < rho_max)

    nrows = 10
    ncols = 3
    fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

    ax = axs[0, 0]

    mask1 = (ele1.typeClus == 0) * (dphi1_photon_all >= -0.05) * (dphi1_photon_all < 0.05)
    mask2 = (ele2.typeClus == 0) * (dphi2_photon_all >= -0.05) * (dphi2_photon_all < 0.05)

    arr1 = ak.ravel(tmtd_1[mask1 * bad_events_1])
    arr2 = ak.ravel(tmtd_2[mask2 * bad_events_2])
    arr_tmtd = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
    ax.hist(arr_tmtd.to_numpy(), bins = 100, histtype = "step", label = r"$|\Delta\phi| < 0.05$", range = (3, 8))
    ax.set_xlabel(r"$t_\text{MTD}$ [ns]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ax = axs[0, 1]
    tof1 = np.sqrt((xmtd_1 - genPV.x)**2 + (ymtd_1 - genPV.y)**2 + (zmtd_1 - genPV.z)**2) / c
    tof2 = np.sqrt((xmtd_2 - genPV.x)**2 + (ymtd_2 - genPV.y)**2 + (zmtd_2 - genPV.z)**2) / c

    suspect_events_1 = tof1 < 4
    suspect_events_2 = tof2 < 4

    tof1 = np.sqrt((xmtd_1[mask1 * bad_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1] - genPV.z)**2) / c
    tof2 = np.sqrt((xmtd_2[mask2 * bad_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2] - genPV.z)**2) / c
    arr_tof = ak.concatenate([ak.ravel(tof1), ak.ravel(tof2)])

    ax.hist(arr_tof.to_numpy(), bins = 100, histtype = "step", label = r"$|\Delta\phi| < 0.05$", range = (3, 8))
    # fill area under hist below 4 ns
    count, bins = np.histogram(arr_tof.to_numpy(), bins = 100, range = (3, 8))
    # make histogram out of the bins below 4 ns
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = bins[1:] - bins[:-1]
    # remove bins above 4 ns
    count = count[bin_centers < 4]
    bin_widths = bin_widths[bin_centers < 4]
    bin_centers = bin_centers[bin_centers < 4]
    # # draw histogram
    # ax.bar(bin_centers, count, width = bin_widths, align = "center", alpha = 0.5, color = "C0", label = "TOF < 4 ns (def.)")
    # ax.fill_between(hist[1][:-1], hist[0], where = (hist[1][:-1] < 4), alpha = 0.5, color = "C0", label = "Suspect events")

    ax.set_xlabel(r"$\overline{\text{TOF}}$ [ns]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ax = axs[0, 2]
    arr = genPV.t[ak.sum(bad_events_1, axis = 1) > 0]
    ax.hist(arr, bins = 100, histtype = "step", label =  r"$|\Delta\phi| < 0.05$")
    ax.set_xlabel(r"$t_\text{genPV}$ [ns]")
    ax.set_ylabel("Entries")
    ax.grid()
    ax.legend()

    ax = axs[1, 0]
    # plot tmtd vs tof
    h = ax.hist2d(arr_tmtd.to_numpy(), arr_tof.to_numpy(), bins = 100, label =  r"$|\Delta\phi| < 0.05$", range = ((3, 8), (3, 8)))
    fig.colorbar(h[3], ax = ax)
    ax.set_xlabel(r"$t_\text{MTD}$ [ns]")
    ax.set_ylabel(r"$\overline{\text{TOF}}$ [ns]")
    ax.set_title("tMTD vs TOF")

    ax = axs[1, 1]
    # bring it all together
    arr1 = tmtd_1[mask1 * bad_events_1] - np.sqrt((xmtd_1[mask1 * bad_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_1, axis = 1) > 0]
    arr2 = tmtd_2[mask2 * bad_events_2] - np.sqrt((xmtd_2[mask2 * bad_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_2, axis = 1) > 0]

    awful_events_1 = arr1 > 0.1
    awful_events_2 = arr2 > 0.1

    arr = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
    ax.hist(arr.to_numpy(), bins = 30, histtype = "step", label =  r"$|\Delta\phi| < 0.05$", range = (-0.1, 1))
    arr1_sus = tmtd_1[mask1 * bad_events_1 * suspect_events_1] - np.sqrt((xmtd_1[mask1 * bad_events_1 * suspect_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1 * suspect_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1 * suspect_events_1] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_1, axis = 1) > 0]
    arr2_sus = tmtd_2[mask2 * bad_events_2 * suspect_events_2] - np.sqrt((xmtd_2[mask2 * bad_events_2 * suspect_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2 * suspect_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2 * suspect_events_2] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_2, axis = 1) > 0]
    arr_suspect = ak.concatenate([ak.ravel(arr1_sus), ak.ravel(arr2_sus)])
    # ax.hist(arr_suspect.to_numpy(), bins = 30, range = (-0.1, 1), histtype = "step", label = "+ TOF < 4 ns", linestyle = "dashed", alpha = 0.6, color = "C0")
    ax.set_xlabel(r"$\Delta t$ [ns]")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title(r"Uncorrected resolution (aka $\Delta TOF$)", pad = 10)
    ax.grid()
    ax.legend()

    ax = axs[1, 2]
    # plot difference of tmtd and tof
    arr = arr_tmtd - arr_tof
    ax.hist(arr.to_numpy(), bins = 100, histtype = "step", label =  r"$|\Delta\phi| < 0.05$", range = (-0.6, 0.6))
    ax.set_xlabel(r"$t_\text{MTD} - \overline{\text{TOF}}$")
    ax.set_ylabel("Entries")
    ax.set_title("Uncorrected time @ vtx (compare to tGenPV)", pad = 10)
    ax.grid()
    ax.legend()

    ### go back to previous plots and plot awful events distribution
    ax = axs[0, 0]
    arr1 = ak.ravel(tmtd_1[mask1 * bad_events_1][awful_events_1])
    arr2 = ak.ravel(tmtd_2[mask2 * bad_events_2][awful_events_2])
    arr_tmtd_awful = ak.concatenate([arr1, arr2])
    ax.hist(arr.to_numpy(), bins = 100, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", range = (3, 8))
    ax.legend()

    ax = axs[0, 1]
    tof1_awful = np.sqrt((xmtd_1[mask1 * bad_events_1][awful_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1][awful_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1][awful_events_1] - genPV.z)**2) / c
    tof2_awful = np.sqrt((xmtd_2[mask2 * bad_events_2][awful_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2][awful_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2][awful_events_2] - genPV.z)**2) / c
    arr_tof_awful = ak.concatenate([ak.ravel(tof1_awful), ak.ravel(tof2_awful)])
    ax.hist(arr_tof_awful.to_numpy(), bins = 100, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", range = (3, 8))
    ax.legend()

    ax = axs[0, 2]
    arr1 = genPV.t[(ak.sum(bad_events_1, axis = 1) > 0) * (ak.sum(ak.sum(awful_events_1, axis = 2), axis = 1) > 0)]
    arr2 = genPV.t[(ak.sum(bad_events_2, axis = 1) > 0) * (ak.sum(ak.sum(awful_events_2, axis = 2), axis = 1) > 0)]
    arr = ak.concatenate([arr1, arr2])
    ax.hist(arr, bins = 100, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$")
    ax.legend()

    ax = axs[1, 2]
    arr = arr_tmtd_awful - arr_tof_awful
    ax.hist(arr.to_numpy(), bins = 100, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", range = (-0.6, 0.6))

    # # print event ID of awful events
    # print("Awful events:")
    # print(evts.event[(ak.sum(ak.sum(awful_events_1, axis = 2), axis = 1) + ak.sum(ak.sum(awful_events_2, axis = 2), axis = 1)) > 0])
    # # print cluster time of incriminated events
    # for evt in evts.event[(ak.sum(ak.sum(awful_events_1, axis = 2), axis = 1) + ak.sum(ak.sum(awful_events_2, axis = 2), axis = 1)) > 0][:5]:
    #     print(f"Event {evt}:")
    #     print(f"GenPV time: {genPV.t[evts.event == evt]}")
    #     # print(f"MTD clusters: ele1 : {tmtd_1[evts.event == evt]}, ele2 : {tmtd_2[evts.event == evt]}")
    #     print(f"MTD cluster types: ele1 : {ele1.typeClus[evts.event == evt]}, ele2 : {ele2.typeClus[evts.event == evt]}")
    #     all_info_1 = ak.zip({"t" : tmtd_1[evts.event == evt], "x": xmtd_1[evts.event == evt], "y" : ymtd_1[evts.event == evt], "z" : zmtd_1[evts.event == evt]})
    #     all_info_2 = ak.zip({"t" : tmtd_2[evts.event == evt], "x": xmtd_2[evts.event == evt], "y" : ymtd_2[evts.event == evt], "z" : zmtd_2[evts.event == evt]})
    #     print("\tEle1:")
    #     for event in all_info_1:
    #         for j, photon in enumerate(event):
    #             print(f"\tPhoton {j}:")
    #             for idx, cluster in enumerate(photon):
    #                 print(f"\t\tCluster {idx}: {cluster}")
    #     print("\tEle2:")
    #     for event in all_info_2:
    #         for j, photon in enumerate(event):
    #             print(f"\tPhoton {j}:")
    #             for idx, cluster in enumerate(photon):
    #                 print(f"\t\tCluster {idx}: {cluster}")
    #     # print(f"MTD cluster position (x): ele1 : {xmtd_1[evts.event == evt]}, ele2 : {xmtd_2[evts.event == evt]}")
    #     # print(f"MTD cluster position (y): ele1 : {ymtd_1[evts.event == evt]}, ele2 : {ymtd_2[evts.event == evt]}")
    #     # print(f"MTD cluster position (z): ele1 : {zmtd_1[evts.event == evt]}, ele2 : {zmtd_2[evts.event == evt]}")
    #     arr1 = tmtd_1[mask1 * bad_events_1] - np.sqrt((xmtd_1[mask1 * bad_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_1, axis = 1) > 0]
    #     arr2 = tmtd_2[mask2 * bad_events_2] - np.sqrt((xmtd_2[mask2 * bad_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_2, axis = 1) > 0]
    #     print(f"Uncorrected resolution: ele1 : {arr1[evts.event == evt]}, ele2 : {arr2[evts.event == evt]}")
    #     print(f"TOF: ele1 = {tof1[evts.event == evt]}, ele2 = {tof2[evts.event == evt]}")

    ## Geometry of bad clusters
    # plot x_mtd - gen PV x
    ax = axs[2, 0]
    arr1 = ak.ravel(xmtd_1[mask1 * bad_events_1] - genPV.x)
    arr2 = ak.ravel(xmtd_2[mask2 * bad_events_2] - genPV.x)
    arr = ak.concatenate([arr1, arr2])
    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)

    arr1_sus = ak.ravel(xmtd_1[mask1 * bad_events_1][awful_events_1])
    arr2_sus = ak.ravel(xmtd_2[mask2 * bad_events_2][awful_events_2])
    arr_suspect = ak.concatenate([arr1_sus, arr2_sus])
    ax.hist(arr_suspect.to_numpy(), bins = 50, histtype = "step", linestyle = "dashed", label = r"$|\Delta\phi| < 0.05$ (MTD x only)", alpha = 0.8, density = True)

    ax.set_xlabel(r"$x_\text{MTD} - x_\text{genPV}$ [cm]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ax = axs[2, 1]
    # plot y_mtd - gen PV y
    arr1 = ak.ravel(ymtd_1[mask1 * bad_events_1] - genPV.y)
    arr2 = ak.ravel(ymtd_2[mask2 * bad_events_2] - genPV.y)
    arr = ak.concatenate([arr1, arr2])
    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)

    arr1_sus = ak.ravel(ymtd_1[mask1 * bad_events_1][awful_events_1])
    arr2_sus = ak.ravel(ymtd_2[mask2 * bad_events_2][awful_events_2])
    arr_suspect = ak.concatenate([arr1_sus, arr2_sus])
    ax.hist(arr_suspect.to_numpy(), bins = 50, histtype = "step", linestyle = "dashed", label = r"$|\Delta\phi| < 0.05$ (MTD y only)", alpha = 0.8, density = True)

    ax.set_xlabel(r"$y_\text{MTD} - y_\text{genPV}$ [cm]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ax = axs[2, 2]
    # plot z_mtd - gen PV z
    arr1 = ak.ravel(zmtd_1[mask1 * bad_events_1] - genPV.z)
    arr2 = ak.ravel(zmtd_2[mask2 * bad_events_2] - genPV.z)
    arr = ak.concatenate([arr1, arr2])
    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True, range = [-200, 200])

    arr1_sus = ak.ravel(zmtd_1[mask1 * bad_events_1][awful_events_1])
    arr2_sus = ak.ravel(zmtd_2[mask2 * bad_events_2][awful_events_2])
    arr_suspect = ak.concatenate([arr1_sus, arr2_sus])
    ax.hist(arr_suspect.to_numpy(), bins = 50, histtype = "step", linestyle = "dashed", label = r"$|\Delta\phi| < 0.05$ (MTD z only)", alpha = 0.8, density = True)

    ax.set_xlabel(r"$z_\text{MTD} - z_\text{genPV}$ [cm]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ax = axs[3, 0]
    # plot eta of MTD clusters for suspect events and all bad events
    plot_range = [-1.5, 1.5]
    x1 = ak.ravel(xmtd_1[mask1 * bad_events_1])
    y1 = ak.ravel(ymtd_1[mask1 * bad_events_1])
    z1 = ak.ravel(zmtd_1[mask1 * bad_events_1])
    eta1 = np.arcsinh(z1 / np.sqrt(x1**2 + y1**2))
    x2 = ak.ravel(xmtd_2[mask2 * bad_events_2])
    y2 = ak.ravel(ymtd_2[mask2 * bad_events_2])
    z2 = ak.ravel(zmtd_2[mask2 * bad_events_2])
    eta2 = np.arcsinh(z2 / np.sqrt(x2**2 + y2**2))
    arr = ak.concatenate([eta1, eta2])
    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)

    x1_sus = ak.ravel(xmtd_1[mask1 * bad_events_1][awful_events_1])
    y1_sus = ak.ravel(ymtd_1[mask1 * bad_events_1][awful_events_1])
    z1_sus = ak.ravel(zmtd_1[mask1 * bad_events_1][awful_events_1])
    eta1_sus = np.arcsinh(z1_sus / np.sqrt(x1_sus**2 + y1_sus**2))
    x2_sus = ak.ravel(xmtd_2[mask2 * bad_events_2][awful_events_2])
    y2_sus = ak.ravel(ymtd_2[mask2 * bad_events_2][awful_events_2])
    z2_sus = ak.ravel(zmtd_2[mask2 * bad_events_2][awful_events_2])
    eta2_sus = np.arcsinh(z2_sus / np.sqrt(x2_sus**2 + y2_sus**2))
    arr_suspect = ak.concatenate([eta1_sus, eta2_sus])
    ax.hist(arr_suspect.to_numpy(), bins = 50, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
    
    ax.set_xlabel(r"$\eta_\text{MTD}$")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    # plot rho
    ax = axs[3, 1]
    arr = ak.concatenate([ak.ravel(np.sqrt(x1**2 + y1**2)), ak.ravel(np.sqrt(x2**2 + y2**2))])
    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True, range = (115.4, 116.2))
    arr = ak.concatenate([ak.ravel(np.sqrt(x1_sus**2 + y1_sus**2)), ak.ravel(np.sqrt(x2_sus**2 + y2_sus**2))])
    ax.hist(arr.to_numpy(), bins = 50, range = (115.4, 116.2), histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
    ax.set_xlabel(r"$\rho_\text{MTD}$ [cm]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    # plot phi
    ax = axs[3, 2]
    arr = ak.concatenate([ak.ravel(np.arctan2(y1, x1)), ak.ravel(np.arctan2(y2, x2))])
    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True, range = (-np.pi, np.pi))
    arr = ak.concatenate([ak.ravel(np.arctan2(y1_sus, x1_sus)), ak.ravel(np.arctan2(y2_sus, x2_sus))])
    ax.hist(arr.to_numpy(), bins = 50, range = (-np.pi, np.pi), histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
    ax.set_xlabel(r"$\phi_\text{MTD}$")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ### Conversion position VS MTD position
    ax = axs[4, 0]
    # convert convRadius, convZ, convPhi to x, y, z
    conv_x = conv_photons.convRadius * np.cos(conv_photons.convPhi)
    conv_y = conv_photons.convRadius * np.sin(conv_photons.convPhi)
    conv_z = conv_photons.convZ

    distance_1 = np.sqrt((xmtd_1[mask1 * bad_events_1] - conv_x)**2 + (ymtd_1[mask1 * bad_events_1] - conv_y)**2 + (zmtd_1[mask1 * bad_events_1] - conv_z)**2)
    distance_2 = np.sqrt((xmtd_2[mask2 * bad_events_2] - conv_x)**2 + (ymtd_2[mask2 * bad_events_2] - conv_y)**2 + (zmtd_2[mask2 * bad_events_2] - conv_z)**2)
    arr = ak.concatenate([ak.ravel(distance_1), ak.ravel(distance_2)])

    # plot_range = (115 - rho_max , 135 - rho_min)
    if rho_max == 110:
        plot_range = (0, 60)
    else:
        plot_range = (0, 20)

    ax.hist(arr.to_numpy(), bins = 50, histtype = "step", label = r"$|\Delta\phi| < 0.05$", range = plot_range)

    distance_1_awful = np.sqrt((xmtd_1[mask1 * bad_events_1][awful_events_1] - conv_x)**2 + (ymtd_1[mask1 * bad_events_1][awful_events_1] - conv_y)**2 + (zmtd_1[mask1 * bad_events_1][awful_events_1] - conv_z)**2)
    distance_2_awful = np.sqrt((xmtd_2[mask2 * bad_events_2][awful_events_2] - conv_x)**2 + (ymtd_2[mask2 * bad_events_2][awful_events_2] - conv_y)**2 + (zmtd_2[mask2 * bad_events_2][awful_events_2] - conv_z)**2)

    arr_awful = ak.concatenate([ak.ravel(distance_1_awful), ak.ravel(distance_2_awful)])

    ax.hist(arr_awful.to_numpy(), bins = 50, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", range = plot_range)

    # draw vertical line at 3 cm
    if rho_min < 115 and rho_max > 115:
        ax.axvline(3, color = "red", linestyle = "dashed", label = "d(conv, MTD) = 3 cm")

    ax.set_xlabel(r"$d(conv, MTD)$ [cm]")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")

    ### go back to deltaTof plot and only select events with high d(conv, MTD)
    ax = axs[1, 1]
    far_events_1 = distance_1 > 3
    far_events_2 = distance_2 > 3

    arr1 = tmtd_1[mask1 * bad_events_1][far_events_1] - np.sqrt((xmtd_1[mask1 * bad_events_1][far_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1][far_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1][far_events_1] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_1, axis = 1) > 0]
    arr2 = tmtd_2[mask2 * bad_events_2][far_events_2] - np.sqrt((xmtd_2[mask2 * bad_events_2][far_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2][far_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2][far_events_2] - genPV.z)**2) / c - genPV.t#[ak.sum(bad_events_2, axis = 1) > 0]
    arr = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
    ax.hist(arr.to_numpy(), bins = 30, histtype = "step", label = "+ d(conv, MTD) > 3 cm", range = (-0.1, 1))
    ax.legend()

    ax = axs[0, 0]
    # plot tmtd
    arr1 = ak.ravel(tmtd_1[mask1 * bad_events_1][far_events_1])
    arr2 = ak.ravel(tmtd_2[mask2 * bad_events_2][far_events_2])
    arr = ak.concatenate([arr1, arr2])
    ax.hist(arr.to_numpy(), bins = 100, histtype = "step", label = "+ d(conv, MTD) > 3 cm", range = (3, 8))
    ax.legend()

    ax = axs[0, 1]
    # plot tof
    tof1 = np.sqrt((xmtd_1[mask1 * bad_events_1][far_events_1] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1][far_events_1] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1][far_events_1] - genPV.z)**2) / c
    tof2 = np.sqrt((xmtd_2[mask2 * bad_events_2][far_events_2] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2][far_events_2] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2][far_events_2] - genPV.z)**2) / c
    arr_tof = ak.concatenate([ak.ravel(tof1), ak.ravel(tof2)])
    ax.hist(arr_tof.to_numpy(), bins = 100, histtype = "step", label = "+ d(conv, MTD) > 3 cm", range = (3, 8))
    ax.legend()


    ### Conversion position VS MTD position: breakdown, deltaPhi
    ax = axs[5, 0]

    deltaPhi1 = angle_to_mpi_pi(np.arctan2(ymtd_1[mask1 * bad_events_1], xmtd_1[mask1 * bad_events_1]) - np.arctan2(conv_y, conv_x))
    deltaPhi2 = angle_to_mpi_pi(np.arctan2(ymtd_2[mask2 * bad_events_2], xmtd_2[mask2 * bad_events_2]) - np.arctan2(conv_y, conv_x))
    # deltaPhi1 = angle_to_mpi_pi(np.arctan2(ymtd_1[mask1 * bad_events_1], xmtd_1[mask1 * bad_events_1])[bad_photons_1] - np.arctan2(conv_y_1, conv_x_1))
    # deltaPhi2 = angle_to_mpi_pi(np.arctan2(ymtd_2[mask2 * bad_events_2], xmtd_2[mask2 * bad_events_2])[bad_photons_2] - np.arctan2(conv_y_2, conv_x_2))
    arr = ak.concatenate([ak.ravel(deltaPhi1), ak.ravel(deltaPhi2)])

    ax.hist(arr.to_numpy(), bins = 50, range = [-0.05, 0.05], histtype = "step", label = r"$|\Delta\phi| < 0.05$")

    deltaPhi1_awful = angle_to_mpi_pi(np.arctan2(ymtd_1[mask1 * bad_events_1][awful_events_1], xmtd_1[mask1 * bad_events_1][awful_events_1]) - np.arctan2(conv_y, conv_x))
    deltaPhi2_awful = angle_to_mpi_pi(np.arctan2(ymtd_2[mask2 * bad_events_2][awful_events_2], xmtd_2[mask2 * bad_events_2][awful_events_2]) - np.arctan2(conv_y, conv_x))

    arr_awful = ak.concatenate([ak.ravel(deltaPhi1_awful), ak.ravel(deltaPhi2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 50, range = [-0.05, 0.05], histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0")

    ax.set_xlabel(r"$\Delta\phi(conv, MTD)$")
    ax.set_ylabel("Entries")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")

    ### Conversion position VS MTD position: breakdown, deltaR
    ax = axs[5, 1]

    mtd_r1 = np.sqrt(xmtd_1[mask1 * bad_events_1]**2 + ymtd_1[mask1 * bad_events_1]**2)
    conv_r1 = np.sqrt(conv_x**2 + conv_y**2)
    mtd_r2 = np.sqrt(xmtd_2[mask2 * bad_events_2]**2 + ymtd_2[mask2 * bad_events_2]**2)
    conv_r2 = np.sqrt(conv_x**2 + conv_y**2)

    deltaR1 = mtd_r1 - conv_r1
    deltaR2 = mtd_r2 - conv_r2
    # deltaR1 = np.sqrt((xmtd_1[mask1 * bad_events_1] - conv_x)**2 + (ymtd_1[mask1 * bad_events_1] - conv_y)**2)
    # deltaR2 = np.sqrt((xmtd_2[mask2 * bad_events_2] - conv_x)**2 + (ymtd_2[mask2 * bad_events_2] - conv_y)**2)
    arr = ak.concatenate([ak.ravel(deltaR1), ak.ravel(deltaR2)])
    ax.hist(arr.to_numpy(), bins = 50, range = [0, 1.5], histtype = "step", label = r"$|\Delta\phi| < 0.05$")

    mtd_r1_awful = np.sqrt(xmtd_1[mask1 * bad_events_1][awful_events_1]**2 + ymtd_1[mask1 * bad_events_1][awful_events_1]**2)
    conv_r1_awful = np.sqrt(conv_x**2 + conv_y**2)
    mtd_r2_awful = np.sqrt(xmtd_2[mask2 * bad_events_2][awful_events_2]**2 + ymtd_2[mask2 * bad_events_2][awful_events_2]**2)
    conv_r2_awful = np.sqrt(conv_x**2 + conv_y**2)
    deltaR1_awful = mtd_r1_awful - conv_r1_awful
    deltaR2_awful = mtd_r2_awful - conv_r2_awful
    # deltaR1_awful = np.sqrt((xmtd_1[mask1 * bad_events_1][awful_events_1] - conv_x)**2 + (ymtd_1[mask1 * bad_events_1][awful_events_1] - conv_y)**2)
    # deltaR2_awful = np.sqrt((xmtd_2[mask2 * bad_events_2][awful_events_2] - conv_x)**2 + (ymtd_2[mask2 * bad_events_2][awful_events_2] - conv_y)**2)
    arr_awful = ak.concatenate([ak.ravel(deltaR1_awful), ak.ravel(deltaR2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 50, range = [0, 1.5], histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0")

    ax.set_xlabel(r"$\Delta R(conv, MTD)$ [cm]")
    ax.set_ylabel("Entries")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")

    ### Conversion position VS MTD position: breakdown, deltaZ
    ax = axs[5, 2]
    deltaZ1 = np.abs(zmtd_1[mask1 * bad_events_1] - conv_z)
    deltaZ2 = np.abs(zmtd_2[mask2 * bad_events_2] - conv_z)
    # deltaZ1 = np.abs(zmtd_1[mask1 * bad_events_1][bad_photons_1] - conv_z_1)
    # deltaZ2 = np.abs(zmtd_2[mask2 * bad_events_2][bad_photons_2] - conv_z_2)
    arr = ak.concatenate([ak.ravel(deltaZ1), ak.ravel(deltaZ2)])
    ax.hist(arr.to_numpy(), bins = 50, range = (0, 20), histtype = "step", label = r"$|\Delta\phi| < 0.05$")

    deltaZ1_awful = np.abs(zmtd_1[mask1 * bad_events_1][awful_events_1] - conv_z)
    deltaZ2_awful = np.abs(zmtd_2[mask2 * bad_events_2][awful_events_2] - conv_z)
    arr_awful = ak.concatenate([ak.ravel(deltaZ1_awful), ak.ravel(deltaZ2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 50, range = (0, 20), histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0")
    
    ax.set_xlabel(r"$\Delta z(conv, MTD)$ [cm]")
    ax.set_ylabel("Entries")
    ax.grid()   
    ax.legend()
    ax.set_yscale("log")

    # deltaEta
    ax = axs[6, 0]
    
    deltaEta1 = np.arcsinh(zmtd_1[mask1 * bad_events_1] / np.sqrt(xmtd_1[mask1 * bad_events_1]**2 + ymtd_1[mask1 * bad_events_1]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
    deltaEta2 = np.arcsinh(zmtd_2[mask2 * bad_events_2] / np.sqrt(xmtd_2[mask2 * bad_events_2]**2 + ymtd_2[mask2 * bad_events_2]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
    arr = ak.concatenate([ak.ravel(deltaEta1), ak.ravel(deltaEta2)])
    ax.hist(arr.to_numpy(), bins = 50, range = [-0.12, 0.12], histtype = "step", label = r"$|\Delta\phi| < 0.05$")

    deltaEta1_awful = np.arcsinh(zmtd_1[mask1 * bad_events_1][awful_events_1] / np.sqrt(xmtd_1[mask1 * bad_events_1][awful_events_1]**2 + ymtd_1[mask1 * bad_events_1][awful_events_1]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
    deltaEta2_awful = np.arcsinh(zmtd_2[mask2 * bad_events_2][awful_events_2] / np.sqrt(xmtd_2[mask2 * bad_events_2][awful_events_2]**2 + ymtd_2[mask2 * bad_events_2][awful_events_2]**2)) - np.arcsinh(conv_z / np.sqrt(conv_x**2 + conv_y**2))
    arr_awful = ak.concatenate([ak.ravel(deltaEta1_awful), ak.ravel(deltaEta2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 50, range = [-0.12, 0.12], histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0")

    ax.set_xlabel(r"$\Delta\eta(conv, MTD)$")
    ax.set_ylabel("Entries")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")

    ### Draw MTD hit position and conversion vertex for 4 awful events
    # draw MTD radius (filled circle from 115.4 to 116.2 cm)
    ax = axs[4, 1]

    mtd_ring = mpl.patches.Annulus((0, 0), 116.2, 116.2 - 115.4, color = "C0", alpha = 0.3, label = "MTD")
    ax.add_patch(mtd_ring)

    for i, evt in enumerate(evts.event[(ak.sum(ak.sum(awful_events_1, axis = 2), axis = 1) + ak.sum(ak.sum(awful_events_2, axis = 2), axis = 1)) > 0][:4]):
        # bring it all together
        arr1 = tmtd_1[mask1 * bad_events_1 * (evts.event == evt)] - np.sqrt((xmtd_1[mask1 * bad_events_1 * (evts.event == evt)] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1 * (evts.event == evt)] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1 * (evts.event == evt)] - genPV.z)**2) / c - genPV.t
        arr2 = tmtd_2[mask2 * bad_events_2 * (evts.event == evt)] - np.sqrt((xmtd_2[mask2 * bad_events_2 * (evts.event == evt)] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2 * (evts.event == evt)] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2 * (evts.event == evt)] - genPV.z)**2) / c - genPV.t
        awful_events_1_local = arr1 > 0.1
        awful_events_2_local = arr2 > 0.1

        bad_photons_1 = (evts.event == evt) * (ak.sum(awful_events_1_local, axis = 2) > 0)
        bad_photons_2 = (evts.event == evt) * (ak.sum(awful_events_2_local, axis = 2) > 0)

        # plot conversion vertex position
        conv_x_1 = conv_photons.convRadius[bad_photons_1] * np.cos(conv_photons.convPhi[bad_photons_1])
        conv_y_1 = conv_photons.convRadius[bad_photons_1] * np.sin(conv_photons.convPhi[bad_photons_1])
        conv_x_2 = conv_photons.convRadius[bad_photons_2] * np.cos(conv_photons.convPhi[bad_photons_2])
        conv_y_2 = conv_photons.convRadius[bad_photons_2] * np.sin(conv_photons.convPhi[bad_photons_2])

        # plot MTD cluster position
        ax.plot(ak.ravel(ak.concatenate([conv_x_1, conv_x_2])), ak.ravel(ak.concatenate([conv_y_1, conv_y_2])), "x", color = f"C{i}", label = f"Event {evt}")

        # plot MTD cluster position
        mtd_x_1 = xmtd_1[mask1 * bad_events_1 * (evts.event == evt)][awful_events_1_local]
        mtd_x_2 = xmtd_2[mask2 * bad_events_2 * (evts.event == evt)][awful_events_2_local]

        mtd_x = ak.concatenate([mtd_x_1, mtd_x_2])

        mtd_y_1 = ymtd_1[mask1 * bad_events_1 * (evts.event == evt)][awful_events_1_local]  
        mtd_y_2 = ymtd_2[mask2 * bad_events_2 * (evts.event == evt)][awful_events_2_local]

        mtd_y = ak.concatenate([mtd_y_1, mtd_y_2])

        # plot MTD cluster position
        ax.plot(ak.ravel(mtd_x), ak.ravel(mtd_y), "o", color = f"C{i}")

    ax.grid()
    ax.legend()
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)

    # z, rho view
    ax = axs[4, 2]

    # plot rectangles
    # mtd_rectangle_1 = mpl.patches.Rectangle((-200, 115.4), 400, 0.8, color = "C0", alpha = 0.3, label = "MTD")
    # mtd_rectangle_2 = mpl.patches.Rectangle((-200, -116.2), 400, 0.8, color = "C0", alpha = 0.3, label = "MTD")
    # ax.add_patch(mtd_rectangle_1)
    # ax.add_patch(mtd_rectangle_2)

    mtd_rectangle = mpl.patches.Rectangle((-200, 115.4), 400, 0.8, color = "C0", alpha = 0.3, label = "MTD")
    ax.add_patch(mtd_rectangle)

    for i, evt in enumerate(evts.event[(ak.sum(ak.sum(awful_events_1, axis = 2), axis = 1) + ak.sum(ak.sum(awful_events_2, axis = 2), axis = 1)) > 0][:4]):
        # bring it all together
        arr1 = tmtd_1[mask1 * bad_events_1 * (evts.event == evt)] - np.sqrt((xmtd_1[mask1 * bad_events_1 * (evts.event == evt)] - genPV.x)**2 + (ymtd_1[mask1 * bad_events_1 * (evts.event == evt)] - genPV.y)**2 + (zmtd_1[mask1 * bad_events_1 * (evts.event == evt)] - genPV.z)**2) / c - genPV.t
        arr2 = tmtd_2[mask2 * bad_events_2 * (evts.event == evt)] - np.sqrt((xmtd_2[mask2 * bad_events_2 * (evts.event == evt)] - genPV.x)**2 + (ymtd_2[mask2 * bad_events_2 * (evts.event == evt)] - genPV.y)**2 + (zmtd_2[mask2 * bad_events_2 * (evts.event == evt)] - genPV.z)**2) / c - genPV.t
        awful_events_1_local = arr1 > 0.1
        awful_events_2_local = arr2 > 0.1

        bad_photons_1 = (evts.event == evt) * (ak.sum(awful_events_1_local, axis = 2) > 0)
        bad_photons_2 = (evts.event == evt) * (ak.sum(awful_events_2_local, axis = 2) > 0)

        # plot conversion vertex position
        conv_x_1 = conv_photons.convRadius[bad_photons_1] * np.cos(conv_photons.convPhi[bad_photons_1])
        conv_y_1 = conv_photons.convRadius[bad_photons_1] * np.sin(conv_photons.convPhi[bad_photons_1])
        conv_r_1 = np.sqrt(conv_x_1**2 + conv_y_1**2)
        conv_z_1 = conv_photons.convZ[bad_photons_1]
        conv_x_2 = conv_photons.convRadius[bad_photons_2] * np.cos(conv_photons.convPhi[bad_photons_2])
        conv_y_2 = conv_photons.convRadius[bad_photons_2] * np.sin(conv_photons.convPhi[bad_photons_2])
        conv_r_2 = np.sqrt(conv_x_2**2 + conv_y_2**2)
        conv_z_2 = conv_photons.convZ[bad_photons_2]

        # plot MTD cluster position
        ax.plot(ak.ravel(ak.concatenate([conv_z_1, conv_z_2])), ak.ravel(ak.concatenate([conv_r_1, conv_r_2])), "x", color = f"C{i}", label = f"Event {evt}")

        # plot MTD cluster position
        mtd_x_1 = xmtd_1[mask1 * bad_events_1 * (evts.event == evt)][awful_events_1_local]
        mtd_x_2 = xmtd_2[mask2 * bad_events_2 * (evts.event == evt)][awful_events_2_local]

        mtd_y_1 = ymtd_1[mask1 * bad_events_1 * (evts.event == evt)][awful_events_1_local]  
        mtd_y_2 = ymtd_2[mask2 * bad_events_2 * (evts.event == evt)][awful_events_2_local]

        mtd_r_1 = np.sqrt(mtd_x_1**2 + mtd_y_1**2)
        mtd_r_2 = np.sqrt(mtd_x_2**2 + mtd_y_2**2)

        mtd_r = ak.concatenate([mtd_r_1, mtd_r_2])

        mtd_z_1 = zmtd_1[mask1 * bad_events_1 * (evts.event == evt)][awful_events_1_local]
        mtd_z_2 = zmtd_2[mask2 * bad_events_2 * (evts.event == evt)][awful_events_2_local]

        mtd_z = ak.concatenate([mtd_z_1, mtd_z_2])

        # plot MTD cluster position
        ax.plot(ak.ravel(mtd_z), ak.ravel(mtd_r), "o", color = f"C{i}")

    ax.grid()
    ax.legend()
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("R [cm]")
    ax.set_xlim(-200, 200)
    ax.set_ylim(115, 116.8)

    ### Plot ENERGY variables
    ax = axs[6, 1]
    plot_range = [0, 25]
    emtd_1 = ele1.energyClus[mask1 * bad_events_1]
    emtd_2 = ele2.energyClus[mask2 * bad_events_2]
    arr = ak.concatenate([ak.ravel(emtd_1), ak.ravel(emtd_2)])
    ax.hist(arr.to_numpy(), bins = 30, range = plot_range, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)
    emtd_1_awful = ele1.energyClus[mask1 * bad_events_1][awful_events_1]
    emtd_2_awful = ele2.energyClus[mask2 * bad_events_2][awful_events_2]
    arr_awful = ak.concatenate([ak.ravel(emtd_1_awful), ak.ravel(emtd_2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 30, range = plot_range, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
    ax.set_title("Cluster energy")
    ax.set_xlabel(r"$E_\text{MTD}$ [MeV]")
    # ax.set_ylabel("Entries")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")

    ### Plot z std. dev. (cluster extension)
    for var, ax in zip(["tStdClus", "xStdClus", "yStdClus", "zStdClus"], axs.flatten()[20:20+4]):        
        plot_range = [0, 0.15]
        if var == "zStdClus":
            plot_range = [0, 0.2]
        elif var == "xStdClus":
            plot_range = [0, 1]
        elif var == "yStdClus":
            plot_range = [0, 0.2]
        arr1 = ele1[var][mask1 * bad_events_1]
        arr2 = ele2[var][mask2 * bad_events_2]
        arr = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
        ax.hist(arr.to_numpy(), bins = 30, range = plot_range, histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)
        arr1_awful = ele1[var][mask1 * bad_events_1][awful_events_1]
        arr2_awful = ele2[var][mask2 * bad_events_2][awful_events_2]
        arr_awful = ak.concatenate([ak.ravel(arr1_awful), ak.ravel(arr2_awful)])
        ax.hist(arr_awful.to_numpy(), bins = 30, range = plot_range, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
        ax.set_title(f"Std. dev. of hits {var[0]} in cluster")
        ax.set_xlabel(f"$\sigma_{var[0]}$ [cm]")
        # ax.set_ylabel("Entries")
        ax.set_ylabel("Density")
        ax.grid()
        ax.legend()
        ax.set_yscale("log")
    
    ### Plot z std. dev. AMONG clusters
    for var, ax in zip(["tClus", "xClus", "yClus", "zClus"], axs.flatten()[24:24+4]):        
        plot_range = [0, 5]
        # plot_range = [0, 0.15]
        # if var == "zClus":
        #     plot_range = [0, 0.2]
        # elif var == "xClus":
        #     plot_range = [0, 1]
        # elif var == "yClus":
        #     plot_range = [0, 0.2]
        arr1 = ak.std(ele1[var][mask1 * bad_events_1], axis = 2)
        arr2 = ak.std(ele2[var][mask2 * bad_events_2], axis = 2)
        arr = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
        ax.hist(arr.to_numpy(), bins = 30, range = plot_range, histtype = "step", label = r"$|\Delta\phi| < 0.05$")
        arr1_awful = ak.std(ele1[var][mask1 * bad_events_1][awful_events_1], axis = 2)
        arr2_awful = ak.std(ele2[var][mask2 * bad_events_2][awful_events_2], axis = 2)
        arr_awful = ak.concatenate([ak.ravel(arr1_awful), ak.ravel(arr2_awful)])
        ax.hist(arr_awful.to_numpy(), bins = 30, range = plot_range, histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0")
        ax.set_title(f"Std. dev. of cluster {var[0]} per electron")
        ax.set_xlabel(f"$\sigma_{var[0]}$ [cm]")
        ax.set_ylabel("Entries")
        # ax.set_ylabel("Density")
        ax.grid()
        ax.legend()
        ax.set_yscale("log")

    ax = axs[9, 1]
    # #hits/cluster 
    arr1 = ele1.nHitsClus[mask1 * bad_events_1]
    arr2 = ele2.nHitsClus[mask2 * bad_events_2]
    arr = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
    ax.hist(arr.to_numpy(), bins = 6, range = (-0.5, 5.5), histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)
    arr1_awful = ele1.nHitsClus[mask1 * bad_events_1][awful_events_1]
    arr2_awful = ele2.nHitsClus[mask2 * bad_events_2][awful_events_2]
    arr_awful = ak.concatenate([ak.ravel(arr1_awful), ak.ravel(arr2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 6, range = (-0.5, 5.5), histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
    ax.set_title("Hits per cluster")
    ax.set_xlabel("Hits")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()

    ax = axs[9, 2]
    # #clusters/electron
    arr1 = ak.num(ele1.tClus[mask1 * bad_events_1], axis = 2)
    arr2 = ak.num(ele2.tClus[mask2 * bad_events_2], axis = 2)
    arr = ak.concatenate([ak.ravel(arr1), ak.ravel(arr2)])
    ax.hist(arr.to_numpy(), bins = 11, range = (-0.5, 10.5), histtype = "step", label = r"$|\Delta\phi| < 0.05$", density = True)
    arr1_awful = ak.num(ele1.tClus[mask1 * bad_events_1][awful_events_1], axis = 2)
    arr2_awful = ak.num(ele2.tClus[mask2 * bad_events_2][awful_events_2], axis = 2)
    arr_awful = ak.concatenate([ak.ravel(arr1_awful), ak.ravel(arr2_awful)])
    ax.hist(arr_awful.to_numpy(), bins = 11, range = (-0.5, 10.5), histtype = "step", label = r"+ $\Delta TOF > 0.1 ns$", linestyle = "dashed", alpha = 0.6, color = "C0", density = True)
    ax.set_title("Clusters per electron")
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Density")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")

    fig.savefig(f"{local_folder}/debugging_{rho_min}To{rho_max}.png")
    fig.savefig(f"{local_folder}/debugging_{rho_min}To{rho_max}.pdf")

# ---------------------

# percentage of photon conversion vs pt, eta
### PICK *ALL* EVENTS NOW, w/o conversion cut

photon_pt_sort = ak.argsort(evts.GENConvertedPhoton.pt, ascending = False)
pt_all = evts.GENConvertedPhoton.pt[photon_pt_sort][:,:2] # only two leading photons
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

for pt, eta, is_conversion, is_conversion_old, is_conversion_MTD in zip(ak.flatten(pt_all), ak.flatten(eta_all), ak.flatten(are_conversions), ak.flatten(are_conversions_old), ak.flatten(are_conversions_MTD)):
    conversion_rate_pt.Fill(1 * is_conversion, pt)
    conversion_rate_eta.Fill(1 * is_conversion, abs(eta))
    conversion_rate_pt_old.Fill(1 * is_conversion_old, pt)
    conversion_rate_eta_old.Fill(1 * is_conversion_old, abs(eta))
    conversion_rate_pt_MTD.Fill(1 * is_conversion_MTD, pt)
    conversion_rate_eta_MTD.Fill(1 * is_conversion_MTD, abs(eta))

# plot efficiency
fig, axs = plt.subplots(2, 2, figsize = (11, 12))

### Conversion rate vs pt
ax = axs[0, 0]

# # plot pT distribution (normalized to 0.5)
# counts, bins = np.histogram(ak.ravel(pt_all).to_numpy(), bins = 50, range = (0, 10))
# counts = counts/(counts.max() * 1.5)
# bins_mid = (bins[1:] + bins[:-1])/2
# bin_width = bins[1] - bins[0]

# # add second axis on right side for pT distribution
# ax2 = ax.twinx()
# lns1 = ax2.bar(bins_mid, counts, width = bin_width, alpha = 0.2, color = "C0", label = "All photons")
# ax2.set_ylim(0, 1)
# ax2.set_ylabel("A.U.")
# ax2.spines['right'].set_color('C0')
# ax2.yaxis.label.set_color('C0')
# ax2.tick_params(axis='y', colors='C0')

# plot efficiencies
plot_tefficiency(conversion_rate_pt, ax = ax, label = "All conversions")
plot_tefficiency(conversion_rate_pt_MTD, ax = ax, label = "MTD conversions")

ax.set_xlabel("pT [GeV]")
ax.set_title("Conversion rate vs pT (Rconv < 116.1 cm)", pad = 20)

# # combine legends
# ## retrieve legend entries in ax
# handles, labels = ax.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# lns = handles + handles2
# labs = labels + labels2
# ax.legend(lns, labs)

ax.legend()

ax.grid()
ax.set_ylim(0, 1)

### Conversion rate vs eta
ax = axs[0, 1]

# # plot pT distribution (normalized to 0.5)
# counts, bins = np.histogram(ak.ravel(eta_all).to_numpy(), bins = 50, range = (0, 1.2))
# counts = counts/(counts.max() * 1.5)
# bins_mid = (bins[1:] + bins[:-1])/2
# bin_width = bins[1] - bins[0]

# # add second axis on right side for pT distribution
# ax2 = ax.twinx()
# lns1 = ax2.bar(bins_mid, counts, width = bin_width, alpha = 0.2, color = "C0", label = "All photons")
# ax2.set_ylim(0, 1)
# ax2.set_ylabel("A.U.")
# ax2.spines['right'].set_color('C0')
# ax2.yaxis.label.set_color('C0')
# ax2.tick_params(axis='y', colors='C0')

plot_tefficiency(conversion_rate_eta, ax = ax, label = "All conversions")
plot_tefficiency(conversion_rate_eta_MTD, ax = ax, label = "MTD conversions")
ax.set_xlabel(r"$|\eta|$")
ax.set_title("Conversion rate vs $\eta$ (Rconv < 116.1 cm)", pad = 20)
ax.grid()
ax.set_ylim(0, 1)

# # merge legends
# handles, labels = ax.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# lns = handles + handles2
# labs = labels + labels2
# ax.legend(lns, labs)

ax.legend()

### OLD DEFINITION: conversion rate vs pt
ax = axs[1, 0]
plot_tefficiency(conversion_rate_pt_old, ax = ax)
ax.set_xlabel("pT [GeV]")
ax.set_title("Conversion rate vs pT (Rconv < 118, pt > 1)", pad = 20)
ax.grid()
ax.set_ylim(0, 1)

### OLD DEFINITION: conversion rate vs eta
ax = axs[1, 1]
plot_tefficiency(conversion_rate_eta_old, ax = ax)
ax.set_xlabel(r"$|\eta|$")
ax.set_title("Conversion rate vs $\eta$ (Rconv < 118, pt > 1)", pad = 20)
ax.grid()
ax.set_ylim(0, 1)

plt.tight_layout()
fig.savefig(f"{local_folder}/conversion_rate.png")
fig.savefig(f"{local_folder}/conversion_rate.pdf")

# --------------------------- # 

# ------- RECO STUFF -------- #

# 1. rate of MTD hit matching
match_mask = mtd_hit.drMatch < 1e5
matched_reco_hit = ak.sum(match_mask) / ak.sum(ak.num(mtd_hit))
printlog(f"For converted photons, rate of MTD hit matching: {matched_reco_hit * 100:.2f}%", f)

# 2. dR distribution
nrows = 2
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize = (12*ncols, 12*nrows))

ax = axs[0, 0]
ax.hist(ak.ravel(mtd_hit[match_mask].drMatch), bins = 100, histtype = "step", range = [0, 0.5], label = "All")
ax.hist(ak.ravel(mtd_hit[match_mask * rho < 114].drMatch), bins = 100, histtype = "step", range = [0, 0.5], label = "Tracker conversions")
ax.hist(ak.ravel(mtd_hit[match_mask * rho > 114].drMatch), bins = 100, histtype = "step", range = [0, 0.5], label = "MTD conversions")

ax.set_title(r"$\Delta R(\gamma, hit)$ for converted photons")
ax.set_xlabel(r"$\Delta R(\gamma, hit)$")
ax.set_ylabel("Entries")
ax.legend()
ax.grid()

# 3. time resolution of MTD hits
ax = axs[0, 1]

res1 = mtd_hit[match_mask * good_cluster_photons].time - selected1(tmtd_1)[match_mask[good_cluster_photons]] # check if this is correct
res2 = mtd_hit[match_mask * good_cluster_photons].time - selected2(tmtd_2)[match_mask[good_cluster_photons]]

ax.hist(ak.ravel(res1), bins = 100, histtype = "step", range = [-0.5, 0.5], label = "Res. wrt leading")
ax.hist(ak.ravel(res1), bins = 100, histtype = "step", range = [-0.5, 0.5], label = "Res. wrt subleading")

ax.set_title("Time resolution at MTD of MTD hits (conversions with DIRECT HITS only)")
ax.set_xlabel(r"$\Delta t(RECO, SIM)$ [ns]")
ax.set_ylabel("Entries")
ax.legend()
ax.grid()

## saving
fig.savefig(f"{local_folder}/mtd_hit_matching.png")
fig.savefig(f"{local_folder}/mtd_hit_matching.pdf")

# --------------------------- #

# close all figs
plt.close("all")

# copy plots to EOS area
outfolder = "0p1To10" if args.use_low_pt else "8To150"
eosfolder = f"/eos/home-n/npalmeri/www/MTD/PhotonReco/GEN_level/{outfolder}"
os.system(f"cp {local_folder}/*.png %s" % eosfolder)
os.system(f"cp {local_folder}/*.pdf %s" % eosfolder)

# also copy log
f.close()
os.system("cp conversions.log %s" % eosfolder)