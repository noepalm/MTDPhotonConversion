import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
import uproot
import mplhep as hep
import ROOT
import awkward as ak

# ---------- PLOTTING SETTINGS ---------

# Load CMS style including color-scheme
hep.style.use("CMS")
plt.style.use(hep.style.CMS)

# Styling options
mpl.rcParams['patch.linewidth'] = 2 #for step hist plots

plt.rcParams.update({
    "text.usetex": False,
})

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# ------------ UTILITIES ------------

def plot_hist_from_array(array, ax, bins = None, range = None, print_stats = True, flow = False, xlog = False, **kwargs):
    arr = array
    
    # flatten 
    try: 
        arr = ak.flatten(arr, axis = None)
    except: 
        print("Flattening error")
        pass

    # turn to numpy array if necessary
    try:
        arr = ak.to_numpy(arr)
    except:
        print("Error converting to numpy")
        pass

    # compute bins manually
    if isinstance(bins, int):
        if range is None:
            range = [np.min(arr), np.max(arr)]
        if xlog:
            bin_edges = np.logspace(*np.log10(range), bins + 1)
        else:
            bin_edges = np.linspace(*range, bins + 1)
    else:
        bin_edges = bins

    # add extremes for underflow/overflow computation
    bin_edges = [-np.inf, *bin_edges, np.inf]

    n, bins = np.histogram(arr, bins = bin_edges)
    underflow = n[0]
    overflow = n[-1]

    # if showing underflow, replace +-np.inf with reasonable extremes
    if flow:
        bins[0] = bins[1] - (bins[2] - bins[1])
        bins[-1] = bins[-2] + (bins[-2] - bins[-3])
    # otherwise, remove flow bins from plotting
    else:
        bins = bins[1:-1]
        n = n[1:-1]
        
    hep.histplot((n, bins), ax=ax, **kwargs)

    if flow:
        ax.axvline(bins[1], color = "black", linestyle = "dashed", alpha = 0.7)
        ax.axvline(bins[-2], color = "black", linestyle = "dashed", alpha = 0.7)

    if print_stats:
        # add stats box
        text = f"Entries = {np.sum(n):.0f}\nUnderflow = {underflow:.0f}\nOverflow = {overflow:.0f}"
        
        # Create an empty plot with the required text.
        ax.plot([], label=text, color="black", alpha=0)

# TODO: implement flow (i.e. add under/overflow bin grid)
def plot_2Dhist_from_array(arrx, arry, ax, bins = None, range = None, print_stats = True, **kwargs):
    
    # flatten 
    try: 
        arrx = ak.flatten(arrx)
        arry = ak.flatten(arry)
    except:
        pass
    
    # turn to numpy array if necessary
    try:
        arrx = ak.to_numpy(arrx)
        arry = ak.to_numpy(arry)
    except:
        pass

    # compute histogram
    histo_args = {"bins" : bins, "range" : range}
    n, binsx, binsy = np.histogram2d(arrx, arry, **histo_args)

    # plot
    hep.hist2dplot((n, binsx, binsy), ax=ax, flow = None, **kwargs)
    
    if print_stats:
        # add stats box
        text = f"Entries = {np.sum(n):.0f}"
        
        # Create an empty plot with the required text.
        ax.plot([], label=text, color="black", alpha=0)

def plot_profile(profile, ax, **kwargs):
    try:
        p = uproot.from_pyroot(profile)
    except:
        p = profile
    x_centers = p.axis(0).centers()
    x_errors = p.axis(0).widths()
    y_values = p.values()
    y_errors = p.errors(error_mode="s")
    ax.errorbar(x_centers, y_values, xerr=x_errors/2., yerr=y_errors, **kwargs)

def plot_asymm_graph(graph, ax, **kwargs):
    x_centers = graph.values()[0]
    x_errors_low = graph.members["fEXlow"]
    x_errors_high = graph.members["fEXhigh"]
    y_values = graph.values()[1]
    y_errors_low = graph.members["fEYlow"]
    y_errors_high = graph.members["fEYhigh"]

    x_errors = np.array([x_errors_low, x_errors_high])
    y_errors = np.array([y_errors_low, y_errors_high])

    ax.errorbar(x_centers, y_values, xerr=x_errors, yerr=y_errors, fmt='o', capsize = 2, **kwargs)

def plot_tefficiency(teff, **kwargs):
    eff = uproot.from_pyroot(teff.CreateGraph().Clone())
    plot_asymm_graph(eff, **kwargs)

def plot_hist_errors(hist, ax, **kwargs):
    try:
        h = uproot.to_pyroot(hist)
    except:
        h = hist

    Nvalues = h.GetNbinsX()
    x_centers = np.array([h.GetBinCenter(i) for i in range(1, Nvalues+1)])
    x_errors_low = np.array([h.GetBinWidth(i)/2 for i in range(1, Nvalues+1)])
    x_errors_high = np.array([h.GetBinWidth(i)/2 for i in range(1, Nvalues+1)])
    y_values = np.array([h.GetBinContent(i) for i in range(1, Nvalues+1)])
    y_errors_low = np.array([h.GetBinError(i) for i in range(1, Nvalues+1)])
    y_errors_high = np.array([h.GetBinError(i) for i in range(1, Nvalues+1)])

    x_errors = np.array([x_errors_low, x_errors_high])
    y_errors = np.array([y_errors_low, y_errors_high])

    # plot error bars without connecting line
    # ax.errorbar(x_centers, y_values, yerr = y_errors, **kwargs)
    p = ax.errorbar(x_centers, y_values, xerr = x_errors, yerr = y_errors, **kwargs)
    return p

def hist_mean_sigma(hist):
    n, bins = hist
    mids = (bins[1:] + bins[:-1]) / 2.
    mean = np.average(mids, weights=n)
    var = np.average((mids - mean)**2, weights=n)
    return (mean, np.sqrt(var))

def print_mean_sigma(label, hist):
    try:
        mean, sigma = hist_mean_sigma(hist)
    except:
        return label
    new_label = label + "\n$\mu \pm \sigma = {:.4f} \pm {:.4f}$".format(mean, sigma)
    return new_label

def print_entries(label, hist):
    return label + "\nEntries = {}".format(hist[0].sum())

def darker(color, scale=1., alpha=1.):
    new_color = np.array(to_rgba(color, alpha)) - np.array([0.1, 0.1, 0.1, 0])*scale
    new_color[new_color < 0] = 0
    new_color[new_color > 1] = 1
    return tuple(new_color)



# ---------- FIT UTILITIES ---------

# TODO: implement this in a class
#       wanted features:
#           - make interactive object so that you can set xmin, xmax stuff as obj.set_lims(...)
#           - (...basically a wrapper to RooFit really)

def fit_histo(pyroot_histo, xmin, xmax):
    g1 = ROOT.TF1("g1", "[0]*exp(-0.5*((x-[1])/[2])*((x-[1])/[2]))", xmin, xmax)
    hist = uproot.to_pyroot(pyroot_histo)

    gsum  = ROOT.TF1("gsum", "g1(0)", xmin, xmax)
    gsum.SetParameters(10, 0, 1)

    # fit histogram without plotting
    hist.Fit(gsum, "R0") # 0 for not plotting

    # return function parameters
    outPars = {}
    parNames = ["const", "mean", "sigma"]
    fNames = ["gauss1", "gauss2"]
    for f in fNames:
        parDict = {}
        for idx, par in enumerate(parNames):
            parDict[par] = gsum.GetParameter(idx + 3*fNames.index(f))
        outPars[f] = parDict
    return outPars


def fit_histo_roofit(pyroot_histo, xmin, xmax):
    # define model
    x = ROOT.RooRealVar("x", "x", xmin, xmax)

    # n1 = ROOT.RooRealVar("n1", "n1", 200, 30, 500)
    # mean1 = ROOT.RooRealVar("mean1", "mean1", 0.5, -2, 2)
    # sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 1, 0.1, 5)
    # gauss1 = ROOT.RooGaussian("gauss1", "gauss1", x, mean1, sigma1)

    # n2 = ROOT.RooRealVar("n2", "n2", 20, 0, 500)
    # const = ROOT.RooUniform("const", "const", x)

    # model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(gauss1, const), ROOT.RooArgList(n1, n2))

    n1 = ROOT.RooRealVar("n1", "n1", 0.4, 0.01, 0.6)
    mean1 = ROOT.RooRealVar("mean1", "mean1", 0.5, -2, 2)
    sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 1, 0.1, 5)
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", x, mean1, sigma1)

    # model = ROOT.RooExtendPdf("egauss", "egauss", gauss1, n1)
    model = gauss1

    # prepare data
    hist = uproot.to_pyroot(pyroot_histo)
    data = ROOT.RooDataHist("data", "data", ROOT.RooArgList(x), hist)

    # fit data
    result = model.fitTo(data, ROOT.RooFit.Range(xmin, xmax), ROOT.RooFit.Save())

    outParsDict = {}
    parNames = ["const", "mean", "sigma"]

    # for idx, f in enumerate(["gauss1", "const"]):
    for idx, f in enumerate(["gauss1"]):
        parDict = {}
        parlist = [n1, mean1, sigma1] if f == "gauss1" else [n2]
        for parname, par in zip(parNames, parlist):
            # parDict[par] = result.floatParsFinal().find(f"{par}{idx}").getVal()
            parDict[parname] = par.getVal()
        outParsDict[f] = parDict

    return outParsDict    

def gauss(x, const, mean, sigma):
    return const * np.exp(-(x-mean)**2/(2*sigma**2))


# ------------------------------------------------------------------------------

# write main for library
if __name__ == "__main__":
    pass