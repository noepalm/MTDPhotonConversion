import subprocess
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('datasets',
                 '',
                 VarParsing.multiplicity.list,
                 VarParsing.varType.string,
                 "Input dataset(s)")
options.register('eosdirs',
                 '',
                 VarParsing.multiplicity.list,
                 VarParsing.varType.string,
                 "files location(s) on EOS")
options.register('eosusdirs',
                 '',
                 VarParsing.multiplicity.list,
                 VarParsing.varType.string,
                 "files location(s) on EOS-US")
options.register('pattern',
                 '',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "pattern of file names to be processed")
options.register('antipattern',
                 '',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "pattern of file names not to be processed")
options.register('output',
                 'test.root',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "output file name")
options.register('sample',
                 'lowPt',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,)
options.register('useparent',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Load data from parent datasets")
options.register('storeclusters',
                 True,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "store vector with all MTD cluster info")
options.register('debug',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Print debug messages")
options.register('crysLayout',
                 'barzflat',
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "crystal layout (tile, barphi, barzflat)")
options.register('nThreads',
                 1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "# threads")
                 
options.maxEvents = 100
options.parseArguments()

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('MTDNeutralsDumper',Phase2C17I13M9)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    numberOfThreads=cms.untracked.uint32(options.nThreads),
    numberOfStreams=cms.untracked.uint32(0),
    wantSummary = cms.untracked.bool(True)
    )

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# Global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

# import of standard configurations
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')    
process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi")
process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdGeometry_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cfi")
# process.mtdGeometry.applyAlignment = cms.bool(False)

# files = ["file:/eos/home-n/npalmeri/MTD_photonReco/samples/CMSSW_15_0_0_pre3_noPU/MINIAOD_D110/step3.root"]

files = []

if options.sample == 'lowPt':
    # 0.1-10 GeV SingleGammaFlat sample (self-produced)
    files = [f"root://xrootd-cms.infn.it///store/user/npalmeri/MTDPhotonReco/crab_MTDPhotonReco/CRAB_UserFiles/SingleGammaFlatPt0p1To10_Run4D110_aging1000_noPU_MTDPhotonReco/250307_134457/0000/step3_{i}.root" for i in range(1, 11)]
elif options.sample == 'highPt':
    # 8-150 GeV SingleGammaFlat sample (just 1k produced)
    files = [f"root://xrootd-cms.infn.it///store/user/npalmeri/MTDPhotonReco/crab_MTDPhotonReco/CRAB_UserFiles/SingleGammaFlatPt8To150_Run4D110_aging1000_noPU_MTDPhotonReco/250307_142923/0000/step3_{i}.root" for i in range(1, 21)]
else:
    # throw error
    raise ValueError("Invalid sample name: either lowPt (0.1 - 10 GeV) or highPt (8 - 150 GeV)")

secondary_files = []

# for dataset in options.datasets:
#     print('>> Creating list of files from: \n'+dataset)
#     for instance in ['global', 'phys03']:
#         query = "-query='file dataset="+dataset+" instance=prod/"+instance+"'"
#         if options.debug:
#             print(query)
#         lsCmd = subprocess.Popen(['dasgoclient '+query+' -limit=0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#         str_files, err = lsCmd.communicate()
#         files.extend(['root://cms-xrd-global.cern.ch/'+ifile for ifile in str_files.split("\n")])
#         files = [k for k in files if options.pattern in k]
#         files.pop()
#         if options.useparent:
#             print('>> Creating list of secondary files from: \n'+dataset)
#             for file in files:
#                 query = "-query='parent file="+file[len('root://cms-xrd-global.cern.ch/'):]+" instance=prod/"+instance+"'"
#                 if options.debug:
#                     print(query)
#                 lsCmd = subprocess.Popen(['dasgoclient '+query+' -limit=0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#                 str_files, err = lsCmd.communicate()
#                 secondary_files.extend(['root://cms-xrd-global.cern.ch/'+ifile for ifile in str_files.split("\n")])        
#                 secondary_files.pop()

# for eosdir in options.eosdirs:
#     if eosdir[-1] != '/':
#         eosdir += '/'
#     print('>> Creating list of files from: \n'+eosdir)
    
#     command = '/bin/find '+eosdir+' -type f | grep root | grep -v failed'
#     str_files = subprocess.check_output(command,shell=True).splitlines()
#     files.extend(['file:'+ifile for ifile in str_files if options.pattern in ifile])

# for eosusdir in options.eosusdirs:
#     if eosusdir[-1] != '/':
#         eosusdir += '/'
#     print('>> Creating list of files from: \n'+eosusdir)
    
#     command = 'eos root://cmseos.fnal.gov ls '+eosusdir+' | grep root | grep -v failed | grep '+options.pattern+' | grep -v '+options.antipattern
#     str_files = subprocess.check_output(command,shell=True).splitlines()
#     files.extend(['root://cms-xrd-global.cern.ch/'+eosusdir+'/'+ifile for ifile in str_files])
        
if options.debug:
    for ifile in files:
        print(ifile)        
    for ifile in secondary_files:
        print(ifile)        

# Input source
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(files),
    secondaryFileNames = cms.untracked.vstring(secondary_files)
)
#process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

# Analyzer
process.load('PrecisionTiming.FTLAnalysis.myMTDNeutralsAnalyzer_cfi')
MTDDumper = process.MTDNeutralsAnalyzer
MTDDumper.storeClusters = options.storeclusters

# Output TFile
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.output)
)

process.runseq = cms.Sequence(MTDDumper)
process.path = cms.Path(process.runseq)
process.schedule = cms.Schedule(process.path)
