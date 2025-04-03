#ifndef _MTD_NEUTRALS_TREE_
#define _MTD_NEUTRALS_TREE_

#include "ExternalTools/DynamicTTree/interface/DynamicTTreeBase.h"

using namespace std;

//---Define the TTree branches
#define DYNAMIC_TREE_NAME MTDNeutralsTree_vect

#define DATA_TABLE                              \
    DATA(int, run)                              \
    DATA(int, luminosityBlock)                  \
    DATA(int, event)                            \
    DATA(int, nGENConvertedPhoton)              \
    DATA(int, nGENElectron)                     \
    DATA(float, genPV_t)                        \
    DATA(float, genPV_x)                        \
    DATA(float, genPV_y)                        \
    DATA(float, genPV_z)                        \

#define DATA_CLASS_TABLE                        \
    DATA(vector<float>, GENConvertedPhoton_pt)         \
    DATA(vector<float>, GENConvertedPhoton_eta)        \
    DATA(vector<float>, GENConvertedPhoton_phi)        \
    DATA(vector<float>, GENConvertedPhoton_energy)     \
    DATA(vector<float>, GENConvertedPhoton_t0)         \
    DATA(vector<float>, GENConvertedPhoton_x0)          \
    DATA(vector<float>, GENConvertedPhoton_y0)          \
    DATA(vector<float>, GENConvertedPhoton_z0)          \
    DATA(vector<float>, GENConvertedPhoton_convRadius) \
    DATA(vector<float>, GENConvertedPhoton_convZ)      \
    DATA(vector<float>, GENConvertedPhoton_convPhi)    \
    DATA(vector<int>,   GENConvertedPhoton_nlegs)      \
    DATA(vector<vector<int>>, GENConvertedPhoton_eleIdxs)   \
    DATA(vector<int>, GENElectron_photonIdx)          \
    DATA(vector<float>, GENElectron_pt)                \
    DATA(vector<float>, GENElectron_eta)               \
    DATA(vector<float>, GENElectron_phi)               \
    DATA(vector<float>, GENElectron_energy)            \
    DATA(vector<vector<float>>, GENElectron_tClus)    \
    DATA(vector<vector<float>>, GENElectron_xClus)    \
    DATA(vector<vector<float>>, GENElectron_yClus)    \
    DATA(vector<vector<float>>, GENElectron_zClus)    \
    DATA(vector<vector<int>>, GENElectron_typeClus)   \
    DATA(vector<vector<float>>, GENElectron_energyClus)    \
    DATA(vector<float>, Photon_pt)                     \
    DATA(vector<float>, Photon_eta)                    \
    DATA(vector<float>, Photon_phi)                    \
    DATA(vector<float>, Photon_energy)                 \
    DATA(vector<float>, Photon_drMatch)                \
    DATA(vector<float>, MTDHit_drMatch)                \
    DATA(vector<float>, MTDHit_energy)                  \
    DATA(vector<float>, MTDHit_time)                    \
    DATA(vector<float>, MTDHit_timeErr)                 \
    DATA(vector<float>, MTDHit_rho)                     \
    DATA(vector<float>, MTDHit_z)                       \
    DATA(vector<float>, MTDHit_phi)                     \
    DATA(vector<float>, MTDHit_eta)                     \
    DATA(vector<float>, MTDHitSim_energy)             \
    DATA(vector<float>, MTDHitSim_time)               \
    DATA(vector<float>, MTDHitSim_rho)                \
    DATA(vector<float>, MTDHitSim_z)                  \
    DATA(vector<float>, MTDHitSim_phi)                \
    DATA(vector<float>, MTDHitSim_eta)                \
    DATA(vector<float>, MTDHitSim_clusType)           \

#include "ExternalTools/DynamicTTree/interface/DynamicTTreeInterface.h"

//---Define the TTree branches
#define DYNAMIC_TREE_NAME MTDNeutralsToyTree

#define DATA_TABLE                              \
    DATA(float, vtx_z)                          \
    DATA(float, vtx_t)                          \
    DATA(int, barrel_npho)                      \
    DATA(float, barrel_sumet)           

#define DATA_CLASS_TABLE                        \
    DATA(vector<double>, t_res)                 \
    DATA(vector<int>, barrel_mtd_noeff_npho)    \
    DATA(vector<float>, barrel_mtd_noeff_sumet) \
    DATA(vector<int>, barrel_mtd_npho)          \
    DATA(vector<float>, barrel_mtd_sumet)                   

#include "ExternalTools/DynamicTTree/interface/DynamicTTreeInterface.h"

#endif
