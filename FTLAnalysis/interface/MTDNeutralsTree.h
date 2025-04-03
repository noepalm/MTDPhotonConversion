#ifndef _MTD_NEUTRALS_TREE_
#define _MTD_NEUTRALS_TREE_

#include "ExternalTools/DynamicTTree/interface/DynamicTTreeBase.h"

using namespace std;

//---Define the TTree branches
#define DYNAMIC_TREE_NAME MTDNeutralsTree

#define DATA_TABLE                              \
    DATA(int, run)                              \
    DATA(int, lumi)                             \
    DATA(int, event)                            \
    DATA(float, genpv_t)                        \
    DATA(float, genpv_z)                        \
    DATA(float, pt)                             \
    DATA(float, eta)                            \
    DATA(float, phi)                            \
    DATA(float, ecalEnergy)                     \
        DATA(float, hcalEnergy)                 \
    DATA(float, particleId)                     \
    DATA(int, clus_n)                           \
    DATA(float, gen_pt)                         \
    DATA(float, gen_eta)                        \
    DATA(float, gen_phi)                        \
    DATA(float, gen_DR)                         \
    DATA(int, gen_pdgId)                        \
    DATA(int, gen_status)                       \
    DATA(int, chosen_clus_pos)                  \
    DATA(float, minDR)                          \
    DATA(float, minDEta)                        \
    DATA(float, minDPhi)                        \
    DATA(float, minSChi2)                       \
    DATA(float, minTChi2)                       \
    DATA(float, tof)                            \
    DATA(float, mtdTime)                        \
    DATA(float, mtdEnergy)                      \
    DATA(float, simh_DR)                        \
    DATA(float, simh_recoh_DR)                  \
    DATA(float, simh_recoh_DPhi)                \
    DATA(float, simh_energy)                    \
    DATA(float, simh_time)                      \
    DATA(float, simh_tof)                       \
    DATA(float, simh_clus_DR)                   \
    DATA(float, simh_recoh_clus_DR)             \
    DATA(float, simh_recoh_clus_DPhi)           \
    DATA(float, simh_clus_energy)               \
    DATA(float, simh_clus_time)                 \
    DATA(float, simh_clus_tof)                  \
    DATA(int, mct_nlegs)                        \
    DATA(float, mct_pt)                         \
    DATA(float, mct_eta)                        \
    DATA(float, mct_phi)                        \
    DATA(float, mct_energy)                     \
    DATA(float, mct_t0)                         \
    DATA(float, mct_convRadius)                 \
    DATA(float, mct_convZ)                      \
    DATA(float, mct_convPhi)                    \
    DATA(float, mct_ele1_pt)                    \
    DATA(float, mct_ele1_eta)                   \
    DATA(float, mct_ele1_phi)                   \
    DATA(float, mct_ele2_pt)                    \
    DATA(float, mct_ele2_eta)                   \
    DATA(float, mct_ele2_phi)                   \
    DATA(float, mct_eles_dr)                    \
    DATA(float, mct_eles_dphi)                  \
    DATA(float, mct_ele1_dr)                    \
    DATA(float, mct_ele2_dr)                    \
    DATA(float, mct_ele1_tmtd)                  \
    DATA(float, mct_ele2_tmtd)                  \
    DATA(float, mct_ele1_xmtd)                  \
    DATA(float, mct_ele1_ymtd)                  \
    DATA(float, mct_ele1_zmtd)                  \
    DATA(float, mct_ele2_xmtd)                  \
    DATA(float, mct_ele2_ymtd)                  \
    DATA(float, mct_ele2_zmtd)                  \
    DATA(float, mct_ele1_dphi)                  \
    DATA(float, mct_ele2_dphi)                  \
    
#define DATA_CLASS_TABLE                        \
    DATA(vector<float>, clus_size)              \
    DATA(vector<float>, clus_size_x)            \
    DATA(vector<float>, clus_size_y)            \
    DATA(vector<float>, clus_energy)            \
    DATA(vector<float>, clus_time)              \
    DATA(vector<float>, clus_rr)                \
    DATA(vector<float>, clus_eta)               \
    DATA(vector<float>, clus_phi)               \
    DATA(vector<float>, clus_seed_energy)       \
    DATA(vector<float>, clus_seed_time)     \
    DATA(vector<float>, clus_seed_x)            \
    DATA(vector<float>, clus_seed_y)            \
    DATA(vector<float>, clus_x)                 \
    DATA(vector<float>, clus_y)                 \
    DATA(vector<float>, clus_z)                 \
    DATA(vector<float>, clus_global_R)          \
    DATA(vector<float>, clus_global_dist)       \
    DATA(vector<float>, clus_neu_DPhi)          \
    DATA(vector<float>, clus_neu_DEta)          \
    DATA(vector<float>, clus_neu_DR)            \
    DATA(vector<float>, clus_neu_schi2)         \
    DATA(vector<float>, clus_neu_tchi2)         \
    DATA(vector<float>, clus_cele1_DR)          \
    DATA(vector<float>, clus_cele2_DR)          
    
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
