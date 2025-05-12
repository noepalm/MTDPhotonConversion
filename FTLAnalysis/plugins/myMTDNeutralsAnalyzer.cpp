#ifndef _MTD_NEUTRALS_ANALIZER_
#define _MTD_NEUTRALS_ANALIZER_

#include "TMath.h"
#include "TVector3.h"

#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/MTDGeomDet.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"

#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetSector.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

// MTD truth association maps
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToRecoClusterAssociationMap.h"

#include "PrecisionTiming/FTLAnalysis/interface/MTDNeutralsTree_vectorized.h"

using namespace edm;
using namespace geant_units::operators;
using namespace std;

bool DEBUG = false;

// ------ UTILS ------ //
std::pair<float, float> mean_stddev(std::vector<float>& v) {
    float sum = std::accumulate(v.begin(), v.end(), 0.0);
    float mean = sum / v.size();
    
    std::vector<float> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(),
                   [mean](float value){return value - mean;});
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float std = std::sqrt(sq_sum / v.size());

    return std::pair<float, float>(mean, std);
}

struct ConvertedPhoton {
    TrackingParticleRef genPhoton;
    vector<int> genElectron_idxs;
    TrackingParticleRefVector genElectrons;
    reco::PFCandidateRef pfPhoton;
    MTDTrackingRecHit bestHit;
    MtdSimLayerClusterRef bestHit_simMatch;

    float drMatch = 1e6;
    float drMatch_PF = 1e6;
};

class MTDHitMatchingInfo {
    public:
        MTDHitMatchingInfo()
            {
                hit = -1;
                estChi2 = std::numeric_limits<double>::max();
                timeChi2 = std::numeric_limits<double>::max();
            }

        MTDHitMatchingInfo(int this_hit, double this_estChi2, double this_timeChi2) :
            hit(this_hit),
            estChi2(this_estChi2),
            timeChi2(this_timeChi2)
            { }
        
        MTDHitMatchingInfo(const MTDHitMatchingInfo& a) :
            hit(a.hit),
            estChi2(a.estChi2),
            timeChi2(a.timeChi2)
            { }
        //Operator used to sort the hits while performing the matching step at the MTD
        inline bool operator<(const MTDHitMatchingInfo &m2) const {
            //only for good matching in time use estChi2, otherwise use mostly time compatibility
            if (timeChi2<5 && m2.timeChi2<5)
                return chi2(5.) < m2.chi2(5.);
            else
                return chi2(10.) < m2.chi2(10.);
        }

        inline double chi2(float timeWeight=1.) const { return estChi2 + timeWeight*timeChi2; }

        int hit;
        double estChi2;
        double timeChi2;
};

class myMTDNeutralsAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:                             

    typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>,ROOT::Math::DefaultCoordinateSystemTag> genXYZ;
    typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double>, ROOT::Math::DefaultCoordinateSystemTag> Point;
    
    explicit myMTDNeutralsAnalyzer(const edm::ParameterSet& pSet);
    ~myMTDNeutralsAnalyzer() {};

    //---methods
    virtual void beginJob() override {};
    virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
    virtual void endJob() override {};

    //---utils
    pair<float, float> getNeutralsMTDMatchingChi2s(reco::PFCandidate& cand, GlobalPoint& mtd_gp, const math::XYZTLorentzVectorD& genPV, float mtd_time); 
    // const edm::Ref<std::vector<TrackingParticle>>* myMTDNeutralsAnalyzer::getMatchedTP(const reco::GenParticle& genPart, const TrackingParticleCollection& trackingParticleCollection);
    // const edm::Ref<std::vector<TrackingParticle>>* myMTDNeutralsAnalyzer::getMatchedTP(const reco::TrackBaseRef& recoTrack);
    
    
private:
    //---reco tracks
    edm::Handle<reco::TrackCollection> GenRecTrackHandle_;
    edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;

    //---gen tracks (== Tracking Particles)
    edm::Handle<TrackingParticleCollection> trackingParticleCollectionHandle_;
    edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;

    //---gen particles
    edm::Handle<reco::GenParticleCollection> genParticlesHandle_;
    edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;

    //---MTD geometry
    edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
    edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

    //---sim hit tracker, for EGamma mc truth tool
    edm::Handle<edm::SimTrackContainer> simTkHandle_;
    edm::EDGetTokenT<edm::SimTrackContainer> simTkToken_;
    edm::Handle<edm::SimVertexContainer> simVtxHandle_;
    edm::EDGetTokenT<edm::SimVertexContainer> simVtxToken_;
    
    //---sim MTD hits
    edm::Handle<std::vector<PSimHit> > simHitsBTLHandle_;
    edm::EDGetTokenT<std::vector<PSimHit> > simHitsBTLToken_;    
    edm::Handle<std::vector<PSimHit> > simHitsETLHandle_;
    edm::EDGetTokenT<std::vector<PSimHit> > simHitsETLToken_;    
    
    //---BTL clusters
    edm::Handle<FTLRecHitCollection> recHitsBTLHandle_;
    edm::EDGetTokenT<FTLRecHitCollection> recHitsBTLToken_;
    edm::Handle<FTLClusterCollection> clustersBTLHandle_;
    edm::EDGetTokenT<FTLClusterCollection> clustersBTLToken_; 

    //---ETL clusters
    edm::Handle<FTLRecHitCollection> recHitsETLHandle_;
    edm::EDGetTokenT<FTLRecHitCollection> recHitsETLToken_;    
    edm::Handle<FTLClusterCollection> clustersETLHandle_;
    edm::EDGetTokenT<FTLClusterCollection> clustersETLToken_;    

    //---MTD Tracking Rec Hits
    edm::Handle<MTDTrackingDetSetVector> hitsHandle_;
    edm::EDGetTokenT<MTDTrackingDetSetVector> recoMTDhits_;

    //---MTD TP to sim cluster maps
    edm::Handle<reco::TPToSimCollectionMtd> tp2SimAssociationMapHandle_;
    edm::EDGetTokenT<reco::TPToSimCollectionMtd> tp2SimAssociationMapToken_;

    edm::Handle<MtdRecoClusterToSimLayerClusterAssociationMap> r2sAssociationMapHandle_;
    edm::EDGetTokenT<MtdRecoClusterToSimLayerClusterAssociationMap> r2sAssociationMapToken_;

    //---PF
    edm::Handle<reco::PFCandidateCollection> pfCandidatesHandle_;
    edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidatesToken_;

    //---vertices
    edm::EDGetTokenT<genXYZ>                  genXYZToken_;
    edm::Handle<genXYZ>                       genXYZHandle_;
    edm::EDGetTokenT<float>                   genT0Token_;
    edm::Handle<float>                        genT0Handle_;
    edm::EDGetTokenT<vector<reco::Vertex> >   vtx3DToken_;
    edm::Handle<vector<reco::Vertex> >        vtx3DHandle_;
    edm::EDGetTokenT<vector<reco::Vertex> >   vtx4DToken_;
    edm::Handle<vector<reco::Vertex> >        vtx4DHandle_;    
    
    //---options
    bool storeClusters_;
    BTLDetId::CrysLayout crysLayout_;
    
    //---outputs
    MTDNeutralsTree_vect outTree_;
    edm::Service<TFileService> fs_;  
  
};

myMTDNeutralsAnalyzer::myMTDNeutralsAnalyzer(const edm::ParameterSet& pSet):
    mtdgeoToken_(esConsumes<MTDGeometry, MTDDigiGeometryRecord>()),
    mtdtopoToken_(esConsumes<MTDTopology, MTDTopologyRcd>()){
    GenRecTrackToken_ = consumes<reco::TrackCollection>(pSet.getUntrackedParameter<edm::InputTag>("recoTracks"));
    trackingParticleCollectionToken_ = consumes<TrackingParticleCollection>(pSet.getUntrackedParameter<edm::InputTag>("genTracks"));
    genParticlesToken_ = consumes<reco::GenParticleCollection>(pSet.getUntrackedParameter<edm::InputTag>("genParticlesTag"));
    simTkToken_ = consumes<edm::SimTrackContainer>(pSet.getUntrackedParameter<edm::InputTag>("simTkTag"));
    simVtxToken_ = consumes<edm::SimVertexContainer>(pSet.getUntrackedParameter<edm::InputTag>("simVtxTag"));
    simHitsBTLToken_ = consumes<std::vector<PSimHit> >(pSet.getUntrackedParameter<edm::InputTag>("simHitsBTLTag"));
    simHitsETLToken_ = consumes<std::vector<PSimHit> >(pSet.getUntrackedParameter<edm::InputTag>("simHitsETLTag"));
    clustersBTLToken_ = consumes<FTLClusterCollection>(pSet.getUntrackedParameter<edm::InputTag>("clustersBTLTag"));
    clustersETLToken_ = consumes<FTLClusterCollection>(pSet.getUntrackedParameter<edm::InputTag>("clustersETLTag"));
    tp2SimAssociationMapToken_ = consumes<reco::TPToSimCollectionMtd>(pSet.getUntrackedParameter<edm::InputTag>("tp2SimAssociationMapTag"));
    r2sAssociationMapToken_ = consumes<MtdRecoClusterToSimLayerClusterAssociationMap>(pSet.getUntrackedParameter<edm::InputTag>("r2sAssociationMapTag"));
    pfCandidatesToken_ = consumes<reco::PFCandidateCollection>(pSet.getUntrackedParameter<edm::InputTag>("pfCandidatesTag"));
    genXYZToken_ = consumes<genXYZ>(pSet.getUntrackedParameter<edm::InputTag>("genXYZTag"));
    genT0Token_ = consumes<float>(pSet.getUntrackedParameter<edm::InputTag>("genT0Tag"));
    vtx3DToken_ = consumes<vector<reco::Vertex> >(pSet.getUntrackedParameter<edm::InputTag>("vtx3DTag"));
    vtx4DToken_ = consumes<vector<reco::Vertex> >(pSet.getUntrackedParameter<edm::InputTag>("vtx4DTag"));
    recoMTDhits_ = consumes<MTDTrackingDetSetVector>(pSet.getUntrackedParameter<edm::InputTag>("mtdTrackingRecHits"));
    storeClusters_ = pSet.getUntrackedParameter<bool>("storeClusters");
    crysLayout_ = (BTLDetId::CrysLayout)(pSet.getUntrackedParameter<int>("crysLayout"));
    outTree_ = MTDNeutralsTree_vect(pSet.getUntrackedParameter<string>("outTreeName").c_str(), "4D TOFPID studies");
}

void myMTDNeutralsAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup){
    //---load reco tracks
    iEvent.getByToken(GenRecTrackToken_, GenRecTrackHandle_);
    auto GenRecTracks = *GenRecTrackHandle_.product();

    //---load gen tracks
    iEvent.getByToken(trackingParticleCollectionToken_, trackingParticleCollectionHandle_);
    auto trackingParticles = *trackingParticleCollectionHandle_.product();

    //---load gen particles
    iEvent.getByToken(genParticlesToken_, genParticlesHandle_);
    auto genParticles = *genParticlesHandle_.product();
    
    //---gen mc-truth photons
    iEvent.getByToken(simTkToken_, simTkHandle_);
    iEvent.getByToken(simVtxToken_, simVtxHandle_);


    //---load sim hits
    iEvent.getByToken(simHitsBTLToken_, simHitsBTLHandle_);
    auto simHitsBTL = *simHitsBTLHandle_.product();
    iEvent.getByToken(simHitsETLToken_, simHitsETLHandle_);
    auto simHitsETL = *simHitsETLHandle_.product();

    //---NEW: load TP2SimCluster association maps
    iEvent.getByToken(tp2SimAssociationMapToken_, tp2SimAssociationMapHandle_);
    //---NEW: load reco2sim MTD cluster association maps
    iEvent.getByToken(r2sAssociationMapToken_, r2sAssociationMapHandle_);


    //---get the MTD geometry
    auto mtdgeoHandle_ = iSetup.getTransientHandle(mtdgeoToken_);
    const MTDGeometry* geom = mtdgeoHandle_.product();
    // check that geom is okay
    if (geom == nullptr) {
        cout << "ERROR: MTDGeometry not found!" << endl;
        return;
    }
    auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
    const MTDTopology* topology = topologyHandle.product();
  

    // edm::ESHandle<MTDGeometry> geoHandle;
    // iSetup.get<MTDDigiGeometryRecord>().get(geoHandle);
    // mtdGeometry_ = geoHandle.product();
  
    // edm::ESHandle<MTDDetLayerGeometry> layerGeo;
    // iSetup.get<MTDRecoGeometryRecord>().get(layerGeo);
    
    //---MTD clusters
    // iEvent.getByToken(clustersBTLToken_, clustersBTLHandle_);
    // auto clustersBTL = *clustersBTLHandle_.product();
    auto btlRecCluHandle = makeValid(iEvent.getHandle(clustersBTLToken_));

    // iEvent.getByToken(clustersETLToken_, clustersETLHandle_);
    // auto clustersETL = *clustersETLHandle_.product();

    //---load PFCandidates
    iEvent.getByToken(pfCandidatesToken_, pfCandidatesHandle_);
    auto pfCandidates = *pfCandidatesHandle_.product();    
    
    //---load gen, sim and reco vertices
    // GEN
    iEvent.getByToken(genXYZToken_, genXYZHandle_);
    iEvent.getByToken(genT0Token_, genT0Handle_);
    auto xyz = genXYZHandle_.product();
    auto t = *genT0Handle_.product();
    auto v = math::XYZVectorD(xyz->x(), xyz->y(), xyz->z());
    auto genPV = SimVertex(v, t).position();
    // 3D
    iEvent.getByToken(vtx3DToken_, vtx3DHandle_);
    auto vtxs3D = *vtx3DHandle_.product();
    // Full 4D
    iEvent.getByToken(vtx4DToken_, vtx4DHandle_);
    auto vtxs4D = *vtx4DHandle_.product();

    // // DEBUG: PRINT ALL TPs
    // if(DEBUG){
    //     for(size_t i = 0; i < trackingParticles.size(); i++){
    //         auto& tp = trackingParticles[i];
    //         cout << "tp idx = " << i << ": pdgId = " << tp.pdgId() << ", status = " << tp.status() << ", pt = " << tp.pt() << ", eta = " << tp.eta() << ", phi = " << tp.phi() << endl;
    //         // if tp has decay vertex, print daughter tracks
    //         for(size_t j = 0; j < tp.decayVertices().size(); j++){
    //             auto vtx = tp.decayVertices()[j];
    //             cout << "decay vtx: r = " << vtx->position().rho() << ", z = " << vtx->position().z() << endl;
    //             for(size_t k = 0; k < tp.decayVertices()[j]->nDaughterTracks(); k++){
    //                 // find idx of k-th tp in trackingParticles collection
    //                 auto daughter_tp = tp.decayVertices()[j]->daughterTracks()[k];
    //                 auto daughter_tp_idx = -1;
    //                 for(size_t l = 0; l < trackingParticles.size(); l++){
    //                     if(trackingParticles[l].pdgId() == daughter_tp->pdgId() && trackingParticles[l].pt() == daughter_tp->pt() && trackingParticles[l].eta() == daughter_tp->eta() && trackingParticles[l].phi() == daughter_tp->phi()){
    //                         daughter_tp_idx = l;
    //                         break;
    //                     }
    //                 }
    //                 cout << "\tTP " << daughter_tp_idx << ": pdgId = " << vtx->daughterTracks()[k]->pdgId() << ", pt = " << vtx->daughterTracks()[k]->pt() << ", eta = " << vtx->daughterTracks()[k]->eta() << ", phi = " << vtx->daughterTracks()[k]->phi() << endl;
    //             }
    //         }
    //     }

    //     cout << endl;
    // }

    std::vector<ConvertedPhoton> convertedPhotons;
    TrackingParticleRefVector gen_electrons;

    // ------ 1. find gen photons and electrons (TrackingParticle) ------
    for (size_t i = 0; i < trackingParticles.size(); ++i) {
        TrackingParticleRef tp(trackingParticleCollectionHandle_, i);

        // skip particle if not in BTL (WITH MARGIN)
        if(abs(tp->momentum().eta()) > 1.2) //ACTUALLY 1.5 (need to figure out DetId problem for ETL tracks -- BTL tracks can be deviated)
            continue;

        if(tp->pdgId() == 22){
            ConvertedPhoton cp;
            cp.genPhoton = tp;

            // look at electron daughters
            for(auto& decayVertex : tp->decayVertices()){
                for(auto& daughterTrack : decayVertex->daughterTracks()){
                    if(abs(daughterTrack->pdgId()) == 11)
                        cp.genElectrons.push_back(daughterTrack);                        
                }
            }
            // sort genElectrons by pt
            std::vector<TrackingParticleRef> sortedRefs(cp.genElectrons.begin(), cp.genElectrons.end());
            std::sort(sortedRefs.begin(), sortedRefs.end(), [](const TrackingParticleRef& a, const TrackingParticleRef& b) { return a->pt() > b->pt(); });
            // TODO: fix reassignment.
            cp.genElectrons.clear();
            for(auto& ref : sortedRefs)
                cp.genElectrons.push_back(ref);
            
            convertedPhotons.push_back(cp);
        }
    }

    // DEBUGGING:
    // print ALL tracking particles for event 46
    if((iEvent.id().event() == 35) || (iEvent.id().event() == 65) || (iEvent.id().event() == 134)){
        cout << "Event " << iEvent.id().event() << ":" << endl;
        for(size_t i = 0; i < trackingParticles.size(); i++){
            auto& tp = trackingParticles[i];
            cout << "tp idx = " << i << ": pdgId = " << tp.pdgId() << ", status = " << tp.status() << ", E = " << tp.energy() << ", pt = " << tp.pt() << ", eta = " << tp.eta() << ", phi = " << tp.phi() << endl;
            cout << "           t at vtx = " << tp.parentVertex()->position().t() * 1e9 << " ns" << std::endl;

            // if tp has decay vertex, print daughter tracks
            for(size_t j = 0; j < tp.decayVertices().size(); j++){
                auto vtx = tp.decayVertices()[j];
                cout << "decay vtx: r = " << vtx->position().rho() << ", z = " << vtx->position().z() << endl;
                for(size_t k = 0; k < tp.decayVertices()[j]->nDaughterTracks(); k++){
                    // find idx of k-th tp in trackingParticles collection
                    auto daughter_tp = tp.decayVertices()[j]->daughterTracks()[k];
                    auto daughter_tp_idx = -1;
                    for(size_t l = 0; l < trackingParticles.size(); l++){
                        if(trackingParticles[l].pdgId() == daughter_tp->pdgId() && trackingParticles[l].pt() == daughter_tp->pt() && trackingParticles[l].eta() == daughter_tp->eta() && trackingParticles[l].phi() == daughter_tp->phi()){
                            daughter_tp_idx = l;
                            break;
                        }
                    }
                    cout << "\tTP " << daughter_tp_idx << ": pdgId = " << vtx->daughterTracks()[k]->pdgId() << ", pt = " << vtx->daughterTracks()[k]->pt() << ", eta = " << vtx->daughterTracks()[k]->eta() << ", phi = " << vtx->daughterTracks()[k]->phi() << endl;
                }
            }
        }

        cout << endl;
    }
    
    // ------ 2. match PF candidates to photons and electrons
    for(size_t i = 0; i < pfCandidates.size(); i++){
        reco::PFCandidateRef cand(pfCandidatesHandle_, i);
        // try to match to gen photons
        for(auto& cp : convertedPhotons){
            if (cand->particleId() == reco::PFCandidate::gamma) { // PF candidate is photon
                // check match with photon itself
                float dr = deltaR(cand->eta(), cand->phi(), cp.genPhoton->eta(), cp.genPhoton->phi());
                float dptOverPt = (cand->pt() - cp.genPhoton->pt()) / cp.genPhoton->pt();
                if(dr < cp.drMatch_PF && dr < 0.5 && dptOverPt < 0.3){
                    cp.pfPhoton = cand;
                    cp.drMatch_PF = dr;
                }
            }
        }
    }

    // ------- 3. match MTD hit to photon
    // load MTD tracking rec hits
    iEvent.getByToken(recoMTDhits_, hitsHandle_);
    auto mtdHits = *hitsHandle_.product();

    // iterate over hits
    for(auto& cp : convertedPhotons){
        if(cp.pfPhoton.isNull())
            continue;
    
        // iterate over detId in MTD hits collection
        for(const auto& detSet : mtdHits){
            
            // iterate over hits in collection
            for(const auto& hit : detSet){
                // compute deltaR between hit and PF photon
                BTLDetId detId = hit.geographicalId().rawId();
                DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
                const MTDGeomDet* thedet = geom->idToDet(geoId);
                if(thedet == nullptr) continue;

                const auto& global_point = thedet->toGlobal(hit.localPosition());

                float dr = deltaR(cp.pfPhoton->eta(), cp.pfPhoton->phi(), global_point.eta(), global_point.phi());

                if(dr < cp.drMatch){
                    cp.drMatch = dr;
                    cp.bestHit = hit;
                }
            }

        }

    }

    // ------- 4. match MTD reco hit to its SIM hit
    for(auto& cp : convertedPhotons){
        if(cp.drMatch < 1e6){ //if RECO hit match found
            const auto& hitCluster = cp.bestHit.mtdCluster();
            if (hitCluster.size() != 0){
                auto recoClusterRef = edmNew::makeRefTo(btlRecCluHandle, &hitCluster); //NB: BTL only here
                // auto recoClusterRef = edmNew::makeRefTo(clustersBTLHandle_, &hitCluster); //NB: BTL only here; to be eventually extended
                        
                // use association map
                auto itp = r2sAssociationMapHandle_->equal_range(recoClusterRef);

                if (itp.first != itp.second) {
                    const auto& simCluster = (*itp.first).second; //NB: returns a VECTOR of MtdSimLayerCluster refs 
                    if(simCluster.size() > 0)
                        cp.bestHit_simMatch = simCluster[0]; // ASSUMPTION: assume only 1 sim cluster per reco cluster
                }
            }
        }

    }

    // -----------------------------------

    // SAVE ALL INFO TO TREE
    outTree_.Reset();

    // event-level information
    outTree_.event = iEvent.id().event();
    outTree_.luminosityBlock = iEvent.id().luminosityBlock();
    outTree_.run = iEvent.id().run();

    // [! NB !] same as GENConvertedPhotons.x0, t0, etc. But t0 is in SECONDS there.
    outTree_.genPV_t = genPV.t(); // ns
    outTree_.genPV_x = genPV.x(); // cm
    outTree_.genPV_y = genPV.y(); // cm
    outTree_.genPV_z = genPV.z(); // cm

    int ele_idx = 0;
    outTree_.nGENConvertedPhoton = int(convertedPhotons.size());

    int photon_idx = 0;
    for(auto& cp : convertedPhotons){

        outTree_.GENConvertedPhoton_pt->push_back(cp.genPhoton->pt());
        outTree_.GENConvertedPhoton_eta->push_back(cp.genPhoton->eta());
        outTree_.GENConvertedPhoton_phi->push_back(cp.genPhoton->phi());
        outTree_.GENConvertedPhoton_energy->push_back(cp.genPhoton->energy());
        outTree_.GENConvertedPhoton_t0->push_back(cp.genPhoton->parentVertex()->position().t());
        outTree_.GENConvertedPhoton_x0->push_back(cp.genPhoton->parentVertex()->position().x());
        outTree_.GENConvertedPhoton_y0->push_back(cp.genPhoton->parentVertex()->position().y());
        outTree_.GENConvertedPhoton_z0->push_back(cp.genPhoton->parentVertex()->position().z());

        if(cp.genPhoton->decayVertices().size() > 0){
            // ASSUMPTION: assume always 1 decay vtx only
            auto vtx = cp.genPhoton->decayVertices()[0];
            outTree_.GENConvertedPhoton_convRadius->push_back(vtx->position().rho());
            outTree_.GENConvertedPhoton_convZ->push_back(vtx->position().z());
            outTree_.GENConvertedPhoton_convPhi->push_back(vtx->position().phi());
        }

        outTree_.GENConvertedPhoton_nlegs->push_back(cp.genElectrons.size());

        vector<int> ele_idxs;

        for(auto& electron : cp.genElectrons){
            outTree_.GENElectron_photonIdx->push_back(photon_idx);
            outTree_.GENElectron_pt->push_back(electron->pt());
            outTree_.GENElectron_eta->push_back(electron->eta());
            outTree_.GENElectron_phi->push_back(electron->phi());
            outTree_.GENElectron_energy->push_back(electron->energy());

            ele_idxs.push_back(ele_idx++);

            vector<int> cluster_type = {};
            vector<int> cluster_nHits = {};
            vector<float> cluster_t = {};
            vector<float> cluster_x = {};
            vector<float> cluster_y = {};
            vector<float> cluster_z = {};
            vector<float> cluster_energy = {};
            vector<float> cluster_tStd = {};
            vector<float> cluster_xStd = {};
            vector<float> cluster_yStd = {};
            vector<float> cluster_zStd = {};
            vector<float> cluster_energyStd = {};

            // retrieve matched MTDSimLayerCluster for each electron
            if(tp2SimAssociationMapHandle_->find(electron) != tp2SimAssociationMapHandle_->end()){
                const auto& simClusterRefs = tp2SimAssociationMapHandle_->find(electron)->val;

                for(size_t j = 0; j < simClusterRefs.size(); j++){
                    const auto& simCluster = simClusterRefs[j];

                    BTLDetId cluId = simCluster->detIds_and_rows()[0].first;
                    DetId geoId = cluId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
                    const MTDGeomDet* genericDet = geom->idToDet(geoId);
                    
                    if(genericDet == nullptr) continue;

                    cluster_t.push_back(simCluster->simLCTime());
                    cluster_type.push_back(simCluster->trackIdOffset());
                    
                    LocalPoint simClusLocalPos = simCluster->simLCPos();
                    const auto& simClusGlobalPos = genericDet->toGlobal(simClusLocalPos);
                    
                    cluster_x.push_back(simClusGlobalPos.x());
                    cluster_y.push_back(simClusGlobalPos.y());
                    cluster_z.push_back(simClusGlobalPos.z());

                    // save energy too
                    float simClusEnergy = convertUnitsTo(0.001_MeV, simCluster->simLCEnergy()); // GeV --> MeV                    
                    cluster_energy.push_back(simClusEnergy);

                    // INFO ON HITS AND CLUSTER EXTENSION
                    // retrieve hits and times, hits and positions
                    std::vector<std::pair<uint64_t, LocalPoint>> hits_and_positions = simCluster->hits_and_positions();
                    std::vector<std::pair<uint64_t, float>> hits_and_times = simCluster->hits_and_times();
                    std::vector<std::pair<uint64_t, float>> hits_and_energies = simCluster->hits_and_energies();
                    
                    // create vector of x, y, z, t from above
                    std::vector<float> hits_times;
                    std::transform(begin(hits_and_times), end(hits_and_times),
                                   std::back_inserter(hits_times),
                                   [](auto const& pair){return pair.second;});
                    std::vector<float> hits_x, hits_y, hits_z;
                    std::transform(begin(hits_and_positions), end(hits_and_positions),
                                   std::back_inserter(hits_x),
                                   [](auto const& pair){return pair.second.x();});                    
                    std::transform(begin(hits_and_positions), end(hits_and_positions),
                                   std::back_inserter(hits_y),
                                   [](auto const& pair){return pair.second.y();});
                    std::transform(begin(hits_and_positions), end(hits_and_positions),
                                   std::back_inserter(hits_z),
                                   [](auto const& pair){return pair.second.z();});
                    std::vector<float> hits_energies;
                    std::transform(begin(hits_and_energies), end(hits_and_energies),
                                   std::back_inserter(hits_energies),
                                   [](auto const& pair){return pair.second;});
                    
                    // compute and store std dev 
                    cluster_tStd.push_back(mean_stddev(hits_times).second);
                    cluster_xStd.push_back(mean_stddev(hits_x).second);
                    cluster_yStd.push_back(mean_stddev(hits_y).second);
                    cluster_zStd.push_back(mean_stddev(hits_z).second);
                    cluster_energyStd.push_back(mean_stddev(hits_energies).second);

                    cluster_nHits.push_back(int(hits_times.size()));
                }   
            }

            // if no cluster saved, fill with -1
            if(cluster_t.size() == 0){
                cluster_t.push_back(-999.);
                cluster_x.push_back(-999.);
                cluster_y.push_back(-999.);
                cluster_z.push_back(-999.);
                cluster_energy.push_back(-999.);
                cluster_type.push_back(-999.);
                cluster_tStd.push_back(-999.);
                cluster_xStd.push_back(-999.);
                cluster_yStd.push_back(-999.);
                cluster_zStd.push_back(-999.);
                cluster_energyStd.push_back(-999.);
                cluster_nHits.push_back(-999);
            }

            // sort clusters by time
            std::vector<std::pair<float, int>> sortedClusters;
            for(size_t j = 0; j < cluster_t.size(); j++){
                sortedClusters.push_back(std::make_pair(cluster_t[j], j));
            }

            std::sort(sortedClusters.begin(), sortedClusters.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first < b.first; });
            // reorder clusters
            std::vector<float> sorted_cluster_t, sorted_cluster_x, sorted_cluster_y, sorted_cluster_z, sorted_cluster_energy;
            std::vector<float> sorted_cluster_tStd, sorted_cluster_xStd, sorted_cluster_yStd, sorted_cluster_zStd, sorted_cluster_energyStd;
            std::vector<int> sorted_cluster_type, sorted_cluster_nhits;
            for(size_t j = 0; j < sortedClusters.size(); j++){
                int idx = sortedClusters[j].second;
                sorted_cluster_t.push_back(cluster_t[idx]);
                sorted_cluster_x.push_back(cluster_x[idx]);
                sorted_cluster_y.push_back(cluster_y[idx]);
                sorted_cluster_z.push_back(cluster_z[idx]);
                sorted_cluster_energy.push_back(cluster_energy[idx]);
                sorted_cluster_type.push_back(cluster_type[idx]);
                sorted_cluster_tStd.push_back(cluster_tStd[idx]);
                sorted_cluster_xStd.push_back(cluster_xStd[idx]);
                sorted_cluster_yStd.push_back(cluster_yStd[idx]);
                sorted_cluster_zStd.push_back(cluster_zStd[idx]);
                sorted_cluster_energyStd.push_back(cluster_energyStd[idx]);
                sorted_cluster_nhits.push_back(cluster_nHits[idx]);
            }

            outTree_.GENElectron_tClus->push_back(sorted_cluster_t);
            outTree_.GENElectron_xClus->push_back(sorted_cluster_x);
            outTree_.GENElectron_yClus->push_back(sorted_cluster_y);
            outTree_.GENElectron_zClus->push_back(sorted_cluster_z);
            outTree_.GENElectron_energyClus->push_back(sorted_cluster_energy);
            outTree_.GENElectron_tStdClus->push_back(sorted_cluster_tStd);
            outTree_.GENElectron_xStdClus->push_back(sorted_cluster_xStd);
            outTree_.GENElectron_yStdClus->push_back(sorted_cluster_yStd);
            outTree_.GENElectron_zStdClus->push_back(sorted_cluster_zStd);
            outTree_.GENElectron_energyStdClus->push_back(sorted_cluster_energyStd);
            outTree_.GENElectron_typeClus->push_back(sorted_cluster_type);
            outTree_.GENElectron_nHitsClus->push_back(sorted_cluster_nhits);
        }    

        outTree_.GENConvertedPhoton_eleIdxs->push_back(ele_idxs);
        photon_idx++;

        // RECO information

        //---- PF photon
        outTree_.Photon_drMatch->push_back(cp.drMatch_PF);
        if(!cp.pfPhoton.isNull()){
            outTree_.Photon_pt->push_back(cp.pfPhoton->pt());
            outTree_.Photon_eta->push_back(cp.pfPhoton->eta());
            outTree_.Photon_phi->push_back(cp.pfPhoton->phi());
            outTree_.Photon_energy->push_back(cp.pfPhoton->energy());
        } else { //to maintain same dimensionality
            outTree_.Photon_pt->push_back(-999);
            outTree_.Photon_eta->push_back(-999);
            outTree_.Photon_phi->push_back(-999);
            outTree_.Photon_energy->push_back(-999);
        }

        //----- MTD hit
        outTree_.MTDHit_drMatch->push_back(cp.drMatch);
        if(cp.drMatch < 1e6){
            outTree_.MTDHit_time->push_back(cp.bestHit.time());
            outTree_.MTDHit_energy->push_back(cp.bestHit.energy());
    
            // compute deltaR between hit and PF photon
            BTLDetId detId = cp.bestHit.geographicalId().rawId();
            DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
            const MTDGeomDet* thedet = geom->idToDet(geoId);
            if(thedet == nullptr) continue;
    
            const auto& global_point = thedet->toGlobal(cp.bestHit.localPosition());
    
            outTree_.MTDHit_rho->push_back(global_point.perp());
            outTree_.MTDHit_z->push_back(global_point.z());
            outTree_.MTDHit_phi->push_back(global_point.phi());
            outTree_.MTDHit_eta->push_back(global_point.eta());

            // repeat for sim hit matched to MTD hit
            if(!cp.bestHit_simMatch.isNull()){
                float simClusEnergy = convertUnitsTo(0.001_MeV, cp.bestHit_simMatch->simLCEnergy()); // GeV --> MeV                    
                outTree_.MTDHitSim_energy->push_back(simClusEnergy);
                outTree_.MTDHitSim_clusType->push_back(cp.bestHit_simMatch->trackIdOffset());
                
                outTree_.MTDHitSim_time->push_back(cp.bestHit_simMatch->simLCTime());
                
                BTLDetId cluId = cp.bestHit_simMatch->detIds_and_rows()[0].first;
                DetId geoId = cluId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
                const MTDGeomDet* genericDet = geom->idToDet(geoId);
                
                if(genericDet == nullptr) continue;
                LocalPoint simClusLocalPos = cp.bestHit_simMatch->simLCPos();
                const auto& simClusGlobalPos = genericDet->toGlobal(simClusLocalPos);
    
                outTree_.MTDHitSim_rho->push_back(simClusGlobalPos.perp());
                outTree_.MTDHitSim_z->push_back(simClusGlobalPos.z());
                outTree_.MTDHitSim_phi->push_back(simClusGlobalPos.phi());
                outTree_.MTDHitSim_eta->push_back(simClusGlobalPos.eta());
            } else {
                outTree_.MTDHitSim_energy->push_back(-998);
                outTree_.MTDHitSim_time->push_back(-998);
                outTree_.MTDHitSim_rho->push_back(-998);
                outTree_.MTDHitSim_z->push_back(-998);
                outTree_.MTDHitSim_phi->push_back(-998);
                outTree_.MTDHitSim_eta->push_back(-998);
                outTree_.MTDHitSim_clusType->push_back(-998);
            }
        } else {
            // push -999 for all fields
            outTree_.MTDHit_energy->push_back(-999);
            outTree_.MTDHit_time->push_back(-999);
            outTree_.MTDHit_rho->push_back(-999);
            outTree_.MTDHit_z->push_back(-999);
            outTree_.MTDHit_phi->push_back(-999);
            outTree_.MTDHit_eta->push_back(-999);
            outTree_.MTDHitSim_energy->push_back(-999);
            outTree_.MTDHitSim_time->push_back(-999);
            outTree_.MTDHitSim_rho->push_back(-999);
            outTree_.MTDHitSim_z->push_back(-999);
            outTree_.MTDHitSim_phi->push_back(-999);
            outTree_.MTDHitSim_eta->push_back(-999);
            outTree_.MTDHitSim_clusType->push_back(-999);
        }


    }

    outTree_.nGENElectron = int(outTree_.GENElectron_pt->size());


    // if at least one converted photon, fill
    if(convertedPhotons.size() > 0)
        outTree_.GetTTreePtr()->Fill();

}

pair<float, float> myMTDNeutralsAnalyzer::getNeutralsMTDMatchingChi2s(reco::PFCandidate& cand, GlobalPoint& mtd_gp, const math::XYZTLorentzVectorD& genPV, float mtd_time) {
    //---space chi2    
    TVector3 cand_pos_at_mtd;
    cand_pos_at_mtd.SetPtEtaPhi(mtd_gp.perp(), cand.eta(), cand.phi());
    auto s_chi2 = pow((cand_pos_at_mtd.z()-mtd_gp.z())/(5.6/sqrt(12)/2), 2) +
        pow(deltaPhi(cand_pos_at_mtd.Phi(), mtd_gp.phi())*mtd_gp.perp()/(11.7/sqrt(12)/2), 2);

    //----time chi2
    auto tof = sqrt(pow(mtd_gp.x()-genPV.x(), 2)+pow(mtd_gp.y()-genPV.y(), 2)+pow(mtd_gp.z()-genPV.z(), 2))/2.99792458e1;
    tof -= 13.25*deltaPhi(cand.phi(), mtd_gp.phi())*deltaPhi(cand.phi(), mtd_gp.phi()); // [Q]: is this already corrected? 
    auto t_chi2 = pow(mtd_time-tof-genPV.t(), 2)/(0.05*0.05); //[Q]: assuming 50 ps uncertainty on MTD hit?
    
    return make_pair(s_chi2, t_chi2);
}

// // TP MATCHING: from reco track to TP
// const edm::Ref<std::vector<TrackingParticle>>* myMTDNeutralsAnalyzer::getMatchedTP(const reco::TrackBaseRef& recoTrack) {
//     auto found = r2s_->find(recoTrack);
  
//     // reco track not matched to any TP
//     if (found == r2s_->end())
//       return nullptr;
  
//     //matched TP equal to any TP associated to in time events
//     for (const auto& tp : found->val) {
//       if (tp.first->eventId().bunchCrossing() == 0)
//         return &tp.first;
//     }
  
//     // reco track not matched to any TP from vertex
//     return nullptr;
// }

// // TP matching: from GenPart to TP
// const edm::Ref<std::vector<TrackingParticle>>* myMTDNeutralsAnalyzer::getMatchedTP(const reco::GenParticle& genPart, const TrackingParticleCollection& trackingParticleCollection) {
//     // iterate over tracking particles
//     for (size_t i = 0; i < trackingParticleCollection.size(); ++i) {
//       const auto& tp = trackingParticleCollection[i];

//         //   // check if the tracking particle is from the same vertex
//         //   if (tp.eventId().bunchCrossing() != 0)
//         //     continue;
    
//         // check if the tracking particle is matched to a gen particle
//         if (tp.genParticles().size() == 0)
//         continue;

//         for (const auto& gp : tp.genParticles()){
//             // check if genParticle matches input genPart
//             if (gp->pdgId() == genPart.pdgId() &&
//                 gp->status() == genPart.status() &&
//                 gp->pt() == genPart.pt() &&
//                 gp->eta() == genPart.eta() &&
//                 gp->phi() == genPart.phi()){
                
//                 cout << "FOUND MATCHING TP!" << endl;
//                 cout << "Input genPart: ptr = " << &genPart << " pdgId = " << genPart.pdgId() << " status = " << genPart.status() << " pt = " << genPart.pt() << " eta = " << genPart.eta() << " phi = " << genPart.phi() << endl;
//                 cout << "Matched GP: ptr = " << &gp << " pdgId = " << gp->pdgId() << " status = " << gp->status() << " pt = " << gp->pt() << " eta = " << gp->eta() << " phi = " << gp->phi() << endl;
//                 cout << "Matched TP: pt = " << tp.pt() << " eta = " << tp.eta() << " phi = " << tp.phi() << endl;

//                 return &tp;
//             }
//         }
    
//     }

//     return nullptr;
// }

  

DEFINE_FWK_MODULE(myMTDNeutralsAnalyzer);

#endif
