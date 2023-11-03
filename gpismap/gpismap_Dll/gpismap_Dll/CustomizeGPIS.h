#pragma once
#ifndef __GPIS_MAP3__H__
#define __GPIS_MAP3__H__

#include "ObsGP.h"
#include "OnGPIS.h"
#include "octree.h"
#include "params.h"
#include <mutex>

typedef struct camParam_ {
    float fx;
    float fy;
    float cx;
    float cy;
    int width;
    int height;

    camParam_() {
        width = 640;
        height = 480;
        fx = 568.0; // 570.9361;
        fy = 568.0; // 570.9361;
        cx = 310;// 307;
        cy = 224; //240;
    }
    camParam_(float fx_, float fy_, float cx_, float cy_, int w_, int h_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(w_), height(h_) {
    }
} camParam;

typedef struct GPisMap3Param_ {
    float delx;         // numerical step delta (e.g. surface normal sampling)
    float fbias;        // constant map bias values (mean of GP)
    float obs_var_thre; // threshold for variance of ObsGP
    //  - If var(prediction) > v_thre, then don't rely on the prediction.
    int   obs_skip;     // use every 'skip'-th pixel
    float min_position_noise;
    float min_grad_noise;

    float map_scale_param;
    float map_noise_param;

    GPisMap3Param_() {
        delx = GPISMAP3_DELX;
        fbias = GPISMAP3_FBIAS;
        obs_skip = GPISMAP3_OBS_SKIP;
        obs_var_thre = GPISMAP3_OBS_VAR_THRE;
        min_position_noise = GPISMAP3_MIN_POS_NOISE;
        min_grad_noise = GPISMAP3_MIN_GRAD_NOISE;
        map_scale_param = GPISMAP3_MAP_SCALE;
        map_noise_param = GPISMAP3_MAP_NOISE;
    }

    GPisMap3Param_(GPisMap3Param_& par) {
        delx = par.delx;
        fbias = par.fbias;
        obs_skip = par.obs_skip;
        obs_var_thre = par.obs_var_thre;
        min_position_noise = par.min_position_noise;
        min_grad_noise = par.min_grad_noise;
        map_scale_param = par.map_scale_param;
        map_noise_param = par.map_noise_param;
    }
}GPisMap3Param;

class OcTreeNew : public OcTree {

};

class CustomizeGPIS {
protected:
    GPisMap3Param setting;

    OcTree* t;
    std::unordered_set<OcTree*> activeSet;
    const int mapDimension = 3;

    void init();
    bool preprocData(float* dataz, float* datan, float* p_sig, int N);
    void updateMapPoints();
    void reEvalPoints(std::vector<std::shared_ptr<Node3> >& nodes);
    void evalPoints();
    void addNewMeas();
    void updateGPs();

    std::vector<float> obs_valid_xyzglobal;
    std::vector<float> obs_valid_xyznormal;
    std::vector<float> obs_valid_dis;
    std::vector<float> obs_valid_psig;
    std::vector<float> obs_valid_nsig;

    std::vector<float> pose_tr;
    std::vector<float> pose_R;
    int obs_numdata;
    float range_obs_max;

public:
    CustomizeGPIS();
    CustomizeGPIS(GPisMap3Param par);
    CustomizeGPIS(GPisMap3Param par, camParam c);
    ~CustomizeGPIS();
    void reset();

    // to be called as C API
    void update(float* dataz, float* datan, float*, int N);
    bool test(float* x, int dim, int leng, float* res);
    void resetCam(camParam c);

    int getSampleCount();
    bool getAllSamples(float* psamples, int dim, int leng, bool grad = false, bool var = false);
    bool getAllSamples(std::vector<float>& samples, bool grad = false, bool var = false);

private:
    void test_kernel(int thread_idx,
        int start_idx,
        int end_idx,
        float* x,
        float* res);

    void updateGPs_kernel(int thread_idx,
        int start_idx,
        int end_idx,
        std::vector<OcTree*>& nodes_to_update);

    void reEvalPoints_kernel(int thread_idx,
        int start_idx,
        int end_idx,
        std::vector<std::shared_ptr<Node3> >& nodes);

    void evalPoints_kernel(int thread_idx,
        int start_idx,
        int end_idx);

    std::mutex mux;
};

typedef CustomizeGPIS* cusGPMHandle;

#endif

