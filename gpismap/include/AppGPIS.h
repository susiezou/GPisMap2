#pragma once
#ifndef __APPGPIS_H__
#define __APPGPIS_H__

#include <vector>
#include <memory>
#include <iostream>
#include <cstdint>
#include <Eigen/Dense>
#include "strct.h"
#include "params.h"
#include "octree.h"
#include <mutex>

typedef Eigen::MatrixXf EMatrixX;
typedef Eigen::VectorXf EVectorX;
typedef Eigen::RowVectorXf ERowVectorX;
 
typedef std::vector<std::shared_ptr<Node3> > vecNode3;

class AppGPIS {
    EMatrixX x;
    EMatrixX L;
    EVectorX alpha;
    std::vector<float> gradflag;
    std::vector<float> obs_y;
    EVectorX prior_sig;
    EVectorX p_sig_valid;

    onGPISparam param;     // defined in strct.h
    // currently noise param is not effective
    float three_over_scale;
    bool trained;
    int nSamples;
    OcTree* t;
    std::unordered_set<OcTree*> activeSet;
    std::mutex mux;

public:
    AppGPIS() :param(DEFAULT_MAP_SCALE_PARAM, DEFAULT_MAP_NOISE_PARAM, DEFAULT_MAP_NOISE_PARAM),
        three_over_scale(3.0 / (DEFAULT_MAP_SCALE_PARAM * DEFAULT_MAP_SCALE_PARAM)),
        trained(false),
        t(nullptr),
        nSamples(0) { }

    AppGPIS(float s, float n) :param(s, n, n),
        three_over_scale(3.0 / (s * s)),
        trained(false),
        t(nullptr),
        nSamples(0) {}

    void reset();
    bool isTrained() { return trained; }
    void setGPScaleParam(float l) { param.scale = l; }

    void train(const float * samples, float* p_sig, int N);
    void train(const vecNode3& samples);
    void updateGPs();
    void updateGPs_kernel(int thread_idx, int start_idx, int end_idx, std::vector<OcTree*>& nodes_to_update);
    void addNewMeas(const float* samples, float* p_sig, int N);
    void addNewMeas_kernel(const float* samples, float*, int, int, int);

    void test(const float* samples, float* p_sig, int M, float* val, float* var);
    bool test(float* x, int leng, float* res);
    void test_kernel(int thread_idx, int start_idx, int end_idx, float* x, float* res);
    void test_kernel_s(int thread_idx, int start_idx, int end_idx, float* x, float* res);
    void testSinglePoint(const EVectorX& xt, float& val, float grad[], float var[]);

};
typedef AppGPIS* GPFUNHandle;
#endif


