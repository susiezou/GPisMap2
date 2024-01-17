#include "runtime_api.h"
#include <iostream>
#include <fstream>
/// GPisMap (2d)
int create_gpm_instance(GPMHandle *gh){
    *gh = new GPisMap;
    return 1;
}

int delete_gpm_instance(GPMHandle gh){
    if (gh != NULL)
        delete gh;
    return 1;
}


int reset_gpm(GPMHandle gh){
    if (gh != NULL){
        gh->reset();
    }
    return 1;
}

int config_gpm(GPMHandle gh, const char *p_key, void *p_value) {
    if (gh != NULL){
        gh->setParam(p_key, p_value);
        return 1;
    }
    return 0;
}

int update_gpm(GPMHandle gh, float * datax,  float * dataf, int N, float* pose){ // pose[6]
    if (gh != NULL){
        gh->update(datax, dataf, N, pose);
        return 1;
    }
    return 0;
}

int test_gpm(GPMHandle gh, float * x,  int dim,  int leng, float* res){
    if (gh != NULL){
        gh->test(x, dim, leng, res);
        return 1;
    }
    return 0;
}

int get_sample_count_gpm(GPMHandle gh)
{
    if (gh != NULL){
        return gh->getSampleCount();
    }
    return 0;
}

int get_samples_gpm(GPMHandle gh, float * x,  int dim,  int leng, bool grad, bool var){
    if (gh != NULL){
        if (gh->getAllSamples(x, dim, leng, grad, var))
            return 1;
    }
    return 0;
}

/// GPisMap3 (3d)
int create_gpm3d_instance(GPM3Handle *gh){
    *gh = new GPisMap3;
    return 1;
}

int delete_gpm3d_instance(GPM3Handle gh){
    if (gh != NULL){
        delete gh;
        gh = NULL;
    }
    return 1;
}

int reset_gpm3d(GPM3Handle gh){
    if (gh != NULL){
        gh->reset();
    }
    return 1;
}

int set_gpm3d_camparam(GPM3Handle gh, 
                       float fx, 
                       float fy,
                       float cx,
                       float cy,
                       int w,
                       int h){
    if (gh != NULL){
        camParam c(fx, fy, cx, cy, w, h);
        gh->resetCam(c);
        return 1;
    }
    return 0;
}

int update_gpm3d(GPM3Handle gh, float * depth, int numel, float* pose){ // pose[12]
    if (gh != NULL){
        gh->update(depth, numel, pose);
        return 1;
    }
    return 0;
}

int update_scan3d(GPM3Handle gh, float* depth, int numel, float* pose) { // pose[12]
    if (gh != NULL) {
        std::vector<float> pose_vec;
        for (int i = 0; i < 12; i++)
            pose_vec.push_back(pose[i]);
        return gh->update_scan(depth, numel, pose_vec);
    }
    return 0;
}

int add_scan3d(GPM3Handle gh, float* depth, int numel, float* pose) { // pose[12]
    if (gh != NULL) {
        std::vector<float> pose_vec;
        for (int i = 0; i < 12; i++)
            pose_vec.push_back(pose[i]);
        return gh->add_scan(depth, numel, pose_vec);
    }
    return 0;
}

int train_gp3d(GPM3Handle gh) {
    if (gh != NULL) {
        gh->updateGPs();
        return 1;
    }
    return 0;
}

int test_gpm3d(GPM3Handle gh, float * x,  int dim,  int leng, float* res){
    if (gh != NULL){
        gh->test(x, dim, leng, res);
        return 1;
    }
    return 0;
}

int get_sample_count_gpm3d(GPM3Handle gh)
{
    if (gh != NULL){
        return gh->getSampleCount();
    }
    return 0;
}

int get_samples_gpm3d(GPM3Handle gh, float * x,  int dim,  int leng, bool grad, bool var){
    if (gh != NULL){
        if (gh->getAllSamples(x, dim, leng, grad, var))
            return 1;
    }
    return 0;
}

//////// App GP func /////////
int create_gp_func(GPFUNHandle *gh) {
    *gh = new AppGPIS(0.05 * sqrt(3), 0.0004);
    return 1;
}
int update_gp(GPFUNHandle gh, float* data, float* p_sig, int N) {
    if (gh != NULL) {
        gh->train(data, p_sig, N);
        return 1;
    }
    return 0;
}

int test_gp(GPFUNHandle gh, float* x, float* p_sig, int M, float* val, float* var) {
    if (gh != NULL) {
        gh->test(x, p_sig, M, val, var);
        return 1;
    }
    return 0;
}

int update_gp2(GPFUNHandle gh, float* data, float* p_sig, int N) {
    if (gh != NULL) {
        gh->addNewMeas(data, p_sig, N);
        gh->updateGPs();
        return 1;
    }
    return 0;
}

int test_gp2(GPFUNHandle gh, float* x, int M, float* val) {
    if (gh != NULL) {
        gh->test(x, M, val);
        return 1;
    }
    return 0;
}

int reset_gp(GPFUNHandle gh) {
    if (gh != NULL) {
        gh->reset();
    }
    return 1;
}

int delete_gp_instance(GPFUNHandle gh) {
    if (gh != NULL) {
        delete gh;
        gh = NULL;
    }
    return 1;
}