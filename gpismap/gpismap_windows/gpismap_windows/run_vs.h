#pragma once

#ifndef GPISMAP_RUNTIME_API_H
#define GPISMAP_RUNTIME_API_H


#include "GPisMap.h"
#include "GPisMap3.h"
#include "AppGPIS.h"


// 3D
int create_gpm3d_instance(GPM3Handle* gh);
int delete_gpm3d_instance(GPM3Handle gh);
int reset_gpm3d(GPM3Handle gh);
int set_gpm3d_camparam(GPM3Handle gh,
    float fx,
    float fy,
    float cx,
    float cy,
    int w,
    int h);
int update_gpm3d(GPM3Handle gh, float* depth, int numel, float* pose); // pose[12]
int test_gpm3d(GPM3Handle gh, float* x, int dim, int leng, float* res);
int get_sample_count_gpm3d(GPM3Handle gh);
int get_samples_gpm3d(GPM3Handle gh, float* x, int dim, int leng, bool grad, bool var);


//APP GP
int create_gp_func(GPFUNHandle* gh);
int update_gp(GPFUNHandle gh, float* data, float* p_sig, int N);
int test_gp(GPFUNHandle gh, float* x, int M, float* val);
int reset_gp(GPFUNHandle gh);
int delete_gp_instance(GPFUNHandle gh);

#endif
