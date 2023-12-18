#pragma once

#ifndef GPISMAP_RUNTIME_API_H
#define GPISMAP_RUNTIME_API_H


#include "GPisMap.h"
#include "GPisMap3.h"


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
int update_scan3d(GPM3Handle gh, float* depth, int numel, float* pose);
int test_gpm3d(GPM3Handle gh, float* x, int dim, int leng, float* res);
int get_sample_count_gpm3d(GPM3Handle gh);
int get_samples_gpm3d(GPM3Handle gh, float* x, int dim, int leng, bool grad, bool var);



#endif
