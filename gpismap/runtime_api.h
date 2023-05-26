#pragma once

#ifndef GPISMAP_RUNTIME_API_H
#define GPISMAP_RUNTIME_API_H

#define IMPORT_DLL extern "C" __declspec(dllexport)

#include "GPisMap.h"
#include "GPisMap3.h"



//2D
IMPORT_DLL int create_gpm_instance(GPMHandle *gh);
IMPORT_DLL int delete_gpm_instance(GPMHandle gh);
IMPORT_DLL int reset_gpm(GPMHandle gh);
IMPORT_DLL int config_gpm(GPMHandle gh, const char* p_key, void *p_value);
IMPORT_DLL int update_gpm(GPMHandle gh, float * datax,  float * dataf, int N, float* pose); // pose[6]
IMPORT_DLL int test_gpm(GPMHandle gh, float * x,  int dim,  int leng, float* res);
IMPORT_DLL int get_sample_count_gpm(GPMHandle gh);
IMPORT_DLL int get_samples_gpm(GPMHandle gh, float * x,  int dim,  int leng, bool grad, bool var);

// 3D
IMPORT_DLL int create_gpm3d_instance(GPM3Handle *gh);
IMPORT_DLL int delete_gpm3d_instance(GPM3Handle gh);
IMPORT_DLL int reset_gpm3d(GPM3Handle gh);
IMPORT_DLL int set_gpm3d_camparam(GPM3Handle gh,
                       float fx, 
                       float fy,
                       float cx,
                       float cy,
                       int w,
                       int h);
IMPORT_DLL int update_gpm3d(GPM3Handle gh, float * depth, int numel, float* pose); // pose[12]
IMPORT_DLL int test_gpm3d(GPM3Handle gh, float * x,  int dim,  int leng, float* res);
IMPORT_DLL int get_sample_count_gpm3d(GPM3Handle gh);
IMPORT_DLL int get_samples_gpm3d(GPM3Handle gh, float * x,  int dim,  int leng, bool grad, bool var);



#endif