// gpismap_windows.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "run_vs.h"
#include <pcl/io/ply_io.h>	
#include <pcl/point_types.h>

using namespace std;

float* read_bin(const char* filepath, long& file_size ) {
    FILE* file;
    errno_t err = fopen_s(&file, filepath, "rb");
    if (err == 0) {
        // Get the size of the file
        fseek(file, 0, SEEK_END);
        if (file_size==0)   file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        // Read the binary data into a buffer
        float* buffer = new float[file_size];
        fread(buffer, sizeof(float), file_size, file);

        // Process the binary data (in this case, print the values)
        for (long i = 0; i < 12; i++) {
            printf("%f\n", buffer[i]);
        }

        // Cleanup
        //delete[] buffer;
        fclose(file);
        return buffer;
    }
    else {
        printf("Failed to open the file.\n");
    }
    return NULL;
}

double* read_bin_double(const char* filepath, long& file_size) {
    FILE* file;
    errno_t err = fopen_s(&file, filepath, "rb");
    if (err == 0) {
        // Get the size of the file
        fseek(file, 0, SEEK_END);
        if (file_size == 0)   file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        // Read the binary data into a buffer
        double* buffer = new double[file_size];
        fread(buffer, sizeof(double), file_size, file);

        // Process the binary data (in this case, print the values)
        for (long i = 0; i < file_size; i++) {
            printf("%d\n", buffer[i]);
        }

        // Cleanup
        //delete[] buffer;
        fclose(file);
        return buffer;
    }
    else {
        printf("Failed to open the file.\n");
    }
    return NULL;
}

int main0()
{
    std::cout << "Hello World!\n";
    const std::string frame = "10";
    const std::string filepath = "../../../data/3D/building/";
    const std::string datapath = filepath + "depth/f"+frame+".bin";
    const std::string posepath = filepath + "pose/f"+frame+"pose.bin";
    const std::string campath = filepath + "f" + frame + "cam.bin";
    std::string testpath = filepath + "depth/f"+frame+"test.bin";
    long fsize, test_size, posesize = 12, camsize = 6;
    float* pose = read_bin(posepath.c_str(), posesize);
    float* cam = read_bin(campath.c_str(), camsize);
    fsize = cam[4] * cam[5];
    test_size = 143040*3;//f8:1346752 * 3;
    float* data = read_bin(datapath.c_str(), fsize);
    float* testdata = read_bin(testpath.c_str(), test_size);
    
     
    testpath = "//koko/qianqian/recording_Town10HD/dense_no_occlusion/test_pts/resolution_0.05/10_t.ply";
    pcl::PointCloud<pcl::PointNormal> testCloud;
    if (pcl::io::loadPLYFile<pcl::PointNormal>(testpath.c_str(), testCloud) == -1) {
        PCL_ERROR("Couldn't read the file\n");
        return -1;
    }
    test_size = testCloud.points.size();
    //float* testdata = new float[test_size * 3]();
    //for (size_t i = 0; i < test_size; i++)
    //{
    //    int i3 = i * 3;
    //    // point
    //    const pcl::PointNormal& point = testCloud.points[i];
    //    testdata[i3] = point.x;
    //    testdata[i3 + 1] = point.y;
    //    testdata[i3 + 2] = point.z;
    //    //testdata[i3 + 3] = point.normal_y;
    //}

    GPM3Handle gm = nullptr;
    create_gpm3d_instance(&gm);
    set_gpm3d_camparam(gm,
        cam[0],
        cam[1],
        cam[2],
        cam[3],
        int(cam[4]), int(cam[5])
    );

    int succeed = update_gpm3d(gm, data, fsize, pose);
    delete[] data;
    float* result = new float[test_size * 8]();
    int flag = test_gpm3d(gm, testdata, 3, test_size, result);
    for (size_t i = 0; i < test_size; i++)
    {
        int i8 = i * 8;
        // point
        testCloud.points[i].normal_x = result[i8] + testCloud.points[i].normal_x;
        testCloud.points[i].normal_y = result[i8 + 4];
    }
    //std::cout << max << " " << min << "\n";
    pcl::io::savePLYFileBinary(filepath + "f10out.ply", testCloud);
    delete[] result;
    return  1;

}

int main()
{
    std::cout << "Hello World!\n";
    const std::string filepath = "//koko/qianqian/recording_Town10HD/dense_no_occlusion/";
    const std::string datapath = filepath + "train_pts/10_x.ply";
    const std::string testpath = filepath + "test_pts/resolution_0.05/10_t.ply";
    const std::string priorpath = filepath + "train_pts/10_prior.ply";

    pcl::PointCloud<pcl::PointNormal> dataCloud;
    pcl::PointCloud<pcl::PointNormal> testCloud;
    pcl::PointCloud<pcl::PointNormal> dataCloud2;
    if (pcl::io::loadPLYFile<pcl::PointNormal>(datapath.c_str(), dataCloud) == -1) {
        PCL_ERROR("Couldn't read the file\n");
        return -1;
    }
    pcl::io::loadPLYFile<pcl::PointNormal>(priorpath.c_str(), dataCloud2);
    if (pcl::io::loadPLYFile<pcl::PointNormal>(testpath.c_str(), testCloud) == -1) {
        PCL_ERROR("Couldn't read the file\n");
        return -1;
    }
    long fsize = dataCloud.points.size(), test_size=testCloud.points.size();
    float* testdata = new float[test_size*4]();
    float* data = new float[fsize*8]();
    float* psig = new float[fsize]();
    for (size_t i = 0; i < fsize; i++)
    {
        int i8 = i * 8;
        // point
        const pcl::PointNormal& point = dataCloud.points[i];
        data[i8] = point.x;
        data[i8 + 1] = point.y;
        data[i8 + 2] = point.z;
        data[i8 + 3] = dataCloud2.points[i].normal_x;
        data[i8 + 4] = point.normal_x;
        data[i8 + 5] = point.normal_y;
        data[i8 + 6] = point.normal_z;
        data[i8 + 7] = 0.0005;
        psig[i] = dataCloud2.points[i].normal_y;
    }

    for (size_t i = 0; i < test_size; i++)
    {
        int i3 = i * 4;
        // point
        const pcl::PointNormal& point = testCloud.points[i];
        testdata[i3] = point.x;
        testdata[i3 + 1] = point.y;
        testdata[i3 + 2] = point.z;
        testdata[i3 + 3] = point.normal_y;
    }

    GPFUNHandle gm = nullptr;
    create_gp_func(&gm);

    int succeed = update_gp(gm, data, psig, fsize);
    delete[] data;
    delete[] psig;
    float* result = new float[test_size * 8]();
    int flag = test_gp(gm, testdata, test_size, result);
    delete[] testdata;  
    for (size_t i = 0; i < test_size; i++)
    {
        int i8 = i * 8;
        // point
        testCloud.points[i].normal_x = result[i8] + testCloud.points[i].normal_x;
        testCloud.points[i].normal_y = result[i8 + 4];
    }
    //std::cout << max << " " << min << "\n";
    pcl::io::savePLYFileBinary(filepath + "output/3d_gmmgp/meta_data/nan.ply", testCloud);
    delete[] result;    
    return  1;

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
