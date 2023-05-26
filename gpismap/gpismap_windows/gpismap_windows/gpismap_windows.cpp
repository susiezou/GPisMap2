// gpismap_windows.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "run_vs.h"

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
        for (long i = 0; i < file_size; i++) {
            //printf("%f\n", buffer[i]);
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

int main()
{
    std::cout << "Hello World!\n";
    const char* datapath = "C:/Users/zou/source/repos/susiezou/GPisMap2/data/3D/building/depth/391.bin";
    const char* posepath = "C:/Users/zou/source/repos/susiezou/GPisMap2/data/3D/building/pose/391pose.bin";
    long fsize, test_size, posesize=12, camsize=6;
    float* cam = read_bin("C:/Users/zou/source/repos/susiezou/GPisMap2/data/3D/building/391cam.bin", camsize);
    fsize = cam[4] * cam[5];
    float* pose = read_bin(posepath, posesize);
    float* data = read_bin(datapath, fsize);
    test_size = fsize * 3;
    float* testdata = read_bin("C:/Users/zou/source/repos/susiezou/GPisMap2/data/3D/building/depth/391test.bin", test_size);
    
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
    float result[200000] = {-1};
    int flag = test_gpm3d(gm, testdata, 3, test_size, result);
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
