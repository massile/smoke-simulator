#pragma once

#include "../maths/Vector.h"

#define CUBE_SIZE 256

namespace Fluid {
    __device__ float AttenuationAt(float* attenuation, const Math::Point& voxel);
    __device__ int VoxelIndex(const Math::Vector<int>& voxel);
    __global__ void InitializeAttenuation(float maxAttenuation, float* attenuation);

    struct SmokeBall {
        float* attenuation;
        dim3 numThreads;

        SmokeBall() : numThreads(dim3(CUBE_SIZE/8, CUBE_SIZE/8, CUBE_SIZE/8)) {
            cudaMallocManaged(&attenuation, sizeof(float)*CUBE_SIZE*CUBE_SIZE*CUBE_SIZE);
            const float maxAttenuation = 4.f * 3.14159265f / 570.f;
            InitializeAttenuation<<<numThreads, dim3(8,8,8)>>>(maxAttenuation, attenuation);
            cudaDeviceSynchronize();
        }

        ~SmokeBall() {
            cudaFree(attenuation);
        }
    };

    __device__
    int VoxelIndex(const Math::Vector<int>& voxel) {
        return voxel.x + voxel.y*CUBE_SIZE + voxel.z*CUBE_SIZE*CUBE_SIZE;
    }

    __device__
    float AttenuationAt(float* attenuation, const Math::Point& voxel) {
        // Point à l'extérieur du cube, donc pas de matière
        if (voxel.x < 0 || voxel.x >= CUBE_SIZE ||
            voxel.y < 0 || voxel.y >= CUBE_SIZE ||
            voxel.z < 0 || voxel.z >= CUBE_SIZE) return 0.f;
        return attenuation[VoxelIndex(voxel)];
    }

    __global__
    void InitializeAttenuation(float maxAttenuation, float* attenuation) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        const float distanceFromCenter = Math::Length(voxel - CUBE_SIZE/2.f);
        attenuation[VoxelIndex(voxel)] = maxAttenuation / (1.f + powf(1.2f, distanceFromCenter - 60.f));
    }
}