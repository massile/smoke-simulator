#pragma once

#include "../maths/Vector.h"

namespace Fluid {
    __device__ __host__
    inline int VoxelIndex(const Math::Vector<int>& voxel, int cubeSize) {
        return voxel.x + voxel.y*cubeSize + voxel.z*cubeSize*cubeSize;
    }

    __global__
    void CalculateAttenuation(float* attenuation, int cubeSize) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        const float distanceFromCenter = Math::Length(voxel - cubeSize/2.f);
        attenuation[VoxelIndex(voxel, cubeSize)] = 0.018f / (1.f + powf(1.2f, distanceFromCenter - 60.f));
    }

    struct SmokeBall {
        int cubeSize;
        float* attenuation;

        SmokeBall(int cubeSize) : cubeSize(cubeSize) {
            dim3 numThreads(cubeSize/8, cubeSize/8, cubeSize/8);
            cudaMallocManaged(&attenuation, sizeof(float)*cubeSize*cubeSize*cubeSize);

            CalculateAttenuation<<<numThreads, dim3(8,8,8)>>>(attenuation, cubeSize);
            cudaDeviceSynchronize();
        }

        __device__
        inline float AttenuationAt(const Math::Point& voxel) const {
            // Point à l'extérieur du cube, donc pas de matière
            if (voxel.x < 0 || voxel.x >= cubeSize ||
                voxel.y < 0 || voxel.y >= cubeSize ||
                voxel.z < 0 || voxel.z >= cubeSize) return 0.f;
            // Coefficient calculé pour une longueur d'onde de 570nm
            return attenuation[VoxelIndex(voxel, cubeSize)];
        }
    };
}