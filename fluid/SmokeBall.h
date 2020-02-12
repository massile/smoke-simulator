#pragma once

#include "../system/DoubleBuffer.h"
#include "../maths/Vector.h"

#define CUBE_SIZE 256
#define NB_ELEMENTS CUBE_SIZE*CUBE_SIZE*CUBE_SIZE

namespace Fluid {
    __device__ float AttenuationAt(float* attenuation, const Math::Point& voxel);
    __device__ int VoxelIndex(const Math::Vector<int>& voxel);
    __global__ void InitializeAttenuation(float maxAttenuation, float* attenuation);
    __global__ void InitializeVelocity(Math::Direction initialVelocity, Math::Direction* velocity);
    __global__ void InitializeTemperature(float maxTemperature, float* temperature);

    struct SmokeBall {
        System::DoubleBuffer<float> attenuation = System::DoubleBuffer<float>(NB_ELEMENTS);
        System::DoubleBuffer<float> temperature = System::DoubleBuffer<float>(NB_ELEMENTS);
        System::DoubleBuffer<float> pressure = System::DoubleBuffer<float>(NB_ELEMENTS);
        System::DoubleBuffer<Math::Direction> velocity = System::DoubleBuffer<Math::Direction>(NB_ELEMENTS);

        SmokeBall() {
            dim3 numThreads = dim3(CUBE_SIZE/8, CUBE_SIZE/8, CUBE_SIZE/8);
            dim3 numBlocks = dim3(8,8,8);

            const float maxAttenuation = 4.f * 3.14159265f / 570.f;
            InitializeAttenuation<<<numThreads, numBlocks>>>(maxAttenuation, attenuation.previous);
            InitializeAttenuation<<<numThreads, numBlocks>>>(maxAttenuation, attenuation.current);

            const Math::Direction initialVelocity(0.f, .01f, 0.f);
            InitializeVelocity<<<numThreads, numBlocks>>>(initialVelocity, velocity.previous);
            InitializeVelocity<<<numThreads, numBlocks>>>(initialVelocity, velocity.current);

            const float maxTemperature = .3f;
            InitializeTemperature<<<numThreads, numBlocks>>>(maxTemperature, temperature.previous);
            InitializeTemperature<<<numThreads, numBlocks>>>(maxTemperature, temperature.current);

            cudaDeviceSynchronize();
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

    __global__
    void InitializeVelocity(Math::Direction initialVelocity, Math::Direction* velocity) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        velocity[VoxelIndex(voxel)] = initialVelocity;
    }

    __global__
    void InitializeTemperature(float maxTemperature, float* temperature) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;
        
        const Math::Point voxel(x, y, z);
        const float distanceFromCenter = Math::Length(voxel - Math::Direction(0, 30, 0) - CUBE_SIZE/2.f);
        temperature[VoxelIndex(voxel)] = maxTemperature / (1.f + powf(1.2f, distanceFromCenter - 80.f));
    }
    
}