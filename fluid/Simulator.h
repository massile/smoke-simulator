#pragma once

#include "./Renderer.h"

namespace Fluid {
    template<typename T>
    __global__
    void Advection(T* prevValue, T* currentValue, Math::Direction* velocity, float dt, float dissipation) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        const Math::Direction v = velocity[VoxelIndex(voxel)];
        currentValue[VoxelIndex(voxel)] = dissipation * prevValue[VoxelIndex(voxel - v*dt)];
    }

    __global__
    void Buoyancy(Math::Direction* prevVelocity, Math::Direction* velocity, float* temperature, float* attenuation, float dt) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        const int index = VoxelIndex(voxel);
        velocity[index] = (
            prevVelocity[index] + Math::Direction(.0f, dt*temperature[index] - attenuation[index]*0.05f, .0f)
        );
    }

    void Simulate(Renderer& renderer, Image::AbstractImage& image) {
        SmokeBall& gas = renderer.gas;
        const float dt = 1.f;
        const dim3 numBlocks = dim3(8,8,8);
        const dim3 numThreads = dim3(CUBE_SIZE/8,CUBE_SIZE/8,CUBE_SIZE/8);

        for (int frame = 0; frame < 250; frame++) {
            renderer.RenderImage(image);

            Advection<<<numThreads, numBlocks>>>(gas.velocity.previous, gas.velocity.current, gas.velocity.current, dt, 1.f);
            cudaDeviceSynchronize();
            gas.velocity.Swap();

            Advection<<<numThreads, numBlocks>>>(gas.temperature.previous, gas.temperature.current, gas.velocity.current, dt, 0.998f);
            cudaDeviceSynchronize();
            gas.temperature.Swap();

            Advection<<<numThreads, numBlocks>>>(gas.attenuation.previous, gas.attenuation.current, gas.velocity.current, dt, 0.9999f);
            cudaDeviceSynchronize();
            gas.attenuation.Swap();

            Buoyancy<<<numThreads, numBlocks>>>(
                gas.velocity.previous, gas.velocity.current, gas.temperature.current, gas.attenuation.current, dt);
            cudaDeviceSynchronize();
            gas.velocity.Swap();

            std::stringstream fileName;
            fileName << "out/" << frame << ".ppm";
            image.Write(fileName.str().c_str());
        }
    }
}