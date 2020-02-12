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

            Advection<<<numThreads, numBlocks>>>(gas.attenuation.previous, gas.attenuation.current, gas.velocity.current, dt, 0.98f);
            cudaDeviceSynchronize();
            gas.attenuation.Swap();

            Advection<<<numThreads, numBlocks>>>(gas.temperature.previous, gas.temperature.current, gas.velocity.current, dt, 0.999f);
            cudaDeviceSynchronize();
            gas.temperature.Swap();

            std::stringstream fileName;
            fileName << "out/" << frame << ".ppm";
            image.Write(fileName.str().c_str());
        }
    }
}