#pragma once

#include "./Renderer.h"

namespace Fluid {
    __global__ void Advection(float* prevAtt, float* attenuation, Math::Direction* velocity, float dt, float dissipation);

    void Simulate(Renderer& renderer, Image::AbstractImage& image) {
        SmokeBall& gas = renderer.gas;
        const float dt = 1.f;
        const dim3 numBlocks = dim3(8,8,8);
        const dim3 numThreads = dim3(CUBE_SIZE/8,CUBE_SIZE/8,CUBE_SIZE/8);

        for (int frame = 0; frame < 250; frame++) {
            renderer.RenderImage(image);

            Advection<<<numThreads, numBlocks>>>(gas.attenuation.previous, gas.attenuation.current, gas.velocity.current, dt, 0.98f);
            cudaDeviceSynchronize();
            gas.attenuation.Swap();

            std::stringstream fileName;
            fileName << "out/" << frame << ".ppm";
            image.Write(fileName.str().c_str());
        }
    }

    __global__
    void Advection(float* prevAtt, float* attenuation, Math::Direction* velocity, float dt, float dissipation) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        const Math::Direction v = velocity[VoxelIndex(voxel)];
        attenuation[VoxelIndex(voxel)] = dissipation * prevAtt[VoxelIndex(voxel - v*dt)];
    }
}