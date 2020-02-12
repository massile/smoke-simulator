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

    __global__
    void Divergence(Math::Direction* velocity, float* divergence) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        divergence[VoxelIndex(voxel)] = .5f * (
            velocity[VoxelIndex(voxel + Math::Direction( 1,  0,  0))].x -
            velocity[VoxelIndex(voxel + Math::Direction(-1,  0,  0))].x +
            velocity[VoxelIndex(voxel + Math::Direction( 0,  1,  0))].y -
            velocity[VoxelIndex(voxel + Math::Direction( 0, -1,  0))].y +
            velocity[VoxelIndex(voxel + Math::Direction( 0,  0,  1))].z -
            velocity[VoxelIndex(voxel + Math::Direction( 0,  0, -1))].z
        );
    }

    __global__
    void Zero(float* values) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        values[VoxelIndex(voxel)] = 0.f;
    }

    __global__
    void PressureSolve(float* prevPressure, float* pressure, float* divergence) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;

        const Math::Point voxel(x, y, z);
        const float sum = (
            prevPressure[VoxelIndex(voxel + Math::Direction( 1,  0,  0))] +
            prevPressure[VoxelIndex(voxel + Math::Direction(-1,  0,  0))] +
            prevPressure[VoxelIndex(voxel + Math::Direction( 0,  1,  0))] +
            prevPressure[VoxelIndex(voxel + Math::Direction( 0, -1,  0))] +
            prevPressure[VoxelIndex(voxel + Math::Direction( 0,  0,  1))] +
            prevPressure[VoxelIndex(voxel + Math::Direction( 0,  0, -1))]
        );
        const int index = VoxelIndex(voxel); 
        pressure[index] = (sum - divergence[index])/6.f;
    }

    __global__
    void SubtractGradientToPressure(Math::Direction* prevVelocity, Math::Direction* velocity, float* pressure) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;
        const Math::Point voxel(x, y, z);

        Math::Direction grad;
        grad.x = (
            pressure[VoxelIndex(voxel + Math::Direction(1, 0, 0))] - 
            pressure[VoxelIndex(voxel + Math::Direction(-1, 0, 0))]
        );
        grad.y = (
            pressure[VoxelIndex(voxel + Math::Direction(0, 1, 0))] - 
            pressure[VoxelIndex(voxel + Math::Direction(0, -1, 0))]
        );
        grad.z = (
            pressure[VoxelIndex(voxel + Math::Direction(0, 0, 1))] - 
            pressure[VoxelIndex(voxel + Math::Direction(0, 0, -1))]
        );

        const int index = VoxelIndex(voxel);
        velocity[index] = prevVelocity[index] - grad;
    }
    
    void Simulate(Renderer& renderer, Image::AbstractImage& image) {
        SmokeBall& gas = renderer.gas;
        const float dt = 1.f;
        const dim3 numBlocks = dim3(8,8,8);
        const dim3 numThreads = dim3(CUBE_SIZE/8,CUBE_SIZE/8,CUBE_SIZE/8);
        System::DoubleBuffer<float> divergence = System::DoubleBuffer<float>(NB_ELEMENTS);

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

            Divergence<<<numThreads, numBlocks>>>(gas.velocity.current, divergence.current);
            cudaDeviceSynchronize();
            Zero<<<numThreads, numBlocks>>>(gas.pressure.current);
            cudaDeviceSynchronize();

            for (int i = 0; i < 40; i++) {
                PressureSolve<<<numThreads, numBlocks>>>(gas.pressure.previous, gas.pressure.current, divergence.current);
                cudaDeviceSynchronize();
                gas.pressure.Swap();
            }

            SubtractGradientToPressure<<<numThreads, numBlocks>>>(gas.velocity.previous, gas.velocity.current, gas.pressure.current);
            cudaDeviceSynchronize();
            gas.velocity.Swap();

            std::stringstream fileName;
            fileName << "out/" << frame << ".ppm";
            image.Write(fileName.str().c_str());
        }
    }
}