#pragma once

#include <string>
#include <sstream>
#include "./SmokeBall.h"
#include "../maths/Ray.h"
#include "../image/AbstractImage.h"

namespace Fluid {
    __global__ void RayTrace(Image::Color* pixels, int width, int height, float* attenuation, Math::Direction light, Math::Point eye);
    __device__ Image::Color RaytracePixel(const Math::Ray& rayToPixel, float* att, const Math::Direction& light, const Math::Point& eye);

    struct Renderer {
        SmokeBall gas;
        Math::Direction light;
        Math::Point eye;

        Renderer(const SmokeBall& gas, const Math::Point& eye, const Math::Direction& light) :
            gas(gas), eye(eye), light(light) {}
        
        void RenderImage(const Image::AbstractImage& image) {
            dim3 numThreads(image.width/8, image.height/8);
            RayTrace<<<numThreads, dim3(8,8,8)>>>(image.pixels, image.width, image.height, gas.attenuation.current, light, eye);
            cudaDeviceSynchronize();            
        }
    };

    __global__
    void RayTrace(Image::Color* pixels, int width, int height, float* attenuation, Math::Direction light, Math::Point eye) {
        int x = threadIdx.x + blockDim.x*blockIdx.x;
        int y = threadIdx.y + blockDim.y*blockIdx.y;
        if (x >= width || y >= height) return;

        const float xNormalized = (float(x)/width - .5f) * width/height;
        const float yNormalized = float(y)/height - .5f;
        const Math::Direction toPixel(xNormalized, yNormalized, .5f);
        pixels[x + y*width] = RaytracePixel(Math::Ray(eye, toPixel), attenuation, light, eye);
    }

    __device__
    Image::Color RaytracePixel(const Math::Ray& rayToPixel, float* att, const Math::Direction& light, const Math::Point& eye) {
        float totalIntensity = 0.f;
        float intensityEye = 1.f;
        for (int t = 0; t < CUBE_SIZE; t++) {
            const Math::Point ptGas = rayToPixel(t);
            const float attenuation = AttenuationAt(att, ptGas);
            if (attenuation < 0.001f) {
                continue;
            }
            intensityEye *= fmax(1.f - attenuation, 0.f);

            float intensityLight = intensityEye;
            Math::Ray rayToLight(ptGas, light);
            for (int t2 = 0; t2 < CUBE_SIZE; t2++) {
                const float attenuation2 = AttenuationAt(att, rayToLight(t2));
                if (attenuation2 < 0.001f) {
                    break;
                }
                intensityLight *= fmax(1.f - attenuation2, 0.f);
            }
            totalIntensity += attenuation * intensityLight;
            if (intensityEye < 0.001f) {
                break;
            }
        }
        // Correction gamma
        return fmin(powf(totalIntensity, 1.f/1.22f), 1.f);
    }
}