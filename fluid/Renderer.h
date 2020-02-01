#pragma once

#include "./SmokeBall.h"
#include "../maths/Ray.h"
#include "../image/AbstractImage.h"

#define SKY_COLOR gas.AttenuationAt(ptGas)*STEP

namespace Fluid {
    struct Renderer {
        static constexpr float INITIAL_INTENSITY = 1.f;
        static constexpr float STEP = 0.001f;

        SmokeBall gas;
        Math::Point light;
        Math::Point eye;

        Renderer(const SmokeBall& gas, const Math::Point& eye, const Math::Point& light) :
            gas(gas), eye(eye), light(light) {}
        
        void RenderImage(const Image::AbstractImage& image) {
            for (int x = 0; x < image.width; x++)
            for (int y = 0; y < image.height; y++) {
                const Math::Direction toPixel(float(x)/image.width - .5f, float(y)/image.height - .5f, .5f);
                image.pixels[x + y*image.width] = RaytracePixel(Math::Ray(eye, toPixel));   
            }
        }

    private:
        Image::Color RaytracePixel(const Math::Ray& rayToPixel) const {
            float totalIntensity = 0.f;
            float intensityEye = INITIAL_INTENSITY;
            for (int t = 0; t < gas.cubeSize.z; t++) {
                const Math::Point ptGas = rayToPixel(t);
                intensityEye *= 1.f - gas.AttenuationAt(ptGas)*STEP;

                float intensityLight = intensityEye;
                Math::Ray rayToLight(ptGas, light - ptGas);
                for (int t2 = 0; t2 < gas.cubeSize.y; t2++) {
                    intensityLight *= 1.f - gas.AttenuationAt(rayToLight(t2))*STEP;
                }
                totalIntensity += SKY_COLOR * intensityLight;
            }
            // Correction gamma
            return fmin(powf(totalIntensity, 1.f/1.22f), 1.f);
        }
    };
}