#pragma once

#include <fstream>
#include "./AbstractImage.h"

namespace Image {
    struct Ppm : public AbstractImage {
        Ppm(int width, int height) : AbstractImage(width, height) {}

        void Write(const char* filename) const override {
            std::ofstream file(filename);
            file << "P3" << std::endl << width << ' ' << height << std::endl << 255 << std::endl;

            for(int y = height - 1; y >= 0; y--) {
            for(int x = 0; x < width; x++) {
               file << Math::Vector<int>(255.f * pixels[x + y*width]);
            }}
        }
    };
}