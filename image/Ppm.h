#pragma once

#include <fstream>
#include "./AbstractImage.h"

namespace Image {
    struct Ppm : public AbstractImage {
        Ppm(int width, int height) : AbstractImage(width, height) {}

        void Write(const char* filename) const override {
            char* colorBuffer = new char[width*height*3];
            int index = 0;
            for (int y = height - 1; y >= 0; y--)
            for (int x = 0; x < width; x++) {
               const Color& pixel = pixels[x + y*width];
               colorBuffer[index++] = 255.f * pixel.r;
               colorBuffer[index++] = 255.f * pixel.g;
               colorBuffer[index++] = 255.f * pixel.b;
            }

            std::ofstream file(filename, std::ofstream::binary);
            file << "P6\n" << width << " " << height << "\n" << "255\n";
            file.write(colorBuffer, width*height*3);
            file.close();
        }
    };
}