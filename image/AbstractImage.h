#pragma once

namespace Image {

    struct Color {
        float r;
        float g;
        float b;

        Color() : r(0), g(0), b(0) {}
        Color(float r, float g, float b) : r(r), g(g), b(b) {}
    };

    struct AbstractImage {
        int width;
        int height;
        Color* pixels;

        AbstractImage(int width, int height) :
            width(width),
            height(height),
            pixels(new Color[width * height]) {}

        virtual void Write(const char* filename) const = 0;
    };

}