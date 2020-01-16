#pragma once

#include "../maths/Vector.h"

namespace Image {
    using Color = Math::Vector<float>;

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