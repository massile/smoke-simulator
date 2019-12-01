#include "./image/Ppm.h"

int main() {
    Image::Ppm image(800, 600);

    for (int x = 0; x < image.width; x++)
    for (int y = 0; y < image.height; y++) {
        image.pixels[x + y*image.width] = 
            Image::Color(0, .5, 1);
    }

    image.Write("test.ppm");
}
