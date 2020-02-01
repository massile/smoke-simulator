#include "./image/Ppm.h"
#include "./fluid/Renderer.h"

int main() {
    Image::Ppm image(256, 256);

    Fluid::SmokeBall gas(256);
    Math::Point light(-.2f, .2f, -.9f);
    Math::Point eye(128, 128, 0);

    Fluid::Renderer renderer(gas, eye, light);
    renderer.RenderImage(image);

    image.Write("test.ppm");
}
