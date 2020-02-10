#include "./image/Ppm.h"
#include "./fluid/Renderer.h"
#include "./system/Timer.h"

int main() {
    Image::Ppm image(1280, 720);

    Fluid::SmokeBall gas(256);
    Math::Direction light(-.2f, .9f, .2f);
    Math::Point eye(128, 128, 0);

    Fluid::Renderer renderer(gas, eye, light);
    renderer.RenderImage(image);

    image.Write("test.ppm");
}
