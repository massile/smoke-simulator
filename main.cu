#include "./image/Ppm.h"
#include "./fluid/Renderer.h"
#include "./fluid/Simulator.h"

int main() {
    Fluid::SmokeBall gas;
    Math::Direction light(-.2f, .9f, .2f);
    Math::Point eye(128, 128, -64);

    Image::Ppm image(1280, 720);
    Fluid::Renderer renderer(gas, eye, light);
    Fluid::Simulate(renderer, image);
}
