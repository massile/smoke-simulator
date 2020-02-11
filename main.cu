#include "./image/Ppm.h"
#include "./fluid/Renderer.h"
#include "./system/Timer.h"

int main() {
    Fluid::SmokeBall gas;
    Math::Direction light(-.2f, .9f, .2f);
    Math::Point eye(128, 128, 0);

    Fluid::Renderer renderer(gas, eye, light);
    renderer.MakeVideo();
}
