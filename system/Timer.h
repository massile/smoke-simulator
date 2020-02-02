#pragma once

#include <chrono>
#include <iostream>

namespace System {
    using namespace std::chrono;
    struct Timer {
        time_point<high_resolution_clock> startTime;

        Timer() {
            Start();
        }

        ~Timer() {
            Stop();
        }

        void Start() {
            startTime = high_resolution_clock::now();
        }

        void Stop() {
            auto endTime = high_resolution_clock::now();
            auto elapsedTime = duration_cast<std::chrono::seconds>(endTime - startTime).count();
            std::cout << "Le calcul a pris " << elapsedTime << " secondes." << std::endl;
        }
    };
}