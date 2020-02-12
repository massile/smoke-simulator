#pragma once

namespace System {
    template<typename T>
    struct DoubleBuffer {
        T* previous;
        T* current;

        DoubleBuffer() = default;
        
        DoubleBuffer(int nbElements) {
            cudaMallocManaged(&previous, sizeof(T) * nbElements);
            cudaMallocManaged(&current, sizeof(T) * nbElements);
        }

        ~DoubleBuffer() {
            cudaFree(current);
            cudaFree(previous);
        }

        void Swap() {
            T* tmp = current;
            current = previous;
            previous = tmp;
        }
    };
}