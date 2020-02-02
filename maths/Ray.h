#pragma once

#include "./Vector.h"

namespace Math {
    struct Ray {
        Point origin;
        Direction direction;

        __device__
        Ray(const Point& origin, const Direction& direction) :
            origin(origin),
            direction(Normalize(direction)) {}

        /**
         * Récupère un point du rayon
         * @param t - Distance à l'origine
         */
        __device__
        Point operator()(float t) const {
            return origin + t*direction;
        }
    };
}