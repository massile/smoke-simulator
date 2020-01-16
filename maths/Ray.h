#pragma once

#include "./Vector.h"

namespace Math {
    struct Ray {
        Point origin;
        Direction direction;

        Ray(const Point& origin, const Direction& direction) :
            origin(origin),
            direction(Normalize(direction)) {}

        /**
         * Récupère un point du rayon
         * @param t - Distance à l'origine
         */
        Point operator()(float t) const {
            return origin + t*direction;
        }
    };
}