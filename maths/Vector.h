#pragma once

#include <cmath>

namespace Math {
    template<typename T>
    struct Vector {
        union {
            struct { T x; T y; T z; };
            struct { T r; T g; T b; };
        };

        template<typename U>
        __device__ __host__
        Vector(const Vector<U>& u) : x(T(u.x)), y(T(u.y)), z(T(u.z)) {}
        
        __device__ __host__
        Vector(T x, T y, T z) : x(x), y(y), z(z) {}
        
        __device__ __host__
        Vector(T a) : x(a), y(a), z(a) {}
        
        __device__ __host__
        Vector() : x(0), y(0), z(0) {}

        __device__ __host__
        Vector<T> operator+(const Vector<T>& v) const {
            return Vector(x + v.x, y + v.y, z + v.z);
        }
        
        __device__ __host__
        Vector<T> operator*(const T a) const {
            return Vector(x*a, y*a, z*a);
        }

        __device__ __host__
        Vector<T> operator-(const Vector<T>& v) const {
            return Vector(x - v.x, y - v.y, z - v.z);
        }
        
        __device__ __host__
        Vector<T> operator/(const T a) const {
            return Vector(x/a, y/a, z/a);
        }

        __device__ __host__
        Vector<T> operator+=(const Vector<T>& v) {
            x += v.x; y += v.y; z += v.z;
            return *this;
        }

        __device__ __host__
        Vector<T> operator-=(const Vector<T>& v) {
            x -= v.x; y -= v.y; z -= v.z;
            return *this;
        }
        
        __device__ __host__
        Vector<T> operator*=(const T a) {
            x *= a; y *= a; z *= a;
            return *this;
        }
        
        __device__ __host__
        Vector<T> operator/=(const T a) {
            x /= a; y /= a; z /= a;
            return *this;
        }
    };

    template<typename T>
    __device__ __host__
    Vector<T> operator*(const T a, const Vector<T>& v) {
        return v * a;
    }
    
    template<typename T>
    std::ostream& operator<<(std::ostream& out, const Vector<T>& v) {
        return out << v.x << ' ' << v.y << ' ' << v.z << std::endl;
    }

    template<typename T>
    __device__ __host__
    float Dot(const Vector<T>& u, const Vector<T>& v) {
        return u.x*v.x + u.y*v.y + u.z*v.z;
    }

    template<typename T>
    __device__ __host__
    float LengthSquared(const Vector<T>& u) {
        return Dot(u, u);
    }

    template<typename T>
    __device__ __host__
    float Length(const Vector<T>& u) {
        return sqrtf(LengthSquared(u));
    }

    template<typename T>
    __device__ __host__
    Vector<T> Normalize(const Vector<T>& u) {
        return u / Length(u);
    }

    using Point = Vector<float>;
    using Direction = Vector<float>;
    using Dimension = Vector<int>;
}