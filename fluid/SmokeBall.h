#pragma once

#include "../maths/Vector.h"

namespace Fluid {
    struct SmokeBall {
        Math::Dimension cubeSize;
        float* attenuation;

        SmokeBall(const Math::Dimension& cubeSize) :
            cubeSize(cubeSize), attenuation(new float[cubeSize.x*cubeSize.y*cubeSize.z]) {
            for (int x = 0; x < cubeSize.x; x++)
            for (int y = 0; y < cubeSize.y; y++)
            for (int z = 0; z < cubeSize.z; z++) {
                const Math::Point voxel(x, y, z);
                attenuation[VoxelIndex(voxel)] = CalculateAttenuationAt(voxel);
            }
        }

        inline float AttenuationAt(const Math::Point& voxel) const {
            // Point à l'extérieur du cube, donc pas de matière
            if (voxel.x < 0 || voxel.x >= cubeSize.x ||
                voxel.y < 0 || voxel.y >= cubeSize.y ||
                voxel.z < 0 || voxel.z >= cubeSize.z) return 0.f;
            // Coefficient calculé pour une longueur d'onde de 570nm
            return attenuation[VoxelIndex(voxel)];
        }

    private:
        inline float CalculateAttenuationAt(const Math::Point& voxel) const {
            const float distanceFromCenter = Math::Length(voxel - cubeSize/2.f);
            return 0.018f / (1.f + powf(1.2f, distanceFromCenter - 60.f));
        }

        inline int VoxelIndex(const Math::Vector<int>& voxel) const {
            return voxel.x + voxel.y*cubeSize.x + voxel.z*cubeSize.x*cubeSize.y;
        }
    };
}