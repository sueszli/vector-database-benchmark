#pragma once

#include "FlingMath.h"
#include "Serilization.h"

namespace Fling
{
    /**
     * @brief   Simple representation of a directional light for Fling. Needs to be 16 bytes aligned
     *          for Vulkan 
     */
    struct alignas(16) DirectionalLight
    {
        alignas(16) glm::vec4 DiffuseColor { 1.0f };
        alignas(16) glm::vec4 Direction { 1.0f, -1.0f, -0.5f, 1.0f  };
		alignas(4)  float Intensity = 1.0f;

        template<class Archive>
        void serialize(Archive & t_Archive);
    };

     /** Serilazation to an archive */
    template<class Archive>
    void DirectionalLight::serialize(Archive & t_Archive)
    {
        t_Archive( 
            cereal::make_nvp("DIFFUSE_X", DiffuseColor.x),
            cereal::make_nvp("DIFFUSE_Y", DiffuseColor.y),
            cereal::make_nvp("DIFFUSE_Z", DiffuseColor.z),
            cereal::make_nvp("DIFFUSE_W", DiffuseColor.w),

            cereal::make_nvp("DIRECTION_X", Direction.x),
            cereal::make_nvp("DIRECTION_Y", Direction.y),
            cereal::make_nvp("DIRECTION_Z", Direction.z),

            cereal::make_nvp("INTENSITY", Intensity)
        );
    }
}   // namespace Fling