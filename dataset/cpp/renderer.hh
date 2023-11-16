#ifndef VKHR_RENDERER_HH
#define VKHR_RENDERER_HH

#include <vkhr/scene_graph.hh>

namespace vkhr {
    class Renderer {
    public:
        virtual ~Renderer() noexcept  =  default;
        virtual void load(const SceneGraph&) = 0;
        virtual void draw(const SceneGraph&) = 0;

        enum Type : int {
            Rasterizer = 0,
            Ray_Tracer = 1,
            Raymarcher = 2,
            Hybrid_LoD = 3,
        };
    };
}

#endif
