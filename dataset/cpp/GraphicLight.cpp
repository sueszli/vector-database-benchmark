#include <zeno/utils/vec.h>
#include <zeno/types/UserData.h>
#include <zenovis/Camera.h>
#include <zenovis/Scene.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/ShaderManager.h>
#include <zenovis/opengl/buffer.h>
#include <zenovis/opengl/shader.h>
#include <zeno/types/LightObject.h>

namespace zenovis {
namespace {

struct GraphicLight final : IGraphic {
    Scene *scene;
    zeno::LightData lightData;

    explicit GraphicLight(Scene *scene_, zeno::LightObject *lit) : scene(scene_) {
        //auto nodeid = lit->userData().get("nodeid");
        lightData = lit->get();
        // TODO: implement modify scene->light?
    }
};

}

void MakeGraphicVisitor::visit(zeno::LightObject *obj) {
     this->out_result = std::make_unique<GraphicLight>(this->in_scene, obj);
}

}
