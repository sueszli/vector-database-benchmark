#include <iostream>

#include "sre/Texture.hpp"
#include "sre/Renderer.hpp"
#include "sre/Material.hpp"
#include "sre/Inspector.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <sre/SDLRenderer.hpp>
#include <sre/Resource.hpp>
#include <sre/impl/GL.hpp>

using namespace sre;

class CustomMeshLayoutExample{
public:
    CustomMeshLayoutExample(){
        r.init();


        std::vector<glm::ivec4> positions({
                                                 {0, 1,0,1},
                                                 {0, 0,0,1},
                                                 {1, 0,0,1}
                                         });
        std::vector<glm::vec4> colors({
                                              {1, 0,0,1},
                                              {0, 1,0,1},
                                              {0, 0,1,1},

                                      });

        mesh = Mesh::create()
                .withAttribute("posxyzw",positions)
                .withAttribute("vertex_color",colors)
                .build();

        std::string vertexShaderSource =  R"(#version 330
#if defined(GL_ES) && __VERSION__ == 100
in vec4 posxyzw;
#else
in ivec4 posxyzw;
#endif
in vec4 vertex_color;
out vec4 vColor;

#pragma include "global_uniforms_incl.glsl"

void main(void) {
    gl_Position = g_projection * g_view * g_model * vec4(posxyzw);
    vColor = vertex_color;
}
)";
        std::string fragmentShaderSource = R"(#version 330
out vec4 fragColor;
in vec4 vColor;

void main(void)
{
    fragColor = vColor;
}
)";
        Resource::set("custom-mesh-vert.glsl", vertexShaderSource);
        Resource::set("custom-mesh-frag.glsl",fragmentShaderSource);


        mat1 = Shader::create()
                .withSourceResource("custom-mesh-vert.glsl",ShaderType::Vertex)
                .withSourceResource("custom-mesh-frag.glsl", ShaderType::Fragment)
                .build()->createMaterial();

        std::string info;
        bool isValid = mat1->getShader()->validateMesh(mesh.get(), info);
        std::cout << "Mesh is valid: "<<isValid<<"\ninfo: "<<info<<std::endl;

        r.frameRender = [&](){
            render();
        };
        r.startEventLoop();
    }

    void render(){
        auto rp = RenderPass::create()
                .withCamera(camera)
                .withClearColor(true,{1,0,0,1})
                .build();

        rp.draw(mesh, glm::mat4(1), mat1);

        static Inspector inspector;
        inspector.update();
        inspector.gui();

    }
private:
    SDLRenderer r;
    Camera camera;
    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<Material> mat1;
};

int main() {
    std::make_unique<CustomMeshLayoutExample>();
    return 0;
}

