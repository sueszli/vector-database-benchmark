#include <iostream>
#include <vector>
#include <fstream>

#include "sre/Texture.hpp"
#include "sre/Renderer.hpp"
#include "sre/Material.hpp"
#include "sre/SDLRenderer.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <sre/Inspector.hpp>


using namespace sre;

class RenderToTextureExample {
public:
    RenderToTextureExample(){
        r.init();

        camera.lookAt({0,0,3},{0,0,0},{0,1,0});
        camera.setPerspectiveProjection(60,0.1,100);

        texture = Texture::create().withRGBData(nullptr, 1024,1024).build();

        framebuffer = Framebuffer::create().withColorTexture(texture).build();

        materialOffscreen = Shader::getStandardBlinnPhong()->createMaterial();
        materialOffscreen->setSpecularity({1,1,1,120});
        material = Shader::getStandardBlinnPhong()->createMaterial();
        material->setTexture(texture);

        mesh = Mesh::create().withCube().build();
        worldLights.addLight(Light::create().withPointLight({0,0,3}).withColor({1,1,1}).withRange(20).build());

        r.frameRender = [&](){
            render();
        };
        r.mouseEvent = [&](SDL_Event& event){
            if (event.button.button==SDL_BUTTON_RIGHT){
                showInspector = true;
            }
        };
        r.startEventLoop();
    }

    void render(){

        auto renderToTexturePass = RenderPass::create()                 // Create a renderpass which writes to the texture using a framebuffer
                .withCamera(camera)
                .withWorldLights(&worldLights)
                .withFramebuffer(framebuffer)
                .withClearColor(true, {0, 1, 1, 0})
                .withGUI(false)
                .build();

        renderToTexturePass.draw(mesh, glm::eulerAngleY(glm::radians((float)i)), materialOffscreen);

        auto renderPass = RenderPass::create()                          // Create a renderpass which writes to the screen.
                .withCamera(camera)
                .withWorldLights(&worldLights)
                .withClearColor(true, {1, 0, 0, 1})
                .withGUI(true)
                .build();

        renderPass.draw(mesh, glm::eulerAngleY(glm::radians((float)i)), material);
                                                                        // The offscreen texture is used in material
        static Inspector prof;
        prof.update();
        if (showInspector){
            prof.gui(true);
        }

        i++;
    }
private:
    SDLRenderer r;
    Camera camera;
    WorldLights worldLights;
    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<Material> materialOffscreen;
    std::shared_ptr<Material> material;
    std::shared_ptr<Texture> texture;
    std::shared_ptr<Framebuffer> framebuffer;
    int i=0;
    bool showInspector = false;
};

int main() {
    std::make_unique<RenderToTextureExample>();
    return 0;
}
