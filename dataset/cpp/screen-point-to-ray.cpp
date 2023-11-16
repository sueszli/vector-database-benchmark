#include <iostream>

#include "sre/Texture.hpp"
#include "sre/Renderer.hpp"
#include "sre/Material.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <sre/SDLRenderer.hpp>
#include <sre/impl/GL.hpp>
#include <glm/gtc/random.hpp>

using namespace sre;

class ScreenPointToRayExample{
public:
    ScreenPointToRayExample(){
        r.init();

        camera.lookAt(eye,at,{0,1,0});
        camera.setPerspectiveProjection(fov,near,far);

        mesh = Mesh::create()
                .withSphere()
                .build();

        planeMesh = Mesh::create()
                .withCube(10)
                .build();

        worldLights.addLight(Light::create()
                                     .withDirectionalLight(glm::normalize(glm::vec3(1,1,1)))
                                     .build());

        // Add fake shadows
        worldLights.addLight(Light::create()
                                     .withPointLight(p1-glm::vec3(0,0.8,0))
                                     .withColor({-3.0f,-3.0f,-3.0f})
                                     .withRange(4)
                                     .build());
        worldLights.addLight(Light::create()
                                     .withPointLight(p2-glm::vec3(0,0.8,0))
                                     .withColor({-3.0f,-3.0f,-3.0f})
                                     .withRange(4)
                                     .build());


        mat1 = Shader::getStandardBlinnPhong()->createMaterial();
        mat1->setColor({1,1,1,1});
        mat1->setSpecularity(Color(0,0,0,0));

        mat2 = Shader::getStandardBlinnPhong()->createMaterial();
        mat2->setColor({1,0,0,1});
        mat2->setSpecularity(Color(0,0,0,0));

        matPlane = Shader::getStandardBlinnPhong()->createMaterial();
        matPlane->setColor({1,1,1,1});
        matPlane->setSpecularity(Color(0,0,0,0));

        r.mouseEvent = [&](SDL_Event event){
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button==SDL_BUTTON_RIGHT){
                glm::vec2 pos = {event.button.x,event.button.y};
                pos.y = Renderer::instance->getWindowSize().y - pos.y; // flip y axis
                auto res = camera.screenPointToRay(pos);
                raycastOrigin = res[0];
                raycastDirection =res[1];
                points = {{raycastOrigin, raycastOrigin+raycastDirection}};

                float dist1 = rayToSphere(res, p1);
                if (dist1<1){
                    updateMaterial(mat1);
                }
                float dist2 = rayToSphere(res, p2);
                if (dist2<1){
                    updateMaterial(mat2);
                }
            }
        };

        r.frameUpdate = [&](float deltaTime){
            update(deltaTime);
        };
        r.frameRender = [&](){
            render();
        };
        r.startEventLoop();
    }

    void updateMaterial(std::shared_ptr<Material>& mat){
        Color color (glm::linearRand(0.0f, 1.0f),glm::linearRand(0.0f, 1.0f),glm::linearRand(0.0f, 1.0f));
        mat->setColor(color);
    }

    void cameraGUI(){
        ImGui::Checkbox("Perspective projection", &perspective);
        if (perspective){
            ImGui::DragFloat("FOV", &fov,1,1,179);
        } else {
            ImGui::DragFloat("OrthoSize", &orthoSize,0.1,0.1,10);
        }
        ImGui::DragFloat("Near", &near,0.1,-10,10);
        ImGui::DragFloat("Far", &far,0.1,0.1,100);
        if (perspective){
            camera.setPerspectiveProjection(fov,near,far);
        } else {
            camera.setOrthographicProjection(orthoSize,near,far);
        }
        ImGui::DragFloat3("eye", &eye.x,0.1,-10,10);
        ImGui::DragFloat3("at", &at.x,0.1,-10,10);
        camera.lookAt(eye,at,{0,1,0});

        ImGui::DragFloat2("viewportOffset",&viewportOffset.x,0.05,0,1);
        ImGui::DragFloat2("viewportSize",  &viewportSize.x,0.05,0,1);
        camera.setViewport(viewportOffset,viewportSize);

    }

    void update(float deltaTime){
        time += deltaTime;
    }

    float rayToSphere(std::array<glm::vec3, 2> ray, glm::vec3 sphereCenter){
        float d = dot(sphereCenter-ray[0], ray[1]);
        if (d < 0){
            d = 0;
        }
        glm::vec3 closestPoint =  d* ray[1] + ray[0];
        return glm::distance(closestPoint, sphereCenter);
    }

    void render(){
        auto rp = RenderPass::create()
                .withCamera(camera)
                .withWorldLights(&worldLights)
                .withClearColor(true,{1,0,0,1})
                .build();

        rp.draw(mesh, pos1, mat1);

        checkGLError();
        rp.draw(mesh, pos2, mat2);

        checkGLError();

        ImGui::LabelText("Rightclick to shoot ray", "");
        rp.draw(planeMesh, glm::translate(glm::vec3{0,-1.0f,0})*glm::scale(glm::vec3{1,.01f,1}), matPlane);

        ImGui::LabelText("raycastOrigin", "%.1f,%.1f,%.1f", raycastOrigin.x,raycastOrigin.y,raycastOrigin.z);
        ImGui::LabelText("raycastDirection", "%.1f,%.1f,%.1f", raycastDirection.x,raycastDirection.y,raycastDirection.z);

        rp.drawLines(points);

        cameraGUI();
    }
private:
    float time;
    SDLRenderer r;
    Camera camera;
    std::shared_ptr<Mesh> mesh;
    std::shared_ptr<Mesh> planeMesh;
    WorldLights worldLights;

    // camera properties
    bool perspective=true;
    float fov = 60;
    float near = 0.1;
    float far = 100;
    float orthoSize = 2;

    glm::vec2 viewportOffset = glm::vec2{0};
    glm::vec2 viewportSize = glm::vec2{1};

    glm::vec3 eye = {0,0,3};
    glm::vec3 at = {0,0,0};

    std::shared_ptr<Material> mat1;
    std::shared_ptr<Material> mat2;
    std::shared_ptr<Material> matPlane;
    glm::vec3 p1 =  {-1,0,0};
    glm::vec3 p2 =  {1,0,0};
    glm::mat4 pos1 = glm::translate(glm::mat4(1),p1);
    glm::mat4 pos2 = glm::translate(glm::mat4(1),p2);


    glm::vec3 raycastOrigin{0};
    glm::vec3 raycastDirection{0};
    std::vector<glm::vec3> points{{raycastOrigin, raycastOrigin+raycastDirection}};
};

int main() {
    std::make_unique<ScreenPointToRayExample>();
    return 0;
}

