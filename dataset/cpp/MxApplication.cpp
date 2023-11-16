#include <MxEngine.h>

namespace ProjectTemplate
{
    using namespace MxEngine;

    /*
    this is MxEngine template project. Just copy it and start developing!
    */
    class MxApplication : public Application
    {
    public:
        virtual void OnCreate() override
        {
            // create camera object
            auto cameraObject = MxObject::Create();
            cameraObject->Name = "Camera Object";

            // add CameraController component which handles camera image rendering
            auto controller = cameraObject->AddComponent<CameraController>();
            // add InpitControl which handles keyboard and mouse input events
            auto input = cameraObject->AddComponent<InputController>();
            // set camera to change ratio automatically depending on application window size
            controller->ListenWindowResizeEvent();
            // bind player movement to classic WASD mode and space/shift to fly, rotation is done with mouse
            input->BindMovement(KeyCode::W, KeyCode::A, KeyCode::S, KeyCode::D, KeyCode::SPACE, KeyCode::LEFT_SHIFT);
            input->BindRotation();
            // set controller to be main application viewport
            Rendering::SetViewport(controller);

            // create cube object
            auto cubeObject = MxObject::Create();
            cubeObject->Name = "Cube";
            // move it a bit away from camera
            cubeObject->LocalTransform.Translate(MakeVector3(-1.0f, -1.0f, 3.0f));
            // add mesh to a cube using Primitives class
            auto meshSource = cubeObject->AddComponent<MeshSource>(Primitives::CreateCube());
            // add default (white) material using MeshRenderer component
            auto meshRenderer = cubeObject->AddComponent<MeshRenderer>();

            // create global directional light
            auto lightObject = MxObject::Create();
            lightObject->Name = "Global Light";
            // add DirectionalLight component with custom light direction
            auto dirLight = lightObject->AddComponent<DirectionalLight>();
            dirLight->SetIntensity(0.5f);
            dirLight->Direction = MakeVector3(0.5f, 1.0f, 1.0f);
            // make directional light to be centered at current viewport position (is set by RenderManager::SetViewport)
            dirLight->IsFollowingViewport = true;
        }

        virtual void OnUpdate() override
        {
            
        }

        virtual void OnDestroy() override
        {

        }
    };
}

int main()
{
    MxEngine::LaunchFromSourceDirectory();
    ProjectTemplate::MxApplication app;
    app.Run();
    return 0;
}