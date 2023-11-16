#include "Sandbox.h"
#include "Gameplay/SandboxScene.h"

void SandboxApp::OnCreate()
{
	Event::AddEventListener<FpsUpdateEvent>("CountFPS",
		[](auto& e)
		{
			WindowManager::SetTitle(MxFormat("Sandbox App {0} FPS", e.FPS));
		});

	scene->OnCreate();
}

void SandboxApp::OnUpdate()
{
	scene->OnUpdate();
}

void SandboxApp::OnRender()
{

}

void SandboxApp::OnDestroy()
{
	
}

SandboxApp::SandboxApp()
{
	scene = new SandboxScene();
}

int main()
{
	MxEngine::LaunchFromSourceDirectory();
	SandboxApp app;
	app.Run();
	return 0;
}