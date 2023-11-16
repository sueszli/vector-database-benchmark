#include <MxEngine.h>
using namespace MxEngine;

void InitSound(MxObject& object)
{
    object.Name = "Sound Source";
    object.LocalTransform.Translate(MakeVector3(-10.0f, 2.0f, -10.0f));
    auto audio = object.AddComponent<AudioSource>();
    auto debug = object.AddComponent<DebugDraw>();
    debug->RenderSoundBounds = true;
    audio->Load(AssetManager::LoadAudio("Resources/sounds/never-gonna-give-you-up.mp3"_id));
    audio->Play();
}