// Copyright(c) 2019 - 2020, #Momo
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and /or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "ApplicationEditor.h"
#include "Utilities/ImGui/ImGuiUtils.h"
#include "Core/Config/GlobalConfig.h"
#include "Utilities/FileSystem/FileManager.h"
#include "Core/Application/Scene.h"

namespace MxEngine::GUI
{
    void DrawApplicationEditor(const char* name, bool* isOpen)
    {
        auto app = Application::GetImpl();

        ImGui::Begin(name, isOpen);
        ImGui::AlignTextToFramePadding();
        ImGui::Text("project build type: %s", EnumToString(GlobalConfig::GetBuildType()));
        ImGui::SameLine();
        ImGui::Checkbox("is paused", &app->IsPaused);

        ImGui::DragFloat("time scale", &app->TimeScale, 0.01f);
        ImGui::Text("current FPS: %d | total elapsed time: %f seconds", (int)Time::FPS(), Time::Current());
        ImGui::Text("time delta: %fms | frame interval: %fms", Time::Delta() * 1000.0f, Time::UnscaledDelta() * 1000.0f);

        if (ImGui::Button("save scene"))
        {
            MxString path = FileManager::SaveFileDialog("*.json", "MxEngine scene files");
            if (!path.empty())
            {
                Scene::Save(path);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("load scene"))
        {
            MxString path = FileManager::OpenFileDialog("*.json", "MxEngine scene files");
            if (!path.empty())
            {
                Scene::Load(path);
            }
        }

        ImGui::End();
    }
}