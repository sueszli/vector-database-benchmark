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

#include "VRCameraController.h"
#include "Core/MxObject/MxObject.h"
#include "Core/Application/Rendering.h"
#include "Core/Application/Event.h"
#include "Core/Runtime/Reflection.h"

namespace MxEngine
{
    void VRCameraController::OnUpdate(float timeDelta)
    {
        auto camera = MxObject::GetByComponent(*this).GetComponent<CameraController>();

        if (!this->LeftEye.IsValid() || !this->RightEye.IsValid() || !camera.IsValid())
            return;

        this->UpdateEyes(this->LeftEye, this->RightEye);

        camera->ToggleRendering(false);

        auto leftTexture  = this->LeftEye->GetRenderTexture();
        auto rightTexture = this->RightEye->GetRenderTexture();
        auto resultTexture = camera->GetRenderTexture();

        size_t widthTotal = leftTexture->GetWidth() + rightTexture->GetWidth();
        size_t heightTotal = leftTexture->GetHeight() + rightTexture->GetHeight();

        if (resultTexture->GetWidth() != widthTotal || resultTexture->GetHeight() != heightTotal)
        {
            resultTexture->Load(nullptr, (int)widthTotal, (int)heightTotal, 3, false, resultTexture->GetFormat());
            resultTexture->SetInternalEngineTag(MXENGINE_MAKE_INTERNAL_TAG("vr camera out"));
        }

        this->Render(resultTexture, leftTexture, rightTexture);
    }

    void VRCameraController::UpdateEyes(CameraController::Handle& cameraL, CameraController::Handle& cameraR)
    {
        auto& object = MxObject::GetByComponent(*this);
        auto position = object.LocalTransform.GetPosition();
        auto camera = object.GetComponent<CameraController>();
        
        auto& LEyeTransform = MxObject::GetByComponent(*cameraL).LocalTransform;
        auto& REyeTransform = MxObject::GetByComponent(*cameraR).LocalTransform;

        auto LEyeDistance = -this->EyeDistance * camera->GetRightVector();
        auto REyeDistance = +this->EyeDistance * camera->GetRightVector();

        LEyeTransform.SetPosition(position + LEyeDistance);
        REyeTransform.SetPosition(position + REyeDistance);

        auto LEyeDirection = this->FocusDistance * camera->GetDirection() - LEyeDistance;
        auto REyeDirection = this->FocusDistance * camera->GetDirection() - REyeDistance;
        cameraL->SetDirection(Normalize(LEyeDirection));
        cameraR->SetDirection(Normalize(REyeDirection));
    }

    void VRCameraController::Render(TextureHandle& target, const TextureHandle& leftEye, const TextureHandle& rightEye)
    {
        leftEye->Bind(0);
        rightEye->Bind(1);
        this->shaderVR->Bind();
        this->shaderVR->SetUniform("leftEyeTex", 0);
        this->shaderVR->SetUniform("rightEyeTex", 1);
        Rendering::GetController().RenderToTexture(target, this->shaderVR);
    }

    void VRCameraController::Init()
    {
        this->shaderVR = Rendering::GetController().GetEnvironment().Shaders["VRCamera"_id];
    }

    MXENGINE_REFLECT_TYPE
    {
        rttr::registration::class_<VRCameraController>("VRCameraController")
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::CLONE_COPY | MetaInfo::CLONE_INSTANCE)
            )
            .constructor<>()
            .property("eye distance", &VRCameraController::EyeDistance)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE),
                rttr::metadata(EditorInfo::EDIT_PRECISION, 0.01f)
            )
            .property("focus distance", &VRCameraController::FocusDistance)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE),
                rttr::metadata(EditorInfo::EDIT_PRECISION, 0.01f)
            )
            .property("left eye", &VRCameraController::LeftEye)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property("right eye", &VRCameraController::RightEye)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property("eyes", &VRCameraController::LeftEye) // serialization of both eyes
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE),
                rttr::metadata(SerializeInfo::CUSTOM_SERIALIZE, SerializeExtra<VRCameraController>),
                rttr::metadata(SerializeInfo::CUSTOM_DESERIALIZE, DeserializeExtra<VRCameraController>)
            );
    }
}