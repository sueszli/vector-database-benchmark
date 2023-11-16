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

#include "InputController.h"
#include "Core/Application/Event.h"
#include "Core/Events/MouseEvent.h"
#include "Core/Events/UpdateEvent.h"
#include "Platform/Window/Input.h"
#include "Utilities/Logging/Logger.h"
#include "Core/Runtime/Reflection.h"

namespace MxEngine
{
    InputController::~InputController()
    {    
        MxString uuid = MxObject::GetComponentUUID(*this);
        Event::RemoveEventListener(uuid);
    }

    float InputController::GetMoveSpeed() const
    {
        return this->moveSpeed;
    }

    void InputController::SetMoveSpeed(float speed)
    {
        this->moveSpeed = speed;
    }

    float InputController::GetRotateSpeed() const
    {
        return this->rotateSpeed;
    }

    void InputController::SetRotateSpeed(float speed)
    {
        this->rotateSpeed = speed;
    }

    void InputController::BindMovement(KeyCode forward, KeyCode left, KeyCode back, KeyCode right)
    {
        this->BindMovement(forward, left, back, right, KeyCode::UNKNOWN, KeyCode::UNKNOWN);
    }

    void InputController::BindMovementCallback()
    {
        auto object = MxObject::GetHandleByComponent(*this);
        auto camera = object->GetComponent<CameraController>();
        auto input = object->GetComponent<InputController>();

        this->bindMovement = true;
        MXLOG_DEBUG("MxEngine::InputController", "bound object movement: " + object->Name);

        Event::AddEventListener<KeyEvent>(input.GetUUID(),
            [camera, input, object](auto& event) mutable
            {
                auto vecForward = MakeVector3(0.0f, 0.0f, 1.0f);
                auto vecRight = MakeVector3(-1.0f, 0.0f, 0.0f);
                auto vecUp = MakeVector3(0.0f, 1.0f, 0.0f);
                auto moveDirection = MakeVector3(0.0f);
                if (camera.IsValid())
                {
                    vecForward = camera->GetForwardVector();
                    vecRight = camera->GetRightVector();
                    vecUp = camera->GetUpVector();
                }
                else if (input->IsCameraStyleMovementEnabled())
                {
                    auto horizontalAngle = Radians(object->LocalTransform.GetRotation().y);
                    vecForward = MakeVector3(
                        sin(horizontalAngle),
                        0.0f,
                        cos(horizontalAngle)
                    );
                    vecRight = MakeVector3(
                        sin(horizontalAngle - HalfPi<float>()),
                        0.0f,
                        cos(horizontalAngle - HalfPi<float>())
                    );
                }
                else
                {
                    vecForward = object->LocalTransform.GetRotationQuaternion() * vecForward; //-V807
                    vecRight = object->LocalTransform.GetRotationQuaternion() * vecRight;
                    vecUp = object->LocalTransform.GetRotationQuaternion() * vecUp;
                }

                auto dt = Application::GetImpl()->GetUnscaledTimeDelta();
                if (Input::IsKeyHeld(input->GetForwardKeyBinding()))
                {
                    moveDirection += vecForward;
                }
                if (Input::IsKeyHeld(input->GetBackKeyBinding()))
                {
                    moveDirection -= vecForward;
                }
                if (Input::IsKeyHeld(input->GetRightKeyBinding()))
                {
                    moveDirection += vecRight;
                }
                if (Input::IsKeyHeld(input->GetLeftKeyBinding()))
                {
                    moveDirection -= vecRight;
                }
                if (Input::IsKeyHeld(input->GetUpKeyBinding()))
                {
                    moveDirection += vecUp;
                }
                if (Input::IsKeyHeld(input->GetDownKeyBinding()))
                {
                    moveDirection -= vecUp;
                }

                if (moveDirection != MakeVector3(0.0f))
                {
                    object->LocalTransform.Translate(Normalize(moveDirection) * input->GetMoveSpeed() * dt);
                    input->motion = Normalize(moveDirection);
                }
                else
                {
                    input->motion = MakeVector3(0.0f);
                }
            });
    }

    enum KeyBindingMapper : size_t
    {
        FORWARD = 0,
        LEFT = 1,
        BACK = 2,
        RIGHT = 3,
        UP = 4,
        DOWN = 5,
    };
    
    void InputController::BindMovement(KeyCode forward, KeyCode left, KeyCode back, KeyCode right, KeyCode up, KeyCode down)
    {
        this->keybindings[KeyBindingMapper::FORWARD] = forward;
        this->keybindings[KeyBindingMapper::BACK   ] = back;
        this->keybindings[KeyBindingMapper::LEFT   ] = left;
        this->keybindings[KeyBindingMapper::RIGHT  ] = right;
        this->keybindings[KeyBindingMapper::UP     ] = up;
        this->keybindings[KeyBindingMapper::DOWN   ] = down;
        if(!this->bindMovement)
            this->BindMovementCallback();
    }

    void InputController::BindMovementWASD()
    {
        this->BindMovement(KeyCode::W, KeyCode::A, KeyCode::S, KeyCode::D);
    }

    void InputController::BindMovementWASDSpaceShift()
    {
        this->BindMovement(KeyCode::W, KeyCode::A, KeyCode::S, KeyCode::D, KeyCode::SPACE, KeyCode::LEFT_SHIFT);
    }

    void InputController::BindRotation()
    {
        if (!this->bindHorizontalRotation && !this->bindVerticalRotation)
            this->BindRotationCallback();
        this->bindHorizontalRotation = true;
        this->bindVerticalRotation   = true;
    }
    
    void InputController::BindRotationCallback()
    {
        auto object = MxObject::GetHandleByComponent(*this);
        auto camera = object->GetComponent<CameraController>();
        auto input = object->GetComponent<InputController>();

        MXLOG_DEBUG("MxEngine::InputControl", "bound object rotation: " + object->Name);
        constexpr static auto invalidMousePos = MakeVector2(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());

        Event::AddEventListener<MouseMoveEvent>(input.GetUUID(), [object, camera, input, oldPos = invalidMousePos](auto& event) mutable
        {
            if (oldPos == invalidMousePos) 
                oldPos = event.position;

            auto dt = Application::GetImpl()->GetUnscaledTimeDelta();
            Vector2 diff(dt * (oldPos.x - event.position.x), dt * (oldPos.y - event.position.y));
            diff = MakeVector2(input->bindHorizontalRotation ? diff.x : 0.0f, input->bindVerticalRotation ? diff.y : 0.0f) * input->GetRotateSpeed();
            oldPos = event.position;

            if (camera.IsValid())
            {
                camera->Rotate(diff.x, diff.y);
            }
            else
            {
                object->LocalTransform.Rotate(MakeVector3(-diff.y, diff.x, 0.0f));
            }
        });
    }
    
    void InputController::BindHorizontalRotation()
    {
        if (!this->bindHorizontalRotation && !this->bindVerticalRotation)
            this->BindRotationCallback();
        this->bindHorizontalRotation = true;
    }
    
    void InputController::BindVerticalRotation()
    {
        if (!this->bindHorizontalRotation && !this->bindVerticalRotation)
            this->BindRotationCallback();
        this->bindVerticalRotation = true;
    }

    void InputController::UnbindAll()
    {
        this->BindMovement(KeyCode::UNKNOWN, KeyCode::UNKNOWN, KeyCode::UNKNOWN, KeyCode::UNKNOWN, KeyCode::UNKNOWN, KeyCode::UNKNOWN);
        this->bindVerticalRotation = false;
        this->bindHorizontalRotation = false;
    }

    const Vector3& InputController::GetMotionVector() const
    {
        return this->motion;
    }    

    KeyCode InputController::GetForwardKeyBinding() const
    {
        return this->keybindings[KeyBindingMapper::FORWARD];
    }

    KeyCode InputController::GetBackKeyBinding() const
    {
        return this->keybindings[KeyBindingMapper::BACK];
    }

    KeyCode InputController::GetLeftKeyBinding() const
    {
        return this->keybindings[KeyBindingMapper::LEFT];
    }

    KeyCode InputController::GetRightKeyBinding() const
    {
        return this->keybindings[KeyBindingMapper::RIGHT];
    }

    KeyCode InputController::GetUpKeyBinding() const
    {
        return this->keybindings[KeyBindingMapper::UP];
    }

    KeyCode InputController::GetDownKeyBinding() const
    {
        return this->keybindings[KeyBindingMapper::DOWN];
    }

    bool InputController::IsVerticalRotationBound() const
    {
        return this->bindVerticalRotation;
    }

    bool InputController::IsHorizontalRotationBound() const
    {
        return this->bindHorizontalRotation;
    }

    bool InputController::IsCameraStyleMovementEnabled() const
    {
        return MxObject::GetByComponent(*this).HasComponent<CameraController>() || this->cameraStyleMovement;
    }

    void InputController::ToggleCameraStyleMovement(bool value)
    {
        this->cameraStyleMovement = value;
    }

    void InputController::SetForwardKeyBinding(KeyCode key)
    {
        this->keybindings[KeyBindingMapper::FORWARD] = key;
        if (!this->bindMovement) this->BindMovementCallback();
    }

    void InputController::SetBackKeyBinding(KeyCode key)
    {
        this->keybindings[KeyBindingMapper::BACK] = key;
        if (!this->bindMovement) this->BindMovementCallback();
    }

    void InputController::SetLeftKeyBinding(KeyCode key)
    {
        this->keybindings[KeyBindingMapper::LEFT] = key;
        if (!this->bindMovement) this->BindMovementCallback();
    }

    void InputController::SetRightKeyBinding(KeyCode key)
    {
        this->keybindings[KeyBindingMapper::RIGHT] = key;
        if (!this->bindMovement) this->BindMovementCallback();
    }

    void InputController::SetUpKeyBinding(KeyCode key)
    {
        this->keybindings[KeyBindingMapper::UP] = key;
        if (!this->bindMovement) this->BindMovementCallback();
    }

    void InputController::SetDownKeyBinding(KeyCode key)
    {
        this->keybindings[KeyBindingMapper::DOWN] = key;
        if (!this->bindMovement) this->BindMovementCallback();
    }

    void InputController::ToggleHorizontalRotationBound(bool value)
    {
        if (value)
            this->BindHorizontalRotation();
        else
            this->bindHorizontalRotation = false;
    }

    void InputController::ToggleVerticalRotationBound(bool value)
    {
        if (value)
            this->BindVerticalRotation();
        else
            this->bindVerticalRotation = false;
    }

    MXENGINE_REFLECT_TYPE
    {
        rttr::registration::class_<InputController>("InputController")
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::CLONE_COPY | MetaInfo::CLONE_INSTANCE)
            )
            .constructor<>()
            .property("move speed", &InputController::GetMoveSpeed, &InputController::SetMoveSpeed)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE),
                rttr::metadata(EditorInfo::EDIT_PRECISION, 0.01f)
            )
            .property("rotate speed", &InputController::GetRotateSpeed, &InputController::SetRotateSpeed)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE),
                rttr::metadata(EditorInfo::EDIT_PRECISION, 0.01f)
            )
            .method("bind movement WASD", &InputController::BindMovementWASD)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .method("bind movement WASD + SPACE/SHIFT", &InputController::BindMovementWASDSpaceShift)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .method("unbind all", &InputController::UnbindAll)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property("is horizontal rotation bound", &InputController::IsHorizontalRotationBound, &InputController::ToggleHorizontalRotationBound)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("is vertical rotation bound", &InputController::IsVerticalRotationBound, &InputController::ToggleVerticalRotationBound)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("use camera style movement", &InputController::IsCameraStyleMovementEnabled, &InputController::ToggleCameraStyleMovement)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("forward key binding", &InputController::GetForwardKeyBinding, &InputController::SetForwardKeyBinding)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("left key binding", &InputController::GetLeftKeyBinding, &InputController::SetLeftKeyBinding)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("back key binding", &InputController::GetBackKeyBinding, &InputController::SetBackKeyBinding)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("right key binding", &InputController::GetRightKeyBinding, &InputController::SetRightKeyBinding)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("up key binding", &InputController::GetUpKeyBinding, &InputController::SetUpKeyBinding)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            )
            .property("down key binding", &InputController::GetDownKeyBinding, &InputController::SetDownKeyBinding)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::SERIALIZABLE | MetaInfo::EDITABLE)
            );
    }
}