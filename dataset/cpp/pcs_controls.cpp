#include "pcs_controls.h"

#include "core.h"
#include "string_compare.hpp"

#include "v_file_service.h"

#include <input.hpp>

CREATE_SERVICE(PCS_CONTROLS)

using namespace storm;

PCS_CONTROLS::PCS_CONTROLS()
{
    m_bLockAll = false;
    m_bIsOffDebugKeys = false;

    fMouseSensivityX = 1.0f;
    fMouseSensivityY = 1.0f;

    nFrameCounter = 0;

    nSystemControlsNum = 0;

    nControlsNum = 0;

    nLastControlTime = 0;

    nMouseWheel = 0;
    memset(&ControlsTab[0], 0, sizeof(ControlsTab));

    auto pIni = fio->OpenIniFile(core.EngineIniFileName());
    if (pIni)
    {
        m_bIsOffDebugKeys = pIni->GetInt("controls", "ondebugkeys", 0) == 0;
    }

    input_ = Input::Create();
    inputHandlerID_ = input_->Subscribe([this](const InputEvent &evt) { HandleEvent(evt); });

    // RECT r;
    // GetWindowRect(core.GetAppHWND(),&r);
    // ClipCursor(&r);
}

PCS_CONTROLS::~PCS_CONTROLS()
{
    input_->Unsubscribe(inputHandlerID_);
    Release();
    // ClipCursor(0);
}

void PCS_CONTROLS::AppState(bool state)
{
    updateCursor_ = state;
    if (state)
    {
        // RECT r;
        // GetWindowRect(core.GetAppHWND(),&r);
        // ClipCursor(&r);
    }
    else
    {
        // ClipCursor(0);
    }
}

void PCS_CONTROLS::Release()
{
    int32_t n;
    for (n = 0; n < nControlsNum; n++)
    {
        delete pUserControls[n].name;
    }

    nControlsNum = 0;

    nSystemControlsNum = 0;
}

int32_t PCS_CONTROLS::GetSystemControlsNum()
{
    return nSystemControlsNum;
}

bool PCS_CONTROLS::GetSystemControlDesc(int32_t code, SYSTEM_CONTROL_DESC &_control_desc_struct)
{
    if (code >= CONTROL_ELEMENTS_NUM)
    {
        _control_desc_struct.pControlName = "invalid control";
        return false;
    }

    switch (code)
    {
    case CE_MOUSE_X_AXIS:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_AXIS;
        _control_desc_struct.pControlName = "Mouse X axis";
        break;
    case CE_MOUSE_Y_AXIS:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_AXIS;
        _control_desc_struct.pControlName = "Mouse Y axis";
        break;
    case VK_LBUTTON:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_BUTTON;
        _control_desc_struct.pControlName = "Mouse left button";
        break;
    case VK_RBUTTON:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_BUTTON;
        _control_desc_struct.pControlName = "Mouse right button";
        break;
    case VK_MBUTTON:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_BUTTON;
        _control_desc_struct.pControlName = "Mouse middle button";
        break;
    case VK_XBUTTON1:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_BUTTON;
        _control_desc_struct.pControlName = "Mouse special button 1";
        break;
    case VK_XBUTTON2:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_BUTTON;
        _control_desc_struct.pControlName = "Mouse special button 2";
        break;
    default:
        _control_desc_struct.ControlType = SYSTEM_CONTROL_DESC::CT_BUTTON;
        _control_desc_struct.pControlName = "Keyboard button";
        break;
    }
    return true;
}

int32_t PCS_CONTROLS::CreateControl(const char *control_name)
{
    int32_t n;
    if (control_name == nullptr)
        return INVALID_CONTROL_CODE;
    for (n = 0; n < nControlsNum; n++)
    {
        if (storm::iEquals(control_name, pUserControls[n].name))
            return n;
    }
    n = nControlsNum;
    nControlsNum++;
    pUserControls.resize(nControlsNum);
    const auto len = strlen(control_name) + 1;
    pUserControls[n].name = new char[len];
    memcpy(pUserControls[n].name, control_name, len);
    pUserControls[n].system_code = UNASSIGNED_CONTROL;
    pUserControls[n].flags = 0;
    pUserControls[n].state = CST_INACTIVE;
    pUserControls[n].nframe = 0;
    pUserControls[n].bLocked = false;
    pUserControls[n].control_type = UCT_Standart;

    return n;
}

int32_t PCS_CONTROLS::GetControlsNum()
{
    return nControlsNum;
}

bool PCS_CONTROLS::GetControlDesc(int32_t code, USER_CONTROL &_user_desc_struct)
{
    if (code < 0 || code >= nControlsNum)
        return false;

    _user_desc_struct.name = pUserControls[code].name;
    _user_desc_struct.system_code = pUserControls[code].system_code;

    return true;
}

int32_t PCS_CONTROLS::GetDevicesNum()
{
    return 1;
}

bool PCS_CONTROLS::GetDeviceDesc(int32_t code, DEVICE_DESC &_device_desc)
{
    _device_desc.name = "Keyboard and Mouse";
    return true;
}

int32_t PCS_CONTROLS::AddControlTreeNode(int32_t nParent, const char *pcBaseControl, const char *pcOutControl, float fTimeOut)
{
    const auto ntree = m_ControlTree.AddControlChild(nParent, pcBaseControl, pcOutControl, fTimeOut);
    if (ntree >= 0 && pcOutControl)
    {
        const auto nc = CreateControl((char *)pcOutControl);
        if (nc >= 0)
        {
            pUserControls[nc].control_type = UCT_ControlTree;
            pUserControls[nc].state = CST_INACTIVE;
        }
    }
    return ntree;
}

void PCS_CONTROLS::MapControl(int32_t control_code, int32_t system_control_code)
{
    if (control_code < 0 || control_code >= nControlsNum)
    {
        pUserControls[control_code].system_code = UNASSIGNED_CONTROL;
        return;
    }
    pUserControls[control_code].system_code = system_control_code;
}

bool PCS_CONTROLS::GetControlState(const char *control_name, CONTROL_STATE &_state_struct)
{
    /*
      int32_t n;
      _state_struct.state = CST_INACTIVE;
      _state_struct.lValue = 0;
      _state_struct.fValue = 0.0f;

      if(control_name == 0) return false;
      for(n=0;n<nControlsNum;n++)
      {
        if(storm::iEquals(control_name,pUserControls[n].name)) return GetControlState(n,_state_struct);
      }
      return false;
      //*/

    int32_t n;
    const auto bControlFound = false;

    _state_struct.state = CST_INACTIVE;
    _state_struct.lValue = 0;
    _state_struct.fValue = 0.0f;

    if (m_bLockAll)
        return true;
    if (control_name == nullptr)
        return false;
    for (n = 0; n < nControlsNum; n++)
    {
        if (storm::iEquals(control_name, pUserControls[n].name))
        {
            if (pUserControls[n].bLocked)
            {
                _state_struct.state = CST_INACTIVE;
                _state_struct.lValue = 0;
                _state_struct.fValue = 0.0f;
                return true;
            }

            /*if(GetControlState(n,local_state_struct))
            {
              bControlFound = true;

              if(_state_struct.state == CST_INACTIVE)    // if control inactive just copy what we have on new control
              {
                _state_struct = local_state_struct;
              }
              else
              {
                if(_state_struct.state == CST_ACTIVE)    // if control already active, check for activated state
                {
                  if(local_state_struct.state == CST_ACTIVATED) _state_struct = local_state_struct;
                }
                else return true;
              }


            }  */
            return GetControlState(n, _state_struct);
        }
    }
    return bControlFound;
}

bool PCS_CONTROLS::GetControlState(int32_t control_code, CONTROL_STATE &_state_struct)
{
    uint32_t system_code;
    int32_t lRes;

    if (control_code >= nControlsNum)
    {
        _state_struct.state = CST_INACTIVE;
        _state_struct.lValue = 0;
        _state_struct.fValue = 0.0f;
        return false;
    }

    if (pUserControls[control_code].bLocked)
    {
        _state_struct.state = CST_INACTIVE;
        _state_struct.lValue = 0;
        _state_struct.fValue = 0.0f;
        return true;
    }

    // processing of controls which is a tree of controls
    if (pUserControls[control_code].control_type == UCT_ControlTree)
    {
        if (pUserControls[control_code].nframe != nFrameCounter)
        {
            pUserControls[control_code].nframe = nFrameCounter;
            pUserControls[control_code].state = m_ControlTree.CheckControlActivated(pUserControls[control_code].name);
        }
        _state_struct.state = pUserControls[control_code].state;
        if (_state_struct.state == CST_ACTIVATED || _state_struct.state == CST_ACTIVE)
        {
            _state_struct.fValue = 1.f;
            _state_struct.lValue = 1;
        }
        else
        {
            _state_struct.fValue = 0.f;
            _state_struct.lValue = 0;
        }
        return true;
    }

    system_code = pUserControls[control_code].system_code;

    if (system_code >= CONTROL_ELEMENTS_NUM)
    {
        _state_struct.state = CST_INACTIVE;
        _state_struct.lValue = 0;
        _state_struct.fValue = 0.0f;
        return false;
    }

    if (ControlsTab[system_code].update_frame == nFrameCounter)
    {
        _state_struct = ControlsTab[system_code].state;
    }
    else
    {
        ControlsTab[system_code].update_frame = nFrameCounter;

        if (system_code == CE_MOUSE_X_AXIS || system_code == CE_MOUSE_Y_AXIS)
        {
            _state_struct = ControlsTab[system_code].state;
        }
        else if (system_code == CE_MOUSE_WHEEL_UP || system_code == CE_MOUSE_WHEEL_DOWN)
        {
            _state_struct = ControlsTab[system_code].state;
        }
        else
        {
            if (IsKeyPressed(system_code))
            {
                ControlsTab[system_code].state.lValue = 1;
                ControlsTab[system_code].state.fValue = static_cast<float>(ControlsTab[system_code].state.lValue);

                if (ControlsTab[system_code].state.state == CST_INACTIVE ||
                    ControlsTab[system_code].state.state == CST_INACTIVATED)
                {
                    ControlsTab[system_code].state.state = CST_ACTIVATED;
                }
                else
                    ControlsTab[system_code].state.state = CST_ACTIVE;
            }
            else
            {
                ControlsTab[system_code].state.lValue = 0;
                ControlsTab[system_code].state.fValue = 0;

                if (ControlsTab[system_code].state.state == CST_ACTIVE ||
                    ControlsTab[system_code].state.state == CST_ACTIVATED)
                {
                    ControlsTab[system_code].state.state = CST_INACTIVATED;
                }
                else
                    ControlsTab[system_code].state.state = CST_INACTIVE;
            }
        }
    }

    _state_struct.state = ControlsTab[system_code].state.state;

    if (pUserControls[control_code].flags & INVERSE_CONTROL)
    {
        if (pUserControls[control_code].flags & USE_AXIS_AS_BUTTON)
        {
            _state_struct.lValue = 1 - _state_struct.lValue;
            _state_struct.fValue = 1.f - _state_struct.fValue;
        }
        else
        {
            _state_struct.lValue = -_state_struct.lValue;
            _state_struct.fValue = -_state_struct.fValue;
        }
        switch (_state_struct.state)
        {
        case CST_ACTIVE:
            _state_struct.state = CST_INACTIVE;
            break;
        case CST_INACTIVE:
            _state_struct.state = CST_ACTIVE;
            break;
        case CST_ACTIVATED:
            _state_struct.state = CST_INACTIVATED;
            break;
        case CST_INACTIVATED:
            _state_struct.state = CST_ACTIVATED;
            break;
        }
    }

    if ((pUserControls[control_code].flags & USE_AXIS_AS_BUTTON) && pUserControls[control_code].state != FORCE_DWORD)
    {
        if (_state_struct.lValue > 0)
        {
            if (pUserControls[control_code].state == CST_INACTIVE ||
                pUserControls[control_code].state == CST_INACTIVATED)
                _state_struct.state = CST_ACTIVATED;
            else
                _state_struct.state = CST_ACTIVE;

            _state_struct.lValue = 1;
            _state_struct.fValue = 1.0f;
        }
        else
        {
            if (pUserControls[control_code].state == CST_ACTIVE || pUserControls[control_code].state == CST_INACTIVE)
                _state_struct.state = CST_INACTIVATED;
            else
                _state_struct.state = CST_INACTIVE;
            _state_struct.lValue = 0;
            _state_struct.fValue = 0.0f;
        }
    }
    if (_state_struct.state != CST_INACTIVE)
        if (!(system_code == CE_MOUSE_X_AXIS || system_code == CE_MOUSE_Y_AXIS))
            nLastControlTime = 0;
    return true;
}

void PCS_CONTROLS::Update(uint32_t DeltaTime)
{
#ifdef _WIN32
    static int nMouseXPrev, nMouseYPrev;
    if (updateCursor_) {
        POINT point;
        GetCursorPos(&point);

        nMouseDx = point.x - nMouseXPrev;
        nMouseDy = point.y - nMouseYPrev;

        RECT r;
        GetWindowRect(static_cast<HWND>(core.GetWindow()->OSHandle()), &r);
        nMouseXPrev = r.left + (r.right - r.left) / 2;
        nMouseYPrev = r.top + (r.bottom - r.top) / 2;
        SetCursorPos(nMouseXPrev, nMouseYPrev);
    }
    else {
        nMouseDx = 0;
        nMouseDy = 0;
    }
#endif

    m_ControlTree.Process();
    m_KeyBuffer.Reset();

    nFrameCounter++;
    uint32_t system_code = CE_MOUSE_X_AXIS;
    ControlsTab[system_code].state.lValue = nMouseDx;
    ControlsTab[system_code].state.fValue =
        static_cast<float>(ControlsTab[system_code].state.lValue) * fMouseSensivityX;
    ControlsTab[system_code].update_frame = nFrameCounter;

    system_code = CE_MOUSE_Y_AXIS;
    ControlsTab[system_code].state.lValue = nMouseDy;
    ControlsTab[system_code].state.fValue =
        static_cast<float>(ControlsTab[system_code].state.lValue) * fMouseSensivityY;
    ControlsTab[system_code].update_frame = nFrameCounter;

    nMouseDx = 0;
    nMouseDy = 0;

    system_code = CE_MOUSE_WHEEL_UP;
    if (nMouseWheel > 0)
    {
        ControlsTab[system_code].state.lValue = nMouseWheel / 10;
        ControlsTab[system_code].state.fValue = nMouseWheel * .1f;
        if (ControlsTab[system_code].state.state == CST_INACTIVE ||
            ControlsTab[system_code].state.state == CST_INACTIVATED)
            ControlsTab[system_code].state.state = CST_ACTIVATED;
        else
            ControlsTab[system_code].state.state = CST_ACTIVE;
    }
    else
    {
        ControlsTab[system_code].state.lValue = 0;
        ControlsTab[system_code].state.fValue = 0.f;
        if (ControlsTab[system_code].state.state == CST_ACTIVE || ControlsTab[system_code].state.state == CST_ACTIVATED)
            ControlsTab[system_code].state.state = CST_INACTIVATED;
        else
            ControlsTab[system_code].state.state = CST_INACTIVE;
    }
    ControlsTab[system_code].update_frame = nFrameCounter;

    system_code = CE_MOUSE_WHEEL_DOWN;
    if (nMouseWheel < 0)
    {
        ControlsTab[system_code].state.lValue = -nMouseWheel / 10;
        ControlsTab[system_code].state.fValue = -nMouseWheel * .1f;
        if (ControlsTab[system_code].state.state == CST_INACTIVE ||
            ControlsTab[system_code].state.state == CST_INACTIVATED)
            ControlsTab[system_code].state.state = CST_ACTIVATED;
        else
            ControlsTab[system_code].state.state = CST_ACTIVE;
    }
    else
    {
        ControlsTab[system_code].state.lValue = 0;
        ControlsTab[system_code].state.fValue = 0.f;
        if (ControlsTab[system_code].state.state == CST_ACTIVE || ControlsTab[system_code].state.state == CST_ACTIVATED)
            ControlsTab[system_code].state.state = CST_INACTIVATED;
        else
            ControlsTab[system_code].state.state = CST_INACTIVE;
    }
    ControlsTab[system_code].update_frame = nFrameCounter;

    nMouseWheel = 0;

    nLastControlTime += DeltaTime;
}

bool PCS_CONTROLS::SetControlFlags(int32_t code, uint32_t _flags)
{
    if (code < 0 || code >= nControlsNum)
        return false;
    pUserControls[code].flags = _flags;
    return true;
}

bool PCS_CONTROLS::SetControlState(const char *control_name, CONTROL_STATE &_state_struct)
{
    int32_t n;
    if (control_name == nullptr)
        return false;
    for (n = 0; n < nControlsNum; n++)
    {
        if (storm::iEquals(control_name, pUserControls[n].name))
            return SetControlState(n, _state_struct);
    }
    return false;
}

bool PCS_CONTROLS::SetControlState(int32_t control_code, CONTROL_STATE &_state_struct)
{
    if (control_code < 0 || control_code >= nControlsNum)
        return false;
    pUserControls[control_code].state = _state_struct.state;
    return true;
}

int32_t PCS_CONTROLS::LastControlTime()
{
    return nLastControlTime;
}

void PCS_CONTROLS::SetControlTreshold(int32_t control_code, float thval)
{
    // ~!~
}

void PCS_CONTROLS::LockControl(const char *control_name, bool mode)
{
    int32_t n;
    if (control_name == nullptr || control_name[0] == 0)
    {
        m_bLockAll = mode;
        return;
    }
    for (n = 0; n < nControlsNum; n++)
    {
        if (storm::iEquals(control_name, pUserControls[n].name))
        {
            pUserControls[n].bLocked = mode;
            pUserControls[n].state = FORCE_DWORD;
            return;
        }
    }
}

void PCS_CONTROLS::SetMouseSensivityX(float _s)
{
    fMouseSensivityX = _s;
}

void PCS_CONTROLS::SetMouseSensivityY(float _s)
{
    fMouseSensivityY = _s;
}

int32_t PCS_CONTROLS::GetKeyBufferLength()
{
    return m_KeyBuffer.GetBufferLength();
}

const KeyDescr *PCS_CONTROLS::GetKeyBuffer()
{
    return m_KeyBuffer.c_str();
}

void PCS_CONTROLS::ClearKeyBuffer()
{
    m_KeyBuffer.Reset();
}

short PCS_CONTROLS::GetAsyncKeyState(int vk)
{
    return IsKeyPressed(vk) ? -1 : 0;
}

short PCS_CONTROLS::GetKeyState(int vk)
{
    return GetAsyncKeyState(vk);
}

short PCS_CONTROLS::GetDebugAsyncKeyState(int vk)
{
    // -1 because WinAPI sets msb when key pressed, so old code expects negative value
    if (m_bIsOffDebugKeys)
    {
        return 0;
    }

    return IsKeyPressed(vk) ? -1 : 0;
}

short PCS_CONTROLS::GetDebugKeyState(int vk)
{
    return GetDebugAsyncKeyState(vk);
}

bool PCS_CONTROLS::IsKeyPressed(int vk)
{
    auto pressed = false;
    if (vk == VK_LBUTTON)
        pressed = input_->MouseKeyState(MouseKey::Left);
    else if (vk == VK_RBUTTON)
        pressed = input_->MouseKeyState(MouseKey::Right);
    else if (vk == VK_MBUTTON)
        pressed = input_->MouseKeyState(MouseKey::Middle);
    else if (vk == VK_XBUTTON1)
        pressed = input_->MouseKeyState(MouseKey::Special1);
    else if (vk == VK_XBUTTON2)
        pressed = input_->MouseKeyState(MouseKey::Special2);
    else if (vk == VK_TAB)
    {
        if (!input_->KeyboardKeyState(VK_MENU))
            pressed = input_->KeyboardKeyState(vk);
    }
    else if (vk == VK_SHIFT)
    {
        pressed = input_->KeyboardKeyState(VK_LSHIFT) || input_->KeyboardKeyState(VK_RSHIFT);
    }
    else if (vk == VK_RETURN)
    {
        pressed = input_->KeyboardSDLKeyState(SDL_SCANCODE_RETURN) ||
                  input_->KeyboardSDLKeyState(SDL_SCANCODE_KP_ENTER);
    }
    else if (vk == VK_UP)
    {
        pressed = input_->KeyboardSDLKeyState(SDL_SCANCODE_UP) ||
                  (!input_->KeyboardModState(VK_NUMLOCK) && input_->KeyboardSDLKeyState(SDL_SCANCODE_KP_8));
    }
    else if (vk == VK_DOWN)
    {
        pressed = input_->KeyboardSDLKeyState(SDL_SCANCODE_DOWN) ||
                  (!input_->KeyboardModState(VK_NUMLOCK) && input_->KeyboardSDLKeyState(SDL_SCANCODE_KP_2));
    }
    else if (vk == VK_LEFT)
    {
        pressed = input_->KeyboardSDLKeyState(SDL_SCANCODE_LEFT) ||
                  (!input_->KeyboardModState(VK_NUMLOCK) && input_->KeyboardSDLKeyState(SDL_SCANCODE_KP_4));
    }
    else if (vk == VK_RIGHT)
    {
        pressed = input_->KeyboardSDLKeyState(SDL_SCANCODE_RIGHT) ||
                  (!input_->KeyboardModState(VK_NUMLOCK) && input_->KeyboardSDLKeyState(SDL_SCANCODE_KP_6));
    }
    else if (vk == VK_NUMLOCK)
        pressed = input_->KeyboardModState(vk);
    else
        pressed = input_->KeyboardKeyState(vk);

    return pressed;
}

void PCS_CONTROLS::HandleEvent(const InputEvent &evt)
{
    if (evt.type == InputEvent::KeyboardKeyDown)
    {
        char text[5];
        int size = utf8::CodepointToUtf8(text, (uint32_t)std::get<KeyboardKey>(evt.data));
        m_KeyBuffer.AddKey(text, size, true);
    }
    else if (evt.type == InputEvent::KeyboardText)
    {
        const auto &text = std::get<std::string>(evt.data);
        m_KeyBuffer.AddKey((char *)text.c_str(), text.length(), false);
    }
    else if (evt.type == InputEvent::MouseMove)
    {
        const auto &dxdy = std::get<MousePos>(evt.data);
        nMouseDx += dxdy.x;
        nMouseDy += dxdy.y;
    }
    else if (evt.type == InputEvent::MouseWheel)
    {
        const auto &dxdy = std::get<MousePos>(evt.data);
        nMouseWheel += dxdy.y * input_->GetWheelFactor();
        core.Event("evMouseWeel", "l", static_cast<short>(dxdy.y));
    }
}
