#include "SystemInputClass.h"

SystemInputClass::SystemInputClass()
{
	m_directInput = 0;
	m_keyboard = 0;
	m_mouse = 0;

	// 키 입력 비활성화 처리
	for (int i = 0; i < ARRAYSIZE(m_keys); i++)
	{
		m_keys[i] = false;
	}
}

HRESULT SystemInputClass::Init(HINSTANCE hinstance, HWND hwnd, const int screenWidth, const int screenHeight)
{
	HRESULT hr = S_OK;

	m_screenWidth = screenWidth;
	m_screenHeight = screenHeight;

	m_mouseX = 0;
	m_mouseY = 0;

	// Direct 입력 인터페이스 초기화, 객체 생성
	hr = DirectInput8Create(hinstance, DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)&m_directInput, NULL);
	if (FAILED(hr))
		return hr;

	// 위에서 얻은 입력 객체를 통해 키보드 객체 생성
	hr = m_directInput->CreateDevice(GUID_SysKeyboard, &m_keyboard, NULL);
	if (FAILED(hr))
		return hr;

	// 데이터 포멧 설정, 키보드를 사용한다고 알려주는 그런거 인듯
	hr = m_keyboard->SetDataFormat(&c_dfDIKeyboard);
	if (FAILED(hr))
		return hr;

	// 키보드 협력 레벨 설정, 다른 프로그램에서 접근을 제어
	hr = m_keyboard->SetCooperativeLevel(hwnd, DISCL_FOREGROUND | DISCL_EXCLUSIVE);
	if (FAILED(hr))
		return hr;

	// 키보드 접근 권한 얻기
	m_keyboard->Acquire();

	hr = m_directInput->CreateDevice(GUID_SysMouse, &m_mouse, NULL);
	if (FAILED(hr))
		return hr;

	hr = m_mouse->SetDataFormat(&c_dfDIMouse);
	if (FAILED(hr))
		return hr;

	// DISCL_NONEXCLUSIVE
	hr = m_mouse->SetCooperativeLevel(hwnd, DISCL_FOREGROUND | DISCL_EXCLUSIVE);
	if (FAILED(hr))
		return hr;

	m_mouse->Acquire();

	// 키 미리 설정
	moveKey[0] = DIK_W;
	moveKey[1] = DIK_S;
	moveKey[2] = DIK_A;
	moveKey[3] = DIK_D;

	return hr;
}

bool SystemInputClass::Frame()
{
	bool result;

	result = ReadKeyBoard();
	if (!result)
		return false;

	result = ReadMouse();
	if (!result)
		return false;

	ProcessInput();

	return true;
}

void SystemInputClass::KeyDown(unsigned int input)
{
	m_keys[input] = true;
}

void SystemInputClass::KeyUp(unsigned int input)
{
	m_keys[input] = false;
}

bool SystemInputClass::ReadKeyBoard()
{
	HRESULT result;

	result = m_keyboard->GetDeviceState(sizeof(m_keyboardState), (LPVOID)&m_keyboardState);
	if (FAILED(result))
	{
		// 포커스를 잃었거나 acquire이 되지 않는 상황이면
		if (result == DIERR_INPUTLOST || result == DIERR_NOTACQUIRED)
		{
			m_keyboard->Acquire();
		}
		else
			return false;
	}

	return true;
}

bool SystemInputClass::ReadMouse()
{
	HRESULT result;

	result = m_mouse->GetDeviceState(sizeof(DIMOUSESTATE), &m_mouseState);
	if (FAILED(result))
	{
		// 포커스를 잃었거나 acquire이 되지 않는 상황이면
		if (result == DIERR_INPUTLOST || result == DIERR_NOTACQUIRED)
		{
			m_mouse->Acquire();
		}
		else
			return false;	// 다른 에러 상황일 경우 강제 종료
	}

	return true;
}

void SystemInputClass::ProcessInput()
{
	// 마지막 프레임 이후 입력장치에서 일어난 변화 처리
	// 마우스 위치 변경, 마우스 커서 위치 유지
	m_mouseX = m_mouseState.lX;
	m_mouseY = m_mouseState.lY;
}

bool SystemInputClass::IsEscapePressed()
{
	if (m_keyboardState[DIK_ESCAPE] & 0x80)
		return true;

	return false;
}

void SystemInputClass::GetMouseLocation(int& mouseX, int& mouseY)
{
	mouseX = m_mouseX;
	mouseY = m_mouseY;
}

void SystemInputClass::GetFunctionKeyPressed(unsigned int& key, bool& isKeyUp)
{
	bool isKeyDown = false;

	for (int i = DIK_F1; i <= DIK_F8; i++)
	{
		if (m_keyboardState[i] & 0x80)
		{
			m_keys[i] = true;
			key = i;
			isKeyUp = false;
			isKeyDown = true;
			return;
		}
	}

	if (!isKeyDown)
	{
		key = 0;
		isKeyUp = false;
	}

	for (int i = DIK_F1; i <= DIK_F8; i++)
	{
		if (m_keys[i])
		{
			m_keys[i] = false;
			key = i;
			isKeyUp = true;
			return;
		}
	}
}

void SystemInputClass::GetMoveKeyPressed(unsigned int& key)
{
	bool isKeyDown = false;

	// wsad 순서
	for (int i = 0; i < ARRAYSIZE(moveKey); i++)
	{
		if (m_keyboardState[moveKey[i]] & 0x80)
		{
			isKeyDown = true;
			key = moveKey[i];
			return;
		}
	}
	
	if (!isKeyDown)
	{
		key = 0;
	}
}

DIMOUSESTATE* SystemInputClass::GetMouseState()
{
	return &m_mouseState;
}

void SystemInputClass::Shutdown()
{
	if (m_mouse)
	{
		m_mouse->Unacquire();
		m_mouse->Release();
		m_mouse = 0;
	}

	if (m_keyboard)
	{
		m_keyboard->Unacquire();
		m_keyboard->Release();
		m_keyboard = 0;
	}

	if (m_directInput)
	{
		m_directInput->Release();
		m_directInput = 0;
	}
}
