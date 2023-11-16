#include "CameraClass.h"
#include "SystemInputClass.h"

HRESULT CameraClass::Init(const float width, const float height, 
	ID3D11Device* m_pd3dDevice, ID3D11DeviceContext* m_pImmediateContext)
{
	HRESULT hr = S_OK;

	this->m_pd3dDevice = m_pd3dDevice;
	this->m_pImmediateContext = m_pImmediateContext;

	coordinateConstantBuffer.mWorld = XMMatrixIdentity();
	coordinateConstantBuffer.mView = XMMatrixLookAtLH(Eye, At, Up);
	coordinateConstantBuffer.mProjection = XMMatrixPerspectiveFovLH(XM_PIDIV2, width / (FLOAT)height, 0.01f, 1000.0f);

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory(&bufferDesc, sizeof(bufferDesc));
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.ByteWidth = sizeof(ConstantBuffer);
	bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	bufferDesc.CPUAccessFlags = 0;
	hr = m_pd3dDevice->CreateBuffer(&bufferDesc, NULL, &m_pConstantBuffer);

	prevEye = Eye;
	prevAt = At;

	return hr;
}

ConstantBuffer* CameraClass::GetCoordinateConstantBuffer()
{
	return &coordinateConstantBuffer;
}

ID3D11Buffer* CameraClass::GetConstantBuffer()
{
	return m_pConstantBuffer;
}

void CameraClass::SetFixedViewPoint(std::vector<XMMATRIX> world)
{
	int index = 0;

	index = (inputKey[0].key - DIK_F1)+1;

	if (index < 1)
		return;
	if (index > 8)
		return;

	if (inputKey[0].isKeyup)
	{
		Eye = prevEye;
		At = prevAt;
		SetCameraPosition();
	}
	else
	{
		At = XMVector3TransformCoord(XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f), world[index]);
		Eye = XMVector3TransformCoord(XMVectorSet(0.0f, 0.0f, 3.0f, 0.0f), world[index]);
		SetCameraPosition();
	}
}

void CameraClass::SetCameraPosition()
{
	coordinateConstantBuffer.mView = XMMatrixLookAtLH(Eye, At, Up);
}

XMVECTOR CameraClass::GetCameraEye()
{
	return Eye;
}

void CameraClass::HandleMouseMovement(float deltaTime)
{
	static float moveSpeed = 500.0f;

	if (inputMouseLocation.diffMouseX > 0)	// 마우스 우측 이동
	{
		At -= deltaTime * XMVector4Normalize(XMVector3Cross(XMVector4Normalize(At - Eye), Up)) * moveSpeed;
	}
	else if (inputMouseLocation.diffMouseX < 0)	// 마우스 좌측 이동
	{
		At += deltaTime * XMVector4Normalize(XMVector3Cross(XMVector4Normalize(At - Eye), Up)) * moveSpeed;
	}

	if(inputMouseLocation.diffMouseY > 0)	// 마우스 상단 이동
	{
		At -= deltaTime * XMVector4Normalize(Up) * moveSpeed;
	}
	else if (inputMouseLocation.diffMouseY < 0)	// 마우스 하단 이동
	{
		At += deltaTime * XMVector4Normalize(Up) * moveSpeed;
	}

	SetCameraPosition();
}

void CameraClass::HandleKeyboardMovement(float deltaTime)
{
	static float moveSpeed = 300.0;

	if (!inputKey[1].key)
		return;

	switch (inputKey[1].key)
	{
	case DIK_W:
		Eye += (XMVector4Normalize(At - Eye) * deltaTime * moveSpeed);
		At += (XMVector4Normalize(At - Eye) * deltaTime * moveSpeed);
		break;

	case DIK_A:
		Eye += (XMVector4Normalize(XMVector3Cross(XMVector4Normalize(At - Eye), Up)) * deltaTime * moveSpeed);
		At += (XMVector4Normalize(XMVector3Cross(XMVector4Normalize(At - Eye),Up)) * deltaTime * moveSpeed);
		break;

	case DIK_S:
		Eye -= (XMVector4Normalize(At - Eye) * deltaTime * moveSpeed);
		At -= (XMVector4Normalize(At - Eye) * deltaTime * moveSpeed);
		break;

	case DIK_D:
		Eye -= (XMVector4Normalize(XMVector3Cross(XMVector4Normalize(At - Eye), Up)) * deltaTime * moveSpeed);
		At -= (XMVector4Normalize(XMVector3Cross(XMVector4Normalize(At - Eye), Up)) * deltaTime * moveSpeed);
		break;

	default:
		break;
	}

	SetCameraPosition();
}

void CameraClass::Update(SystemInputClass* inputClass, std::vector<XMMATRIX> world, float deltaTime)
{
	inputClass->GetMouseLocation(inputMouseLocation.diffMouseX, inputMouseLocation.diffMouseY);	// 해당 값을 통해 view를 rotation 수행하여 시점 변환
	HandleMouseMovement(deltaTime);

	inputClass->GetFunctionKeyPressed(inputKey[0].key, inputKey[0].isKeyup);
	SetFixedViewPoint(world);

	inputClass->GetMoveKeyPressed(inputKey[1].key);
	HandleKeyboardMovement(deltaTime);

	m_pImmediateContext->VSSetConstantBuffers(0, 1, &m_pConstantBuffer);
}

void CameraClass::Shutdown()
{
	if (m_pConstantBuffer)	m_pConstantBuffer->Release();
}
