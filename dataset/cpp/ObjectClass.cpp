#include "ObjectClass.h"
#include <fstream>

using namespace std;

VertexType* ObjectClass::GetVertices()
{
	return m_vertices;
}

WORD* ObjectClass::GetIndices()
{
	return m_indices;
}

ID3D11Buffer* ObjectClass::GetVertexBuffer()
{
	return m_pVertexBuffer;
}

ID3D11Buffer* ObjectClass::GetIndexBuffer()
{
	return m_pIndexBuffer;
}


int ObjectClass::GetIndexcount()
{
	return indexCount;
}

UINT ObjectClass::GetStride()
{
	return stride;
}

UINT ObjectClass::GetOffset()
{
	return offset;
}

std::vector<XMMATRIX> ObjectClass::GetObjectCameraWorldVector()
{
	return objectCameraWorld;
}

std::vector<float> ObjectClass::GetScaleVector()
{
	return scale;
}

void ObjectClass::DynamicAllocationVertices(const int size)
{
	delete[] m_vertices;
	m_vertices = nullptr;

	if (size <= 0)
		return;

	m_vertices = new VertexType[size];
	vertexTypeCount = size;
}

void ObjectClass::DynamicAllocationIndices(const int size)
{
	delete[] m_indices;
	m_indices = nullptr;

	if (size <= 0)
		return;

	m_indices = new WORD[size];
	indexCount = size;
}

HRESULT ObjectClass::CreateVertexBuffer(ID3D11Device* pd3dDevice)
{
	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(VertexType) * vertexTypeCount;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	D3D11_SUBRESOURCE_DATA InitData;
	ZeroMemory(&InitData, sizeof(InitData));
	InitData.pSysMem = m_vertices;
	hr = pd3dDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);

	return hr;
}

HRESULT ObjectClass::CreateIndexBuffer(ID3D11Device* pd3dDevice)
{
	HRESULT hr = S_OK;

	D3D11_BUFFER_DESC bd;
	ZeroMemory(&bd, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(WORD) * indexCount;
	bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;
	D3D11_SUBRESOURCE_DATA InitData;
	ZeroMemory(&InitData, sizeof(InitData));
	InitData.pSysMem = m_indices;
	hr = pd3dDevice->CreateBuffer(&bd, &InitData, &m_pIndexBuffer);
	if (FAILED(hr))
		return hr;

	return hr;
}

void ObjectClass::Update(ID3D11DeviceContext* m_pImmediateContext,  float deltaTime)
{
	static float accumDeltaTime = 0.0f;
	accumDeltaTime += deltaTime;

	// 각 행성별 비율 : 109.25, 0.383, 0.950, 1, 0.532, 10.97, 9.14, 3.98, 3.87 
	// 태양과의 거리 : 0, 579.1, 1082, 1496, 2279, 7785, 14340, 28710, 44950
	// 궤도속도 : 0, 48, 35 ,30, 24, 13, 9.6, 7, 5
	// 자전속도 : 1.9, 0.003, 0.0018, 0.4651, 0.2411, 12.6, 9.8, 2.59, 2.68
	// 자전기울기 : 7.25, 0.01, 2.64, 23.44, 25.19, 3.12, 26.73, 82.23, 28.33

	mWorld[0] = XMMatrixScaling(scale[0], scale[0], scale[0]) * XMMatrixRotationX(rotationAngle[0] * PI /180.0f) * XMMatrixRotationY(accumDeltaTime * rotationSpeed[0]);
	for (int i = 1; i < SOLAR_SYSTEM_SIZE; i++)
	{
		mWorld[i] = XMMatrixScaling(scale[i], scale[i], scale[i]) * XMMatrixRotationX(rotationAngle[i] * PI / 180.0f) * XMMatrixRotationY(accumDeltaTime * rotationSpeed[i])
			* XMMatrixTranslation(-distance[i], 0.0f, 0.0f) * XMMatrixRotationY(accumDeltaTime * revolutionSpeed[i]);

		objectCameraWorld[i] = XMMatrixScaling(scale[i], scale[i], scale[i]) * XMMatrixTranslation(-distance[i], 0.0f, 0.0f) * XMMatrixRotationY(accumDeltaTime * revolutionSpeed[i]);
	}
}

void ObjectClass::Render(ID3D11DeviceContext* m_pImmediateContext, CameraClass* cameraClass, vector<ID3D11ShaderResourceView*> shaderResourceView, GraphicClass* graphicClass)
{
	m_pImmediateContext->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	graphicClass->SetVertexShader();

	for (int i = 0; i < SOLAR_SYSTEM_SIZE; i++)	// 태양,수금지화목토천해
	{
		if (i == 0)
		{
			graphicClass->SetPixelShader(PixelShaderNumber::normalPixelShader);
		}
		else
		{
			graphicClass->SetPixelShader(PixelShaderNumber::lightPixelShader);
		}

		m_pImmediateContext->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &stride, &offset);
		m_pImmediateContext->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);

		constantBufferData[i].mWorld = XMMatrixTranspose(mWorld[i]);
		constantBufferData[i].mView = XMMatrixTranspose(cameraClass->GetCoordinateConstantBuffer()->mView);
		constantBufferData[i].mProjection = XMMatrixTranspose(cameraClass->GetCoordinateConstantBuffer()->mProjection);

		m_pImmediateContext->RSSetState(graphicClass->GetGraphicRasterizerState());

		m_pImmediateContext->UpdateSubresource(cameraClass->GetConstantBuffer(), 0, NULL, &constantBufferData[i], 0, 0);
		m_pImmediateContext->PSSetShaderResources(0, 1, &shaderResourceView[i]);
		m_pImmediateContext->DrawIndexed(indexCount, 0, 0);
	}
}

ObjectClass::ObjectClass()
{
	const char* solarSystemDataFilePath = "Textures/data.txt";

	ConstantBuffer constantBuffer;
	for (int i = 0; i < SOLAR_SYSTEM_SIZE; i++)
	{
		mWorld.emplace_back(XMMatrixIdentity());
		objectCameraWorld.emplace_back(XMMatrixIdentity());
		constantBufferData.emplace_back(constantBuffer);
	}

	ifstream fin;
	float data;

	fin.open(solarSystemDataFilePath);

	if (fin.fail())
	{
		return;
	}

	// 크기, 자전기울기, 자전속도, 거리, 공전속도
	while (!fin.eof())
	{
		fin >> data;					
		scale.emplace_back(data);				// 크기

		fin >> data;
		rotationAngle.emplace_back(data);		// 자전기울기

		fin >> data;
		rotationSpeed.emplace_back(data);		// 자전속도

		fin >> data;
		distance.emplace_back(data);			// 거리

		fin >> data;
		revolutionSpeed.emplace_back(data);		// 공전속도
	}
}

ObjectClass::~ObjectClass()
{
	if (m_indices)
	{
		delete[] m_indices;
		m_indices = nullptr;
	}
	
	if (m_vertices)
	{
		delete[] m_vertices;
		m_vertices = nullptr;
	}

	if (m_pIndexBuffer)	m_pIndexBuffer->Release();
	if (m_pVertexBuffer)	m_pVertexBuffer->Release();
}
