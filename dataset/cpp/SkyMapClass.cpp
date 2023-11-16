#include "SkyMapClass.h"
#include "CameraClass.h"
#include "GraphicClass.h"
#include <d3dcompiler.h>

HRESULT SkyMapClass::Init(ID3D11Device * pd3dDevice, ID3D11DeviceContext* immediateContext)
{
	HRESULT hr;

	m_pd3dDevice = pd3dDevice;
	m_pImmediateContext = immediateContext;

	D3D11_RASTERIZER_DESC RSDesc;
	ZeroMemory(&RSDesc, sizeof(D3D11_RASTERIZER_DESC));
	RSDesc.FrontCounterClockwise = true;
	RSDesc.FillMode = D3D11_FILL_SOLID;
	RSDesc.CullMode = D3D11_CULL_NONE;

	hr = m_pd3dDevice->CreateRasterizerState(&RSDesc, &m_pRasterizerState);
	if (FAILED(hr))
	{
		MessageBox(NULL,
			"create rasterizerstate Error", "Error", MB_OK);
		return hr;
	}

	D3D11_DEPTH_STENCIL_DESC dssDesc;
	ZeroMemory(&dssDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	dssDesc.DepthEnable = true;
	dssDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dssDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;

	hr = m_pd3dDevice->CreateDepthStencilState(&dssDesc, &m_pDSLessEqual);
	if (FAILED(hr))
	{
		MessageBox(NULL,
			"create depthStencilState Error", "Error", MB_OK);
		return hr;
	}

	D3DX11_IMAGE_LOAD_INFO loadSMInfo;
	loadSMInfo.MiscFlags = D3D11_RESOURCE_MISC_TEXTURECUBE;
	hr = D3DX11CreateShaderResourceViewFromFile(m_pd3dDevice, fileName, &loadSMInfo, 0, &m_pTextureView, 0);
	if (FAILED(hr))
	{
		MessageBox(NULL,
			"ShaderResourceView Load Error", "Error", MB_OK);
		return hr;
	}

	ID3DBlob* pVSBlob = NULL;
	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG;
	wchar_t szfileName[20] = L"temp.fx";
	hr = D3DX11CompileFromFileW(szfileName, NULL, NULL, "SKYMAP_VS", "vs_4_0", dwShaderFlags, 0, NULL, &pVSBlob, NULL, NULL);
	if (FAILED(hr))
	{
		MessageBox(NULL,
			"fx error, (SKYMAP_VS)", "Error", MB_OK);
		return hr;
	}

	hr = m_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &m_pVertexShader);
	if (FAILED(hr))
	{
		pVSBlob->Release();
		return hr;
	}

	ID3DBlob* pPSBlob = NULL;
	hr = D3DX11CompileFromFileW(szfileName, NULL, NULL, "SKYMAP_PS", "ps_4_0", dwShaderFlags, 0, NULL, &pPSBlob, NULL, NULL);
	if (FAILED(hr))
	{
		MessageBox(NULL,
			"fx error, (SKYMAP_PS)", "Error", MB_OK);
		return hr;
	}

	hr = m_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &m_pPixelShader);
	pPSBlob->Release();
	if (FAILED(hr))
		return hr;

	return hr;
}

VertexType* SkyMapClass::GetVertices()
{
	return m_vertices;
}

WORD* SkyMapClass::GetIndices()
{
	return m_indices;
}

void SkyMapClass::DynamicAllocationVertices(const int size)
{
	delete[] m_vertices;
	m_vertices = nullptr;

	if (size <= 0)
		return;

	m_vertices = new VertexType[size];
	vertexTypeCount = size;
}

void SkyMapClass::DynamicAllocationIndices(const int size)
{
	delete[] m_indices;
	m_indices = nullptr;

	if (size <= 0)
		return;

	m_indices = new WORD[size];
	indexCount = size;
}

HRESULT SkyMapClass::CreateVertexBuffer()
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
	hr = m_pd3dDevice->CreateBuffer(&bd, &InitData, &m_pVertexBuffer);

	return hr;
}

HRESULT SkyMapClass::CreateIndexBuffer()
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
	hr = m_pd3dDevice->CreateBuffer(&bd, &InitData, &m_pIndexBuffer);
	if (FAILED(hr))
		return hr;

	return hr;
}

void SkyMapClass::Render(CameraClass* cameraClass, GraphicClass* graphicClass)
{
	m_pImmediateContext->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	m_pImmediateContext->VSSetShader(m_pVertexShader, NULL, 0);
	m_pImmediateContext->PSSetShader(m_pPixelShader, NULL, 0);

	m_pImmediateContext->IASetVertexBuffers(0, 1, &m_pVertexBuffer, &stride, &offset);
	m_pImmediateContext->IASetIndexBuffer(m_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);

	m_pImmediateContext->OMSetDepthStencilState(m_pDSLessEqual, 0);
	m_pImmediateContext->RSSetState(m_pRasterizerState);
	ID3D11RenderTargetView* rendertargetView = graphicClass->GetRenderTargetView();
	m_pImmediateContext->OMSetRenderTargets(1, &rendertargetView, NULL);

	XMMATRIX mWorld = XMMatrixTranslation(XMVectorGetX(cameraClass->GetCameraEye()), XMVectorGetY(cameraClass->GetCameraEye()), XMVectorGetZ(cameraClass->GetCameraEye()));

	constantBufferData.mWorld = XMMatrixTranspose(mWorld);
	constantBufferData.mView = XMMatrixTranspose(cameraClass->GetCoordinateConstantBuffer()->mView);
	constantBufferData.mProjection = XMMatrixTranspose(cameraClass->GetCoordinateConstantBuffer()->mProjection);

	m_pImmediateContext->UpdateSubresource(cameraClass->GetConstantBuffer(), 0, NULL, &constantBufferData, 0, 0);

	m_pImmediateContext->PSSetShaderResources(1, 1, &(m_pTextureView));
	m_pImmediateContext->DrawIndexed(indexCount, 0, 0);

	m_pImmediateContext->OMSetRenderTargets(1, &rendertargetView, graphicClass->GetDepthStencilView());
}

SkyMapClass::~SkyMapClass()
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

	if (m_pIndexBuffer)		m_pIndexBuffer->Release();
	if (m_pVertexBuffer)	m_pVertexBuffer->Release();
	if (m_pRasterizerState) m_pRasterizerState->Release();
	if (m_pDSLessEqual)	m_pDSLessEqual->Release();
}
