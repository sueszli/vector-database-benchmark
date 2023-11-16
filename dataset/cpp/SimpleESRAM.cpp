//--------------------------------------------------------------------------------------
// SimpleESRAM.cpp
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "SimpleESRAM.h"
#include "PIXHelpers.h"

#include "ATGColors.h"
#include "ControllerFont.h"

extern void ExitSample() noexcept;

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    //--------------------------------------
    // Definitions

    // Barebones definition of scene objects.
    struct ObjectDefinition
    {
        size_t              modelIndex;
        SimpleMath::Matrix  world;
    };


    //--------------------------------------
    // Constants

    const float c_defaultPhi = XM_2PI / 6.0f;
    const float c_defaultRadius = 3.3f;

    // Assest paths.
    const wchar_t* s_modelPaths[] =
    {
        L"scanner.sdkmesh",
        L"occcity.sdkmesh",
        L"column.sdkmesh",
    };

    // Barebones definition of a scene.
    const ObjectDefinition s_sceneDefinition[] = 
    {
        { 0, XMMatrixIdentity() },
        { 0, XMMatrixRotationY(XM_2PI * (1.0f / 6.0f)) },
        { 0, XMMatrixRotationY(XM_2PI * (2.0f / 6.0f)) },
        { 0, XMMatrixRotationY(XM_2PI * (3.0f / 6.0f)) },
        { 0, XMMatrixRotationY(XM_2PI * (4.0f / 6.0f)) },
        { 0, XMMatrixRotationY(XM_2PI * (5.0f / 6.0f)) },
        { 1, XMMatrixIdentity() },
        { 2, XMMatrixIdentity() },
    };

    // Full screen triangle geometry definition.
    std::vector<GeometricPrimitive::VertexType> s_triVertex =
    {
        { XMFLOAT3{ -1.0f,  1.0f, 0.0f }, XMFLOAT3{ 0.0f, 0.0f, -1.0f }, XMFLOAT2{ 0.0f, 0.0f } },  // Top-left
        { XMFLOAT3{ 3.0f,  1.0f, 0.0f }, XMFLOAT3{ 0.0f, 0.0f, -1.0f }, XMFLOAT2{ 0.0f, 2.0f } },   // Top-right
        { XMFLOAT3{ -1.0f, -3.0f, 0.0f }, XMFLOAT3{ 0.0f, 0.0f, -1.0f }, XMFLOAT2{ 2.0f, 0.0f } },  // Bottom-left
    };

    std::vector<uint16_t> s_triIndex = { 0, 1, 2 };

    const DXGI_FORMAT   c_colorFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    const XG_FORMAT     c_colorXGFormat = XG_FORMAT_R8G8B8A8_UNORM;
    const DXGI_FORMAT   c_depthFormat = DXGI_FORMAT_D32_FLOAT;
    const XG_FORMAT     c_depthXGFormat = XG_FORMAT_D32_FLOAT;

    const int           c_esramPageCount = 512;
    const int           c_pageSize = 64 * 1024;

    const int           c_esramTexWidth = 4096;
    const int           c_esramTexHeight = 2048;
    

    //--------------------------------------
    // Helper Functions

    template <typename T> 
    constexpr T DivRoundUp(T num, T denom) 
    { 
        return (num + denom - 1) / denom; 
    }

    template <typename T>
    constexpr T PageCount(T byteSize)
    {
        return DivRoundUp(byteSize, T(c_pageSize));
    }

    template <typename T, typename U, typename V>
    constexpr T Clamp(T value, U min, V max)
    {
        return (value < min) ? min : ((value > max) ? max : value);
    }

    constexpr float Saturate(float value)
    {
        return Clamp(value, 0.0f, 1.0f);
    }

    bool IsMetadataPlane(const XG_PLANE_LAYOUT& layout)
    {
        return layout.Usage == XG_PLANE_USAGE_COLOR_MASK
            || layout.Usage == XG_PLANE_USAGE_FRAGMENT_MASK
            || layout.Usage == XG_PLANE_USAGE_HTILE
#if _XDK_VER >= 0x3F6803F3 /* XDK Edition 170600 */
            || layout.Usage == XG_PLANE_USAGE_DELTA_COLOR_COMPRESSION
#endif
            ;
    }

    void FillLayoutDesc(XG_RESOURCE_LAYOUT layout, MetadataDesc& desc)
    {
        desc.count = 0;

        // Iterate through the resource planes and find the page ranges that contain resource metadata.
        for (unsigned i = 0; i < layout.Planes; ++i)
        {
            auto& plane = layout.Plane[i];

            // Only process metadata planes.
            if (IsMetadataPlane(plane))
            {
                int startPage = int(plane.BaseOffsetBytes / c_pageSize);
                int endPage = int((plane.BaseOffsetBytes + plane.SizeBytes) / c_pageSize);

                if (desc.count > 0 && startPage == desc.ranges[desc.count - 1].End() - 1)
                {
                    // Merge the ranges if the current page range is adjacent to the previous one.
                    desc.ranges[desc.count - 1].count += endPage - startPage;
                }
                else
                {
                    // Add a new range since it's disjoint from the previous one.
                    desc.ranges[desc.count++] = { startPage, endPage - startPage + 1 };
                }
            }
        }
    }

    // Xbox One X doesn't have ESRAM, and attempted ESRAM allocations will throw an error.
    bool SupportsESRAM()
    {
#if _XDK_VER >= 0x3F6803F3 /* XDK Edition 170600 */
        return GetConsoleType() <= CONSOLE_TYPE::CONSOLE_TYPE_XBOX_ONE_S;
#else
        return true;
#endif
    }

    // Creates a D3D11 resource by either ID3D11DeviceX::CreatePlacedResourceX(...), if a virtual 
    // address is supplied, or ID3D11DeviceX::CreateCommittedResource(...) using the supplied descriptor.
    void CreateResource(
        _In_ ID3D11DeviceX* device,
        const D3D11_TEXTURE2D_DESC& desc,
        UINT tileModeIndex,
        _In_opt_ const XG_GPU_VIRTUAL_ADDRESS* placementAddress,
        _In_opt_ const wchar_t* name,
        _Outptr_ ID3D11Texture2D** outResource)
    {
        assert(device != nullptr);

        if (placementAddress == nullptr)
        {
            device->CreateTexture2D(&desc, nullptr, outResource);
        }
        else
        {
            device->CreatePlacementTexture2D(&desc, tileModeIndex, 0, (void*)*placementAddress, outResource);
        }

        if (name != nullptr)
        {
            assert(outResource != nullptr);
            (*outResource)->SetName(name);
        }
    }

    // Creates a color D3D11 texture 2D and render target view for that resource.
    void CreateColorResourceAndView(
        _In_ ID3D11DeviceX* device,
        const D3D11_TEXTURE2D_DESC& desc,
        _In_opt_ const XG_GPU_VIRTUAL_ADDRESS* placementAddress,
        _In_opt_ const wchar_t* name,
        _Outptr_ ID3D11Texture2D** outResource,
        _Outptr_ ID3D11RenderTargetView** outView)
    {
        assert(device != nullptr);

        if (outResource == nullptr)
        {
            return;
        }

        XG_TILE_MODE colorTileMode = XGComputeOptimalTileMode(
            XG_RESOURCE_DIMENSION_TEXTURE2D,
            XG_FORMAT(desc.Format),
            UINT(desc.Width),
            desc.Height,
            desc.ArraySize,
            desc.SampleDesc.Count,
            XG_BIND_RENDER_TARGET);

        CreateResource(device, desc, colorTileMode, placementAddress, name, outResource);

        D3D11_RTV_DIMENSION dim = desc.SampleDesc.Count == 1 ? D3D11_RTV_DIMENSION_TEXTURE2D : D3D11_RTV_DIMENSION_TEXTURE2DMS;
        D3D11_RENDER_TARGET_VIEW_DESC rtvDesc =
        {
            desc.Format,    // DXGI_FORMAT Format;
            dim,            // D3D11_RTV_DIMENSION ViewDimension;
            {               // D3D11_TEX2DMS_RTV Texture2D;
                0,          // UINT UnusedField_NothingToDefine;
            },
        };
        device->CreateRenderTargetView(*outResource, &rtvDesc, outView);
    }


    // Creates a depth D3D11 texture 2D and depth stencil view for that resource.
    void CreateDepthResourceAndView(
        _In_ ID3D11DeviceX* device,
        const D3D11_TEXTURE2D_DESC& desc,
        _In_opt_ const XG_GPU_VIRTUAL_ADDRESS* placementAddress,
        _In_opt_ const wchar_t* name,
        _Outptr_ ID3D11Texture2D** outResource,
        _Outptr_ ID3D11DepthStencilView** outView)
    {
        assert(device != nullptr);

        if (outResource == nullptr)
        {
            return;
        }
        
        XG_TILE_MODE depthTileMode;
        XG_TILE_MODE stencilTileMode;
#if _XDK_VER >= 0x3F6803F3 /* XDK Edition 170600 */
        XGComputeOptimalDepthStencilTileModes(
            XG_FORMAT(desc.Format),
            UINT32(desc.Width),
            desc.Height,
            desc.ArraySize,
            desc.SampleDesc.Count,
            (desc.MiscFlags & D3D11X_RESOURCE_MISC_NO_DEPTH_COMPRESSION) == 0,
            FALSE,
            FALSE,
            &depthTileMode,
            &stencilTileMode);
#else
        XGComputeOptimalDepthStencilTileModes(
            XG_FORMAT(desc.Format),
            UINT32(desc.Width),
            desc.Height,
            desc.ArraySize,
            desc.SampleDesc.Count,
            TRUE,
            &depthTileMode,
            &stencilTileMode);
#endif

        CreateResource(device, desc, depthTileMode, placementAddress, name, outResource);

        D3D11_DSV_DIMENSION dim = desc.SampleDesc.Count == 1 ? D3D11_DSV_DIMENSION_TEXTURE2D : D3D11_DSV_DIMENSION_TEXTURE2DMS;
        D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc =
        {
            desc.Format,            // DXGI_FORMAT Format;
            dim,                    // D3D11_DSV_DIMENSION ViewDimension;
            0,                      // UINT Flags
            {                       // D3D11_TEX2DMS_DSV Texture2D;
                0,                  // UINT UnusedField_NothingToDefine;
            },
        };
        device->CreateDepthStencilView(*outResource, &dsvDesc, outView);
    }

    // Computes the optimal tile mode and calculates the number of 64 KiB pages necessary for the resource.
    // Optionally computes the ranges of 64KiB pages the resource metadata consumes if present.
    int CalculatePagesForColorResource(D3D11_TEXTURE2D_DESC& desc, XG_FORMAT colorFormat, MetadataDesc* layoutDesc = nullptr)
    {
        // Determine the size and alignment for resource
        XG_TILE_MODE colorTileMode = XGComputeOptimalTileMode(
            XG_RESOURCE_DIMENSION_TEXTURE2D,
            colorFormat,
            UINT(desc.Width),
            desc.Height,
            desc.ArraySize,
            desc.SampleDesc.Count,
            XG_BIND_RENDER_TARGET);

        struct XG_TEXTURE2D_DESC xgDesc =
        {
            desc.Width,                     // UINT Width;
            desc.Height,                    // UINT Height;
            desc.MipLevels,                 // UINT MipLevels;
            desc.ArraySize,                 // UINT ArraySize;
            colorFormat,                    // XG_FORMAT Format;
            {                               // XG_SAMPLE_DESC SampleDesc;
                desc.SampleDesc.Count,      // UINT Count;
                desc.SampleDesc.Quality,    // UINT Quality;
            },
            XG_USAGE(desc.Usage),           // XG_USAGE Usage;       
            desc.BindFlags,                 // UINT BindFlags;
            desc.CPUAccessFlags,            // UINT CPUAccessFlags;
            desc.MiscFlags,                 // UINT MiscFlags;
            0,                              // UINT ESRAMOffsetBytes;
            0,                              // UINT ESRAMUsageBytes;
            colorTileMode,                  // XG_TILE_MODE TileMode;
            0,                              // UINT Pitch;
        };

        ComPtr<XGTextureAddressComputer> computer;
        DX::ThrowIfFailed(XGCreateTexture2DComputer(&xgDesc, computer.GetAddressOf()));

        XG_RESOURCE_LAYOUT layout;
        DX::ThrowIfFailed(computer->GetResourceLayout(&layout));

        if (layoutDesc != nullptr)
        {
            FillLayoutDesc(layout, *layoutDesc);
        }

        return (int)PageCount(layout.SizeBytes);
    }


    // Computes the optimal tile mode and calculates the number of 64 KiB pages necessary for the resource.
    // Optionally computes the ranges of 64KiB pages the resource metadata consumes if present.
    int CalculatePagesForDepthResource(D3D11_TEXTURE2D_DESC& desc, XG_FORMAT depthFormat, MetadataDesc* layoutDesc = nullptr)
    {
        XG_TILE_MODE depthTileMode;
        XG_TILE_MODE stencilTileMode;

#if _XDK_VER >= 0x3F6803F3 /* XDK Edition 170600 */
        XGComputeOptimalDepthStencilTileModes(
            depthFormat,
            UINT32(desc.Width),
            desc.Height,
            desc.ArraySize,
            desc.SampleDesc.Count,
            (desc.MiscFlags & D3D11X_RESOURCE_MISC_NO_DEPTH_COMPRESSION) == 0,
            FALSE,
            FALSE,
            &depthTileMode,
            &stencilTileMode);
#else
        XGComputeOptimalDepthStencilTileModes(
            depthFormat,
            UINT32(desc.Width),
            desc.Height,
            desc.ArraySize,
            desc.SampleDesc.Count,
            TRUE,
            &depthTileMode,
            &stencilTileMode);
#endif

        struct XG_TEXTURE2D_DESC xgDesc =
        {
            desc.Width,                     // UINT Width;
            desc.Height,                    // UINT Height;
            desc.MipLevels,                 // UINT MipLevels;
            desc.ArraySize,                 // UINT ArraySize;
            depthFormat,                    // XG_FORMAT Format;
            {                               // XG_SAMPLE_DESC SampleDesc;
                desc.SampleDesc.Count,      // UINT Count;
                desc.SampleDesc.Quality,    // UINT Quality;
            },
            XG_USAGE(desc.Usage),           // XG_USAGE Usage;       
            desc.BindFlags,                 // UINT BindFlags;
            desc.CPUAccessFlags,            // UINT CPUAccessFlags;
            desc.MiscFlags,                 // UINT MiscFlags;
            0,                              // UINT ESRAMOffsetBytes;
            0,                              // UINT ESRAMUsageBytes;
            depthTileMode,                  // XG_TILE_MODE TileMode;
            0,                              // UINT Pitch;
        };


        ComPtr<XGTextureAddressComputer> computer;
        DX::ThrowIfFailed(XGCreateTexture2DComputer(&xgDesc, computer.GetAddressOf()));

        XG_RESOURCE_LAYOUT layout;
        DX::ThrowIfFailed(computer->GetResourceLayout(&layout));

        if (layoutDesc != nullptr)
        {
            FillLayoutDesc(layout, *layoutDesc);
        }

        return (int)PageCount(layout.SizeBytes);
    }
}


Sample::Sample() 
    : m_displayWidth(0)
    , m_displayHeight(0)
    , m_frame(0)
    , m_theta(0.f)
    , m_phi(c_defaultPhi)
    , m_radius(c_defaultRadius)
    , m_generator(std::random_device()())
    , m_showOverlay(SupportsESRAM())
    , m_mapScheme(SupportsESRAM() ? EMS_Simple : EMS_None)
    , m_colorDesc{}
    , m_depthDesc{}
    , m_esramOverlayDesc{}
{ 
    m_deviceResources = std::make_unique<DX::DeviceResources>(c_colorFormat, DXGI_FORMAT_UNKNOWN);
}

// Initialize the Direct3D resources required to run.
void Sample::Initialize(IUnknown* window)
{
    m_deviceResources->SetWindow(window);

    m_deviceResources->CreateDeviceResources();  
    CreateDeviceDependentResources();

    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();
}

#pragma region Frame Update
// Executes basic render loop.
void Sample::Tick()
{
    PIXBeginEvent(PIX_COLOR_DEFAULT, L"Frame %llu", m_frame);

    m_timer.Tick([&]()
    {
        Update(m_timer);
    });

    Render();

    PIXEndEvent();
    m_frame++;
}

void Sample::Update(const DX::StepTimer& timer)
{
    using ButtonState = DirectX::GamePad::ButtonStateTracker::ButtonState;

    ScopedPixEvent Update(PIX_COLOR_DEFAULT, L"Update");

    float elapsedTime = float(timer.GetElapsedSeconds());

    auto pad = m_gamePad.GetState(0);
    if (pad.IsConnected())
    {
        m_gamePadButtons.Update(pad);

        if (pad.IsViewPressed())
        {
            ExitSample();
        }

        bool recreateResources = false;

        if (SupportsESRAM())
        {
            if (m_gamePadButtons.dpadLeft == ButtonState::RELEASED)
            {
                m_mapScheme = EsramMappingScheme((m_mapScheme == 0 ? EMS_Count : m_mapScheme) - 1);
                recreateResources = true;
            }
            else if (m_gamePadButtons.dpadRight == ButtonState::RELEASED)
            {
                m_mapScheme = EsramMappingScheme((m_mapScheme + 1) % EMS_Count);
                recreateResources = true;
            }

            if (m_gamePadButtons.a == ButtonState::RELEASED)
            {
                m_showOverlay = !m_showOverlay;
            }
        }

        const float splitSpeed = 0.2f * elapsedTime;

        switch (m_mapScheme)
        {
        case EMS_Simple:
            if (pad.IsLeftShoulderPressed())
            {
                int maxPageCount = std::min(m_colorPageCount, c_esramPageCount);

                m_colorEsramPageCount = std::min(m_colorEsramPageCount + 1, maxPageCount);
                m_depthEsramPageCount = std::min(m_depthEsramPageCount, c_esramPageCount - m_colorEsramPageCount);

                recreateResources = true;
            }

            if (pad.IsLeftTriggerPressed())
            {
                m_colorEsramPageCount = std::max(m_colorEsramPageCount - 1, 0);
                recreateResources = true;
            }

            if (pad.IsRightShoulderPressed())
            {
                int maxPageCount = std::min(m_depthPageCount, c_esramPageCount);

                m_depthEsramPageCount = std::min(m_depthEsramPageCount + 1, maxPageCount);
                m_colorEsramPageCount = std::min(m_colorEsramPageCount, c_esramPageCount - m_depthEsramPageCount);

                recreateResources = true;
            }

            if (pad.IsRightTriggerPressed())
            {
                m_depthEsramPageCount = std::max(m_depthEsramPageCount - 1, 0);
                recreateResources = true;
            }
            break;

        case EMS_Split:
            if (pad.IsLeftShoulderPressed())
            {
                m_bottomPercent = std::min(m_bottomPercent + splitSpeed, m_topPercent);
                recreateResources = true;
            }

            if (pad.IsLeftTriggerPressed())
            {
                m_bottomPercent = std::max(m_bottomPercent - splitSpeed, 0.0f);
                recreateResources = true;
            }

            if (pad.IsRightShoulderPressed())
            {
                m_topPercent = std::min(m_topPercent + splitSpeed, 1.0f);
                recreateResources = true;
            }

            if (pad.IsRightTriggerPressed())
            {
                m_topPercent = std::max(m_topPercent - splitSpeed, m_bottomPercent);
                recreateResources = true;
            }
            break;

        case EMS_Metadata:
            if (m_gamePadButtons.b == ButtonState::RELEASED)
            {
                m_metadataEnabled = !m_metadataEnabled;
                recreateResources = true;
            }
            break;

        case EMS_Random:
            if (m_gamePadButtons.leftShoulder == ButtonState::RELEASED)
            {
                m_esramProbability = std::max(m_esramProbability - 0.1f, 0.0f); // Decrease by 10% per press.
                recreateResources = true;
            }

            if (m_gamePadButtons.rightShoulder == ButtonState::RELEASED)
            {
                m_esramProbability = std::min(m_esramProbability + 0.1f, 1.0f); // Increase by 10% per press.
                recreateResources = true;
            }
            break;
        }

        if (recreateResources)
        {
            UpdateResourceMappings();
        }

        if (pad.IsRightStickPressed())
        {
            m_theta = 0.f;
            m_phi = c_defaultPhi;
            m_radius = c_defaultRadius;
        }
        else
        {
            m_theta += pad.thumbSticks.rightX * XM_PI * elapsedTime;
            m_phi -= pad.thumbSticks.rightY * XM_PI * elapsedTime;
            m_radius -= pad.thumbSticks.leftY * 5.f * elapsedTime;
        }
    }
    else
    {
        m_gamePadButtons.Reset();
    }

    // Limit to avoid looking directly up or down
    m_phi = std::max(1e-2f, std::min(XM_PIDIV2, m_phi));
    m_radius = std::max(1.f, std::min(10.f, m_radius));

    if (m_theta > XM_PI)
    {
        m_theta -= XM_PI * 2.f;
    }
    else if (m_theta < -XM_PI)
    {
        m_theta += XM_PI * 2.f;
    }

    XMVECTOR lookFrom = XMVectorSet(
        m_radius * sinf(m_phi) * cosf(m_theta),
        m_radius * cosf(m_phi),
        m_radius * sinf(m_phi) * sinf(m_theta),
        0);

    m_view = XMMatrixLookAtLH(lookFrom, g_XMZero, g_XMIdentityR1);
}
#pragma endregion

#pragma region Frame Render
void Sample::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

    // Prepare the command list to render a new frame.
    m_deviceResources->Prepare();
    auto context = m_deviceResources->GetD3DDeviceContext();

    // Begin frame
    m_profiler->BeginFrame(context);
    m_profiler->Start(context);

    // Set descriptor heaps
    {
        ScopedPixEvent Update(context, PIX_COLOR_DEFAULT, L"Clear");

        ID3D11RenderTargetView* rtvs[] = { m_colorRTV.Get() };
        context->OMSetRenderTargets(1, rtvs, m_depthDSV.Get());

        auto viewport = m_deviceResources->GetScreenViewport();
        context->RSSetViewports(1, &viewport);

        // Perform a manual full screen clear operation.
        // NOTE: The ClearRenderTargetView() call on a compressed render target defers the clear to a barrier 
        //       transition. Since we are "tinting" the color buffer via the ESRAM overlay before the ResourceBarrier 
        //       transition this produces the wrong result.
        //       An alternative workaround is to simply use D3D11X_RESOURCE_MISC_NO_COLOR_COMPRESSION & D3D11X_RESOURCE_MISC_NO_COLOR_EXPAND when creating
        //       the aliased buffer. Of course, this disables potential optimizations created by that compression data.
        if (SupportsESRAM())
        {
            m_manualClearEffect->Apply(context);
            m_fullScreenTri->Draw(m_manualClearEffect.get(), m_inputLayout.Get());
        }
        else
        {
            context->ClearRenderTargetView(m_colorRTV.Get(), ATG::ColorsLinear::Background);
        }

        context->ClearDepthStencilView(m_depthDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    }

    {
        ScopedPixEvent Render(context, PIX_COLOR_DEFAULT, L"Render");

        {
            ScopedPixEvent Scene(context, PIX_COLOR_DEFAULT, L"Scene");
            
            // Draw the scene.
            for (auto& obj : m_scene)
            {
                obj.model->Draw(context, *m_commonStates, obj.world, m_view, m_proj);
            }
        }

        // Visualize ESRAM by performing a full screen draw on a render target that covers
        // the entirety of ESRAM with alpha blending. This thrashes all non-color targets
        // residing in ESRAM (including compression textures.)
        if (m_showOverlay)
        {
            ScopedPixEvent VisEsram(context, PIX_COLOR_DEFAULT, L"Visualize ESRAM");

            ID3D11RenderTargetView* rtvs[] = { m_esramOverlayRTV.Get() };
            context->OMSetRenderTargets(1, rtvs, nullptr);

            auto viewport = CD3D11_VIEWPORT(0.0f, 0.0f, float(c_esramTexWidth), float(c_esramTexHeight));
            context->RSSetViewports(1, &viewport);

            m_esramBlendEffect->Apply(context);
            m_fullScreenTri->Draw(m_esramBlendEffect.get(), m_inputLayout.Get(), true);
        }

        // Only profile pertinent ESRAM resource usage.
        m_profiler->Stop(context);
        m_profiler->EndFrame(context);

        // Since our render target is a 2xMSAA target this ResolveSubresource(...) call is required on all platforms.
        // However be wary about lingering copies from ESRAM to DRAM when porting to Xbox One X. On the One X all 
        // resources are in DRAM, so these seemingly innocuous and routine copies may ultimately be a non-trivial 
        // waste of GPU bandwidth and copy fences.
        {
            ScopedPixEvent Resolve(context, PIX_COLOR_DEFAULT, L"Resolve");

            context->ResolveSubresource(m_deviceResources->GetRenderTarget(), 0, m_colorTexture.Get(), 0, c_colorFormat);
        }

        // Draw the HUD.
        {
            ScopedPixEvent HUD(context, PIX_COLOR_DEFAULT, L"HUD");

            ID3D11RenderTargetView* rtvs[] = { m_deviceResources->GetRenderTargetView() };
            context->OMSetRenderTargets(1, rtvs, nullptr);

            auto viewport = m_deviceResources->GetScreenViewport();
            context->RSSetViewports(1, &viewport);

            m_hudBatch->Begin();

            auto size = m_deviceResources->GetOutputSize();
            auto safe = SimpleMath::Viewport::ComputeTitleSafeArea(size.right, size.bottom);

            wchar_t textBuffer[128] = {};
            XMFLOAT2 textPos = XMFLOAT2(float(safe.left), float(safe.left));
            XMVECTOR textColor = Colors::DarkKhaki;

            // Draw title.
            m_smallFont->DrawString(m_hudBatch.get(), L"Simple ESRAM", textPos, textColor);
            textPos.y += m_smallFont->GetLineSpacing();

            // Draw ESRAM usage.
            const wchar_t* titleText = nullptr;
            switch (m_mapScheme)
            {
            case EMS_None:
                titleText = L"DRAM Mapping";
                textBuffer[0] = '\0';
                break;

            case EMS_Simple:
                titleText = L"Simple Mapping";
                _snwprintf_s(textBuffer, _TRUNCATE, L"ESRAM Page Count: Color = %d, Depth = %d", m_colorEsramPageCount, m_depthEsramPageCount);
                break;

            case EMS_Split:
                titleText = L"Split Mapping";
                _snwprintf_s(textBuffer, _TRUNCATE, L"Bottom Percent = %0.2f%%, Top Percent %0.2f%%", m_bottomPercent * 100.0f, m_topPercent * 100.0f);
                break;

            case EMS_Metadata:
                titleText = L"Metadata Mapping";
                _snwprintf_s(textBuffer, _TRUNCATE, L"Metadata Mapping: %s", m_metadataEnabled ? L"Enabled" : L"Disabled");
                break;

            case EMS_Random:
                titleText = L"Random Mapping";
                _snwprintf_s(textBuffer, _TRUNCATE, L"Probability = %0.2f%%", m_esramProbability * 100.0f);
                break;
            }

            m_smallFont->DrawString(m_hudBatch.get(), titleText, textPos, textColor);
            textPos.y += m_smallFont->GetLineSpacing();
            m_smallFont->DrawString(m_hudBatch.get(), textBuffer, textPos, textColor);
            textPos.y += m_smallFont->GetLineSpacing();

            // Draw Frame Stats
            _snwprintf_s(textBuffer, _TRUNCATE, L"GPU time = %3.3lf ms", m_profiler->GetAverageMS());
            m_smallFont->DrawString(m_hudBatch.get(), textBuffer, textPos, textColor);

            // Draw Controllers
            textPos.y = float(safe.bottom) - 2.0f * m_smallFont->GetLineSpacing();
            const wchar_t* commonCtrl = nullptr;
            if (SupportsESRAM())
            {
                commonCtrl = L"[LThumb] Toward/Away   [RThumb]: Orbit Camera    [DPad] Switch Mapping Schemes    [A] Toggle Overlay    [View] Exit ";
            }
            else
            {
                commonCtrl = L"[LThumb] Toward/Away   [RThumb]: Orbit Camera   [View] Exit ";
            }

            DX::DrawControllerString(m_hudBatch.get(), m_smallFont.get(), m_ctrlFont.get(), commonCtrl, textPos, textColor);
            textPos.y += m_smallFont->GetLineSpacing();

            const wchar_t* controlLUT[] =
            {
                // None
                L"\0",
                // Simple
                L"[LB]/[LT] Increase/Decrease Color ESRAM Page Count   [RB]/[RT] Increase/Decrease Depth ESRAM Page Count",
                // Split
                L"[LB]/[LT] Increase/Decrease ESRAM Begin Address    [RB]/[RT] Increase/Decrease ESRAM End Address",
                // Metadata
                L"[B] Toggle Metadata ESRAM Mapping",
                // Random
                L"[LB]/[RB] Decrease/Increase ESRAM Page Probability"
            };

            DX::DrawControllerString(m_hudBatch.get(), m_smallFont.get(), m_ctrlFont.get(), controlLUT[m_mapScheme], textPos, textColor);

            m_hudBatch->End();
        }
    }

    // Show the new frame.
    {
        ScopedPixEvent(context, PIX_COLOR_DEFAULT, L"Present");

        m_deviceResources->Present();
        m_graphicsMemory->Commit();
    }
}
#pragma endregion

#pragma region Message Handlers
// Message handlers
void Sample::OnSuspending()
{
    auto context = m_deviceResources->GetD3DDeviceContext();
    context->Suspend(0);
}

void Sample::OnResuming()
{
    auto context = m_deviceResources->GetD3DDeviceContext();
    context->Resume();
    m_timer.ResetElapsedTime();
    m_gamePadButtons.Reset();
}
#pragma endregion

#pragma region Direct3D Resources
void Sample::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto context = m_deviceResources->GetD3DDeviceContext();

    m_profiler = std::make_unique<DX::GPUTimer>(device);

    m_graphicsMemory = std::make_unique<GraphicsMemory>(device, m_deviceResources->GetBackBufferCount());
    m_commonStates = std::make_unique<DirectX::CommonStates>(device);
    m_effectFactory = std::make_unique<DirectX::EffectFactory>(device);
        
    // Load models from disk.
    m_models.resize(_countof(s_modelPaths));
    for (int i = 0; i < m_models.size(); ++i)
    {
        m_models[i] = Model::CreateFromSDKMESH(device, s_modelPaths[i], *m_effectFactory, ModelLoader_CounterClockwise);
    }

    // HUD
    m_hudBatch = std::make_unique<SpriteBatch>(context);


    //-------------------------------------------------------
    // Instantiate objects from basic scene definition.

    m_scene.resize(_countof(s_sceneDefinition));
    for (int i = 0; i < m_scene.size(); i++)
    {
        size_t index = s_sceneDefinition[i].modelIndex;

        assert(index < m_models.size());
        auto& model = *m_models[index];

        m_scene[i].world = s_sceneDefinition[i].world;
        m_scene[i].model = &model;

        model.UpdateEffects([&](IEffect* e) 
        {
            static_cast<BasicEffect*>(e)->SetEmissiveColor(XMVectorSet(1.0f, 1.0f, 1.0f, 1.0f));
        });
    }


    //----------------------------------------
    // Create post process effects.

    // Create a full-screen triangle for full buffer pixel shader operations.
    m_fullScreenTri = GeometricPrimitive::CreateCustom(context, s_triVertex, s_triIndex);

    // Manual full screen clear - required to properly clear ESRAM-overlayed, compressed MSAA target
    m_manualClearEffect = std::make_unique<BasicEffect>(device);
    m_manualClearEffect->SetDiffuseColor(XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f)); // Disable lighting
    m_manualClearEffect->SetEmissiveColor(ATG::ColorsLinear::Background); // Set emissive

    // ESRAM color blend operation - Manipulate BasicEffect shader's math to perform direct, single-color blend.
    m_esramBlendEffect = std::make_unique<BasicEffect>(device);
    m_esramBlendEffect->SetDiffuseColor(XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f)); // Disable lighting
    m_esramBlendEffect->SetEmissiveColor(XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)); // Set emissive & alpha (direct color & alpha blend)
    m_esramBlendEffect->SetAlpha(0.25f);

    m_fullScreenTri->CreateInputLayout(m_manualClearEffect.get(), &m_inputLayout);
}

// Allocate all memory resources that change on a window SizeChanged event.
void Sample::CreateWindowSizeDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    const auto size = m_deviceResources->GetOutputSize();

    // Calculate display dimensions.
    m_displayWidth = size.right - size.left;
    m_displayHeight = size.bottom - size.top;

    // Set hud sprite viewport
    m_hudBatch->SetViewport(m_deviceResources->GetScreenViewport());

    // Set camera parameters.
    m_proj = XMMatrixPerspectiveFovLH(XM_PIDIV4, float(m_displayWidth) / float(m_displayHeight), 0.1f, 500.0f);

    // Begin uploading texture resources
    m_smallFont = std::make_unique<SpriteFont>(device, L"SegoeUI_18.spritefont");
    m_ctrlFont = std::make_unique<SpriteFont>(device, L"XboxOneControllerLegendSmall.spritefont");

    //------------------------------------------------
    // Step One: 
    // Create the resource descriptors as usual.

    // DRAM color target - ensure aligned to ESRAM page boundary.
    m_colorDesc = CD3D11_TEXTURE2D_DESC(
        c_colorFormat,                              // DXGI_FORMAT Format;
        UINT(m_displayWidth),       // UINT width;
        UINT(m_displayHeight),      // UINT height;
        1,                          // UINT arraySize;
        1,                          // UINT mipLevels;
        D3D11_BIND_RENDER_TARGET,   // UINT bindFlags;
        D3D11_USAGE_DEFAULT,        // D3D11_USAGE usage;
        0,                          // UINT cpuaccessFlags;
        2,                          // UINT sampleCount;
        0,                          // UINT sampleQuality;
        0,                          // UINT miscFlags;
        0,                          // UINT esramOffsetBytes;
        0                           // UINT esramUsageBytes;
    );

    // DRAM depth target - ensure aligned to ESRAM page boundary.
    m_depthDesc = CD3D11_TEXTURE2D_DESC(
        c_depthFormat,              // DXGI_FORMAT Format;
        UINT(m_displayWidth),       // UINT width;
        UINT(m_displayHeight),      // UINT height;
        1,                          // UINT arraySize;
        1,                          // UINT mipLevels;
        D3D11_BIND_DEPTH_STENCIL,   // UINT bindFlags;
        D3D11_USAGE_DEFAULT,        // D3D11_USAGE usage;
        0,                          // UINT cpuaccessFlags;
        2,                          // UINT sampleCount;
        0,                          // UINT sampleQuality;
        0,                          // UINT miscFlags;
        0,                          // UINT esramOffsetBytes;
        0                           // UINT esramUsageBytes;
    );

    // Create a resource descriptor that fills all 32 MB of ESRAM.
    m_esramOverlayDesc = CD3D11_TEXTURE2D_DESC(
        c_colorFormat,              // DXGI_FORMAT Format;
        c_esramTexWidth,            // UINT width;
        c_esramTexHeight,           // UINT height;
        1,                          // UINT arraySize;
        1,                          // UINT mipLevels;
        D3D11_BIND_RENDER_TARGET    // UINT bindFlags;
    );


    //-----------------------------------------------------------------------------------------------
    // Step Two: 
    // Calculate resource page counts via the resource descriptors and XG library functions.

    // Calculate total number of pages for the offscreen color & depth buffers for their respective format/dimensions/layout.
    m_colorPageCount = CalculatePagesForColorResource(m_colorDesc, c_colorXGFormat, &m_colorLayoutDesc);
    m_depthPageCount = CalculatePagesForDepthResource(m_depthDesc, c_depthXGFormat, &m_depthLayoutDesc);
    m_esramOverlayPageCount = CalculatePagesForColorResource(m_esramOverlayDesc, c_colorXGFormat);

    // Determine initial number of pages to map to ESRAM - map all of color, then map depth to remaining pages.
    m_colorEsramPageCount = std::min(m_colorPageCount, c_esramPageCount);
    m_depthEsramPageCount = std::min(m_depthPageCount, c_esramPageCount - m_colorEsramPageCount);

    // Initialize the XGMemoryEngine with maximum page counts.
    // Note that even on Durango we allocate enough system pages to hold our color & depth targets,
    // since the sample's parameterized mapping schemes allow targets to be pushed completely to DRAM.
    UINT32 numEsramPages = SupportsESRAM() ? c_esramPageCount : 0;
    UINT32 numSystemPages = UINT32(m_colorPageCount + m_depthPageCount);

    DX::ThrowIfFailed(m_layoutEngine.InitializeWithPageCounts(numSystemPages, numEsramPages, 0));


    // Step Three & Four (inside): Create the resources using the current mapping scheme.
    UpdateResourceMappings();
}

void Sample::UpdateResourceMappings()
{
    //----------------------------------------------------------------------
    // Step Three: 
    // Create the memory mapping using the XGMemory library.
    // 
    // The XGMemoryLayoutEngine is simply used to create XGMemoryLayouts, intiailized with page counts using 
    // InitializeWithPageCounts() or InitializeWithPageArrays(). Multiple XGMemoryLayoutEngines can be used with
    // different page counts.
    // 
    // The XGMemoryLayout API represents a coherent, stateful mapping of system & ESRAM memory pages.
    // It maintains a free list of ESRAM pages which it allocates when mapping requests are made via MapSimple(), 
    // MapRunLengthArray(), MapPagePriorityArray(), and MapFromPattern(). These pages can be returned to the free list
    // with RelinquishMapping(), which allows subsequent resources to reuse those freed pages. Consequently, consider 
    // the relinquish of a mapping as a barrier where use of the memory in the relinquished mapping must strictly precede 
    // the use of any of the memory in future mappings.
    //
    // The XGMemoryLayoutMapping is a simple POD structure that represents a single mapping within the memory layout.
    // It's created by the XGMemoryLayout::CreateMapping(...) function, and then provided as an argument in the 
    // mapping/relinquish functions.

    if (m_layout == nullptr)
    {
        int totalPageCount = m_colorPageCount + m_depthPageCount + m_esramOverlayPageCount;
        DX::ThrowIfFailed(m_layoutEngine.CreateMemoryLayout(L"Default Layout", totalPageCount * c_pageSize, 0, m_layout.ReleaseAndGetAddressOf()));
    }
    else
    {
        ID3D11DeviceX* device = m_deviceResources->GetD3DDevice();
        ID3D11DeviceContextX* context = m_deviceResources->GetD3DDeviceContext();

        UINT64 fenceValue = context->InsertFence();
        while (device->IsFencePending(fenceValue))
        {
            SwitchToThread();
        }

        m_layout->Reset();
    }

    //---------------------------------------------------
    // Determine the ESRAM/system page mapping scheme for our resources by mapping scheme.

    XGMemoryLayoutMapping colorMapping;
    XGMemoryLayoutMapping depthMapping;

    // Helper that maps all pages to DRAM.
    auto mapAllToDram = [&] () 
    {
        // Map all pages to DRAM using XGMemoryLayout::MapSimple(...).
        // This technique can be used on Xbox One X to maintain consistent API usage without
        // causing runtime crashes.
        DX::ThrowIfFailed(m_layout->CreateMapping(L"Color", &colorMapping));
        DX::ThrowIfFailed(m_layout->MapSimple(&colorMapping, UINT32(m_colorPageCount), 0));

        DX::ThrowIfFailed(m_layout->CreateMapping(L"Depth", &depthMapping));
        DX::ThrowIfFailed(m_layout->MapSimple(&depthMapping, UINT32(m_depthPageCount), 0));
    };

    switch (m_mapScheme)
    {
    case EMS_None: 
        mapAllToDram();
        break;

    case EMS_Simple:
        // 'Simple' mapping makes use of the XGMemoryLayout::MapSimple(...) function.
        //
        // This facility simply maps a number of contiguous pages to ESRAM followed by the
        // remaining pages to system memory.
        //
        // The benefit to this method is it requires minimal work by the developer, just a 
        // single function call, but provides no custom behavior.
        assert((m_colorEsramPageCount + m_depthEsramPageCount) <= c_esramPageCount);

        DX::ThrowIfFailed(m_layout->CreateMapping(L"Color", &colorMapping));
        DX::ThrowIfFailed(m_layout->MapSimple(&colorMapping, UINT32(m_colorPageCount), UINT32(m_colorEsramPageCount)));

        DX::ThrowIfFailed(m_layout->CreateMapping(L"Depth", &depthMapping));
        DX::ThrowIfFailed(m_layout->MapSimple(&depthMapping, UINT32(m_depthPageCount), UINT32(m_depthEsramPageCount)));
        break;

    case EMS_Split:
        // 'Split' mapping maps the virtual address range of the render target into three sections:
        // 1. A DRAM-mapped bottom section (least significant address)
        // 2. An ESRAM-mapped middle section
        // 3. A DRAM-mapped top section (most significant address)
        //
        // Split mapping can be beneficial for render targets that sport both low- and high-utilized regions. 
        // For instance, during outdoor scene rendering the skybox commonly populates the top portion of the frame. 
        // This results in only one draw call for those pixels, which benefits less from ESRAM residency than the rest 
        // of the image which can incur significantly more overdraw.
        //
        // This can be accomplished using the XGMemoryLayout::MapRunLengthArray(...) function. This uses an array 
        // of USHORT 'tokens' specifying the number of contiguous pages of memory to map to either system or ESRAM.
        // The high bit of the token is used to distinguish between ESRAM & system mapped memory. The helper macros 
        // XGSystemToken(...) & XGEsramToken(...) mask this implementation nuance.

        // Check that the user has specified some percentage of the mapping goes to ESRAM
        if ((m_topPercent - m_bottomPercent) > 1e-2f)
        {
            // Helper function that generates the split mapping.
            auto createSplitMapping = [&](int resourcePageCount, const wchar_t* mappingName, XGMemoryLayoutMapping& mapping)
            {
                // Determine how many pages to map for each of the three sections by calculating
                // page counts from the start & stop percentages.
                int dramBeginCount = int(m_bottomPercent * resourcePageCount);
                int dramEndCount = int((1.0f - m_topPercent) * resourcePageCount);
                int esramCount = resourcePageCount - (dramBeginCount + dramEndCount);

                USHORT RLETokens[] =
                {
                    XGSystemToken(dramBeginCount),
                    XGEsramToken(esramCount),
                    XGSystemToken(dramEndCount),
                };

                DX::ThrowIfFailed(m_layout->CreateMapping(mappingName, &mapping));
                DX::ThrowIfFailed(m_layout->MapRunLengthArray(&mapping, RLETokens, _countof(RLETokens), UINT32(resourcePageCount), UINT32(XG_ALL_REMAINING_PAGES)));
            };

            createSplitMapping(m_colorPageCount, L"Color", colorMapping);
            createSplitMapping(m_depthPageCount, L"Depth", depthMapping);
        }
        else
        {
            mapAllToDram();
        }
        break;

    case EMS_Metadata:
        // 'Metadata' mapping will selectively map only the resource metadata to ESRAM. 
        // 
        // This is achieved by calculating the number of metadata pages necessary for each resource by inspecting
        // the resource layout from the XGTextureAddressCompute::GetResourceLayout(...) result. We build a small
        // resource layout descriptor on creation and use that to plan our page mapping.

        if (m_metadataEnabled)
        {
            auto createMetadataMapping = [&](const wchar_t* name, int totalPageCount, MetadataDesc& desc, XGMemoryLayoutMapping& mapping)
            {
                int count = 0;
                USHORT RLETokens[PlaneCount] = {};

                int currPage = 0;
                int i = 0;

                // Iterate through the resource pages until we hit the end of the metadata ranges.
                while (currPage < desc.End())
                {
                    if (currPage == desc.ranges[i].start)
                    {
                        // We're at the start of a metadata range -- map the range to ESRAM memory.
                        RLETokens[count++] = XGEsramToken(desc.ranges[i].count);
                        currPage = desc.ranges[i].End();
                        ++i;
                    }
                    else
                    {
                        // We're within a gap of the ranges -- map to system memory.
                        RLETokens[count++] = XGSystemToken(desc.ranges[i].start - currPage);
                        currPage = desc.ranges[i].start;
                    }
                }

                // Map the remaining pages to system ram (if any.)
                RLETokens[count++] = XGSystemToken(totalPageCount - currPage);

                DX::ThrowIfFailed(m_layout->CreateMapping(name, &mapping));
                DX::ThrowIfFailed(m_layout->MapRunLengthArray(&mapping, RLETokens, count, UINT32(totalPageCount), UINT32(XG_ALL_REMAINING_PAGES)));
            };

            createMetadataMapping(L"Color", m_colorPageCount, m_colorLayoutDesc, colorMapping);
            createMetadataMapping(L"Depth", m_depthPageCount, m_depthLayoutDesc, depthMapping);
        }
        else
        {
            mapAllToDram();
        }
        break;

    case EMS_Random:
        // 'Random' mapping performs a random check over each page against a user-specified 
        // probability to determine whether that page will be mapped to system or ESRAM memory.
        //
        // There's not a specific benefit to this mapping scheme -- simply shows off how to utilize 
        // the provided API in a unique fashion.
        //
        // This mapping method also makes use of the XGMemoryLayout::MapRunLengthArray(...) function.

        // Helper function that generates the random mapping.
        auto createRandomMapping = [&](int resourcePageCount, const wchar_t* mappingName, XGMemoryLayoutMapping& mapping)
        {
            // Each token specifies DRAM or ESRAM mapping and a count of pages to map.
            USHORT RLETokens[c_esramPageCount] = {};
            int currTokenIndex = 0;
            int currTokenCount = 0;

            std::uniform_real_distribution<float> ranDist;

            // Iterate over each page and determine whether to map it to DRAM or ESRAM.
            for (int i = 0; i < resourcePageCount; ++i)
            {
                bool isEsramToken = currTokenIndex % 2 == 0;              // Map even tokens to ESRAM - arbitrary decision.
                bool toEsram = ranDist(m_generator) < m_esramProbability; // Random roll with uniform distribution.

                // Determine if we should stop the current run of DRAM or ESRAM pages.
                if (isEsramToken != toEsram)
                {
                    RLETokens[currTokenIndex++] = isEsramToken ? XGEsramToken(currTokenCount) : XGSystemToken(currTokenCount);
                    currTokenCount = 0;
                }

                ++currTokenCount;
            }
            RLETokens[currTokenIndex++] = currTokenIndex % 2 == 0 ? XGEsramToken(currTokenCount) : XGSystemToken(currTokenCount);

            DX::ThrowIfFailed(m_layout->CreateMapping(mappingName, &mapping));
            DX::ThrowIfFailed(m_layout->MapRunLengthArray(&mapping, RLETokens, UINT32(currTokenIndex), UINT32(resourcePageCount), UINT32(XG_ALL_REMAINING_PAGES)));
        };

        createRandomMapping(m_colorPageCount, L"Color", colorMapping);
        createRandomMapping(m_depthPageCount, L"Depth", depthMapping);

        break;
    }

    //-------------------------------------------------------
    // Step Four: 
    // Create the resources using 'ID3D11DeviceX::CreatePlacedResourceX(...)' with the base virtual memory address.

    auto device = m_deviceResources->GetD3DDevice();
    
    CreateColorResourceAndView(device, m_colorDesc, &colorMapping.MappingBaseAddress, L"Color Texture", m_colorTexture.ReleaseAndGetAddressOf(), m_colorRTV.ReleaseAndGetAddressOf());
    CreateDepthResourceAndView(device, m_depthDesc, &depthMapping.MappingBaseAddress, L"Depth Texture", m_depthTexture.ReleaseAndGetAddressOf(), m_depthDSV.ReleaseAndGetAddressOf());


    //--------------------------------------------------------------
    // For visualization - Create color target that fills all 32 MB of ESRAM.
    //
    // This is accomplished by relinquishing the previously mapped memory for our color & depth target,
    // releasing their ESRAM and system pages back to the free list. We then take the opportunity
    // to request all of the ESRAM pages for our overlay render target, aliasing all resources that lived
    // there.

    if (SupportsESRAM())
    {
        DX::ThrowIfFailed(m_layout->RelinquishMapping(&depthMapping, 0));
        DX::ThrowIfFailed(m_layout->RelinquishMapping(&colorMapping, 0));

        XGMemoryLayoutMapping esramOverlayMapping;
        m_layout->CreateMapping(L"Overlay", &esramOverlayMapping);
        m_layout->MapSimple(&esramOverlayMapping, m_esramOverlayPageCount, c_esramPageCount);

        CreateColorResourceAndView(device, m_esramOverlayDesc, &esramOverlayMapping.MappingBaseAddress, L"Full Cover ESRAM", m_esramOverlayTexture.ReleaseAndGetAddressOf(), m_esramOverlayRTV.ReleaseAndGetAddressOf());
    }
}
#pragma endregion
