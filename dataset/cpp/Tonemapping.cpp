// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Renderer/Tonemapping.h>
#include <AnKi/Renderer/DownscaleBlur.h>
#include <AnKi/Renderer/Renderer.h>
#include <AnKi/Util/Tracer.h>

namespace anki {

Error Tonemapping::init()
{
	const Error err = initInternal();
	if(err)
	{
		ANKI_R_LOGE("Failed to initialize tonemapping");
	}

	return err;
}

Error Tonemapping::initInternal()
{
	m_inputTexMip = getRenderer().getDownscaleBlur().getMipmapCount() - 2;
	const U32 width = getRenderer().getDownscaleBlur().getPassWidth(m_inputTexMip);
	const U32 height = getRenderer().getDownscaleBlur().getPassHeight(m_inputTexMip);

	ANKI_R_LOGV("Initializing tonemapping. Resolution %ux%u", width, height);

	// Create program
	ANKI_CHECK(ResourceManager::getSingleton().loadResource("ShaderBinaries/TonemappingAverageLuminance.ankiprogbin", m_prog));

	ShaderProgramResourceVariantInitInfo variantInitInfo(m_prog);
	variantInitInfo.addConstant("kInputTexSize", UVec2(width, height));

	const ShaderProgramResourceVariant* variant;
	m_prog->getOrCreateVariant(variantInitInfo, variant);
	m_grProg.reset(&variant->getProgram());

	// Create exposure texture.
	// WARNING: Use it only as IMAGE and nothing else. It will not be tracked by the rendergraph. No tracking means no
	// automatic image transitions
	const TextureUsageBit usage = TextureUsageBit::kAllUav;
	const TextureInitInfo texinit = getRenderer().create2DRenderTargetInitInfo(1, 1, Format::kR16G16_Sfloat, usage, "ExposureAndAvgLum1x1");
	ClearValue clearValue;
	clearValue.m_colorf = {0.5f, 0.5f, 0.5f, 0.5f};
	m_exposureAndAvgLuminance1x1 = getRenderer().createAndClearRenderTarget(texinit, TextureUsageBit::kAllUav, clearValue);

	return Error::kNone;
}

void Tonemapping::importRenderTargets(RenderingContext& ctx)
{
	// Just import it. It will not be used in resource tracking
	m_runCtx.m_exposureLuminanceHandle = ctx.m_renderGraphDescr.importRenderTarget(m_exposureAndAvgLuminance1x1.get(), TextureUsageBit::kAllUav);
}

void Tonemapping::populateRenderGraph(RenderingContext& ctx)
{
	ANKI_TRACE_SCOPED_EVENT(Tonemapping);
	RenderGraphDescription& rgraph = ctx.m_renderGraphDescr;

	// Create the pass
	ComputeRenderPassDescription& pass = rgraph.newComputeRenderPass("AvgLuminance");

	pass.setWork([this](RenderPassWorkContext& rgraphCtx) {
		ANKI_TRACE_SCOPED_EVENT(Tonemapping);
		CommandBuffer& cmdb = *rgraphCtx.m_commandBuffer;

		cmdb.bindShaderProgram(m_grProg.get());
		rgraphCtx.bindUavTexture(0, 1, m_runCtx.m_exposureLuminanceHandle);

		TextureSubresourceInfo inputTexSubresource;
		inputTexSubresource.m_firstMipmap = m_inputTexMip;
		rgraphCtx.bindTexture(0, 0, getRenderer().getDownscaleBlur().getRt(), inputTexSubresource);

		cmdb.dispatchCompute(1, 1, 1);
	});

	TextureSubresourceInfo inputTexSubresource;
	inputTexSubresource.m_firstMipmap = m_inputTexMip;
	pass.newTextureDependency(getRenderer().getDownscaleBlur().getRt(), TextureUsageBit::kSampledCompute, inputTexSubresource);
}

} // end namespace anki
