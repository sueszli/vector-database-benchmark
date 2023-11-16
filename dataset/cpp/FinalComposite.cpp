// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Renderer/FinalComposite.h>
#include <AnKi/Renderer/Renderer.h>
#include <AnKi/Renderer/Bloom.h>
#include <AnKi/Renderer/Scale.h>
#include <AnKi/Renderer/Tonemapping.h>
#include <AnKi/Renderer/LightShading.h>
#include <AnKi/Renderer/GBuffer.h>
#include <AnKi/Renderer/Dbg.h>
#include <AnKi/Renderer/DownscaleBlur.h>
#include <AnKi/Renderer/UiStage.h>
#include <AnKi/Renderer/MotionVectors.h>
#include <AnKi/Util/Logger.h>
#include <AnKi/Util/Tracer.h>
#include <AnKi/Core/CVarSet.h>

namespace anki {

static NumericCVar<U32> g_motionBlurSamplesCVar(CVarSubsystem::kRenderer, "MotionBlurSamples", 32, 0, 2048, "Max motion blur samples");
static NumericCVar<F32> g_filmGrainStrengthCVar(CVarSubsystem::kRenderer, "FilmGrainStrength", 16.0f, 0.0f, 250.0f, "Film grain strength");

Error FinalComposite::initInternal()
{
	ANKI_R_LOGV("Initializing final composite");

	ANKI_CHECK(loadColorGradingTextureImage("EngineAssets/DefaultLut.ankitex"));

	m_fbDescr.m_colorAttachmentCount = 1;
	m_fbDescr.bake();

	// Progs
	ANKI_CHECK(ResourceManager::getSingleton().loadResource("ShaderBinaries/FinalComposite.ankiprogbin", m_prog));

	ShaderProgramResourceVariantInitInfo variantInitInfo(m_prog);
	variantInitInfo.addMutation("FILM_GRAIN", (g_filmGrainStrengthCVar.get() > 0.0) ? 1 : 0);
	variantInitInfo.addMutation("BLOOM_ENABLED", 1);
	variantInitInfo.addConstant("kLutSize", U32(kLutSize));
	variantInitInfo.addConstant("kFramebufferSize", getRenderer().getPostProcessResolution());
	variantInitInfo.addConstant("kMotionBlurSamples", g_motionBlurSamplesCVar.get());

	for(U32 dbg = 0; dbg < 2; ++dbg)
	{
		const ShaderProgramResourceVariant* variant;
		variantInitInfo.addMutation("DBG_ENABLED", dbg);
		m_prog->getOrCreateVariant(variantInitInfo, variant);
		m_grProgs[dbg].reset(&variant->getProgram());
	}

	ANKI_CHECK(ResourceManager::getSingleton().loadResource("ShaderBinaries/VisualizeRenderTarget.ankiprogbin", m_defaultVisualizeRenderTargetProg));
	const ShaderProgramResourceVariant* variant;
	m_defaultVisualizeRenderTargetProg->getOrCreateVariant(variant);
	m_defaultVisualizeRenderTargetGrProg.reset(&variant->getProgram());

	return Error::kNone;
}

Error FinalComposite::init()
{
	const Error err = initInternal();
	if(err)
	{
		ANKI_R_LOGE("Failed to init final composite");
	}

	return err;
}

Error FinalComposite::loadColorGradingTextureImage(CString filename)
{
	m_lut.reset(nullptr);
	ANKI_CHECK(ResourceManager::getSingleton().loadResource(filename, m_lut));
	ANKI_ASSERT(m_lut->getWidth() == kLutSize);
	ANKI_ASSERT(m_lut->getHeight() == kLutSize);
	ANKI_ASSERT(m_lut->getDepth() == kLutSize);

	return Error::kNone;
}

void FinalComposite::populateRenderGraph(RenderingContext& ctx)
{
	ANKI_TRACE_SCOPED_EVENT(RFinalComposite);

	RenderGraphDescription& rgraph = ctx.m_renderGraphDescr;

	// Create the pass
	GraphicsRenderPassDescription& pass = rgraph.newGraphicsRenderPass("Final Composite");

	pass.setWork(1, [this](RenderPassWorkContext& rgraphCtx) {
		run(rgraphCtx);
	});
	pass.setFramebufferInfo(m_fbDescr, {ctx.m_outRenderTarget});

	pass.newTextureDependency(ctx.m_outRenderTarget, TextureUsageBit::kFramebufferWrite);

	if(g_dbgCVar.get())
	{
		pass.newTextureDependency(getRenderer().getDbg().getRt(), TextureUsageBit::kSampledFragment);
	}

	pass.newTextureDependency(getRenderer().getScale().getTonemappedRt(), TextureUsageBit::kSampledFragment);
	pass.newTextureDependency(getRenderer().getBloom().getRt(), TextureUsageBit::kSampledFragment);
	pass.newTextureDependency(getRenderer().getMotionVectors().getMotionVectorsRt(), TextureUsageBit::kSampledFragment);
	pass.newTextureDependency(getRenderer().getGBuffer().getDepthRt(), TextureUsageBit::kSampledFragment);

	Array<RenderTargetHandle, kMaxDebugRenderTargets> dbgRts;
	ShaderProgramPtr debugProgram;
	const Bool hasDebugRt = getRenderer().getCurrentDebugRenderTarget(dbgRts, debugProgram);
	if(hasDebugRt)
	{
		for(const RenderTargetHandle& handle : dbgRts)
		{
			if(handle.isValid())
			{
				pass.newTextureDependency(handle, TextureUsageBit::kSampledFragment);
			}
		}
	}
}

void FinalComposite::run(RenderPassWorkContext& rgraphCtx)
{
	ANKI_TRACE_SCOPED_EVENT(FinalComposite);

	CommandBuffer& cmdb = *rgraphCtx.m_commandBuffer;
	const Bool dbgEnabled = g_dbgCVar.get();

	Array<RenderTargetHandle, kMaxDebugRenderTargets> dbgRts;
	ShaderProgramPtr optionalDebugProgram;
	const Bool hasDebugRt = getRenderer().getCurrentDebugRenderTarget(dbgRts, optionalDebugProgram);

	// Bind program
	if(hasDebugRt && optionalDebugProgram.isCreated())
	{
		cmdb.bindShaderProgram(optionalDebugProgram.get());
	}
	else if(hasDebugRt)
	{
		cmdb.bindShaderProgram(m_defaultVisualizeRenderTargetGrProg.get());
	}
	else
	{
		cmdb.bindShaderProgram(m_grProgs[dbgEnabled].get());
	}

	// Bind stuff
	if(!hasDebugRt)
	{
		cmdb.bindSampler(0, 0, getRenderer().getSamplers().m_nearestNearestClamp.get());
		cmdb.bindSampler(0, 1, getRenderer().getSamplers().m_trilinearClamp.get());
		cmdb.bindSampler(0, 2, getRenderer().getSamplers().m_trilinearRepeat.get());

		rgraphCtx.bindColorTexture(0, 3, getRenderer().getScale().getTonemappedRt());

		rgraphCtx.bindColorTexture(0, 4, getRenderer().getBloom().getRt());
		cmdb.bindTexture(0, 5, &m_lut->getTextureView());
		rgraphCtx.bindColorTexture(0, 6, getRenderer().getMotionVectors().getMotionVectorsRt());
		rgraphCtx.bindTexture(0, 7, getRenderer().getGBuffer().getDepthRt(), TextureSubresourceInfo(DepthStencilAspectBit::kDepth));

		if(dbgEnabled)
		{
			rgraphCtx.bindColorTexture(0, 8, getRenderer().getDbg().getRt());
		}

		if(g_filmGrainStrengthCVar.get() > 0.0f)
		{
			const UVec4 pc(0, 0, floatBitsToUint(g_filmGrainStrengthCVar.get()), getRenderer().getFrameCount() & kMaxU32);
			cmdb.setPushConstants(&pc, sizeof(pc));
		}
	}
	else
	{
		cmdb.bindSampler(0, 0, getRenderer().getSamplers().m_nearestNearestClamp.get());

		U32 count = 1;
		for(const RenderTargetHandle& handle : dbgRts)
		{
			if(handle.isValid())
			{
				rgraphCtx.bindColorTexture(0, count++, handle);
			}
		}
	}

	cmdb.setViewport(0, 0, getRenderer().getPostProcessResolution().x(), getRenderer().getPostProcessResolution().y());
	drawQuad(cmdb);

	// Draw UI
	getRenderer().getUiStage().draw(getRenderer().getPostProcessResolution().x(), getRenderer().getPostProcessResolution().y(), cmdb);
}

} // end namespace anki
