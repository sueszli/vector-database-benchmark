// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Graphics module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#include <Nazara/Graphics/ElementRendererRegistry.hpp>
#include <Nazara/Graphics/Enums.hpp>
#include <Nazara/Graphics/Graphics.hpp>
#include <Nazara/Graphics/RenderSpriteChain.hpp>
#include <Nazara/Graphics/RenderSubmesh.hpp>
#include <Nazara/Graphics/SpriteChainRenderer.hpp>
#include <Nazara/Graphics/SubmeshRenderer.hpp>
#include <Nazara/Graphics/Debug.hpp>

namespace Nz
{
	ElementRendererRegistry::ElementRendererRegistry()
	{
		RegisterElementRenderer<RenderSpriteChain>(std::make_unique<SpriteChainRenderer>(*Graphics::Instance()->GetRenderDevice()));
		RegisterElementRenderer<RenderSubmesh>(std::make_unique<SubmeshRenderer>());
	}
}
