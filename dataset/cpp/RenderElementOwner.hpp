// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Graphics module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#pragma once

#ifndef NAZARA_GRAPHICS_RENDERELEMENTOWNER_HPP
#define NAZARA_GRAPHICS_RENDERELEMENTOWNER_HPP

#include <NazaraUtils/Prerequisites.hpp>
#include <Nazara/Graphics/Config.hpp>
#include <NazaraUtils/MovablePtr.hpp>

namespace Nz
{
	class RenderElement;
	class RenderElementPoolBase;

	class NAZARA_GRAPHICS_API RenderElementOwner
	{
		public:
			inline RenderElementOwner(RenderElementPoolBase* pool, std::size_t poolIndex, RenderElement* element);
			RenderElementOwner(const RenderElementOwner&) = delete;
			RenderElementOwner(RenderElementOwner&&) noexcept = default;
			~RenderElementOwner();

			inline RenderElement* GetElement();
			inline const RenderElement* GetElement() const;

			inline RenderElement* operator->();
			inline const RenderElement* operator->() const;

			RenderElementOwner& operator=(const RenderElementOwner&) = delete;
			RenderElementOwner& operator=(RenderElementOwner&&) noexcept = default;

		private:
			std::size_t m_poolIndex;
			MovablePtr<RenderElement> m_element;
			MovablePtr<RenderElementPoolBase> m_pool;
	};
}

#include <Nazara/Graphics/RenderElementOwner.inl>

#endif // NAZARA_GRAPHICS_RENDERELEMENTOWNER_HPP
