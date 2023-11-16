// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Core module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#pragma once

#ifndef NAZARA_CORE_COMPONENTS_DISABLEDCOMPONENT_HPP
#define NAZARA_CORE_COMPONENTS_DISABLEDCOMPONENT_HPP

#include <NazaraUtils/Prerequisites.hpp>
#include <Nazara/Core/Time.hpp>
#include <Nazara/Utility/Config.hpp>

namespace Nz
{
	class DisabledComponent
	{
		public:
			DisabledComponent() = default;
			DisabledComponent(const DisabledComponent&) = default;
			DisabledComponent(DisabledComponent&&) = default;
			~DisabledComponent() = default;

			DisabledComponent& operator=(const DisabledComponent&) = default;
			DisabledComponent& operator=(DisabledComponent&&) = default;
	};
}

#endif // NAZARA_CORE_COMPONENTS_DISABLEDCOMPONENT_HPP
