// Copyright (c) 2016 Juan Delgado (JuDelCo)
// License: MIT License
// MIT License web page: https://opensource.org/licenses/MIT

#pragma once

namespace EntitasPP
{
class IComponent
{
	friend class Entity;

	protected:
		IComponent() = default;
};
}
