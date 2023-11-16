// Copyright (c) 2016 Juan Delgado (JuDelCo)
// License: MIT License
// MIT License web page: https://opensource.org/licenses/MIT

#pragma once

#include "Matcher.hpp"
#include "GroupEventType.hpp"

namespace EntitasPP
{
struct TriggerOnEvent
{
	public:
		TriggerOnEvent(const Matcher trigger, const GroupEventType eventType)
		{
			this->trigger = trigger;
			this->eventType = eventType;
		}

		Matcher trigger;
		GroupEventType eventType;
};
}
