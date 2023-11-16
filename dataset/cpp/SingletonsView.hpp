#pragma once

#include "CmptPtr.hpp"

#include <span>

namespace Ubpa::UECS {
	class SingletonsView {
	public:
		SingletonsView(std::span<const CmptAccessPtr> singletons) noexcept
			: singletons{ singletons } {}

		CmptAccessPtr AccessSingleton(AccessTypeID) const noexcept;
		std::span<const CmptAccessPtr> Singletons() const noexcept { return singletons; }
	private:
		std::span<const CmptAccessPtr> singletons;
	};
}
