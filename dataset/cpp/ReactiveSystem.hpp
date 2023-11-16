// Copyright (c) 2016 Juan Delgado (JuDelCo)
// License: MIT License
// MIT License web page: https://opensource.org/licenses/MIT

#pragma once

#include "GroupObserver.hpp"
#include "ISystem.hpp"

namespace EntitasPP
{
class ReactiveSystem : public IExecuteSystem
{
	public:
		ReactiveSystem(Pool* pool, std::shared_ptr<IReactiveSystem> subsystem);
		ReactiveSystem(Pool* pool, std::shared_ptr<IMultiReactiveSystem> subsystem);
		ReactiveSystem(Pool* pool, std::shared_ptr<IReactiveExecuteSystem> subsystem, std::vector<TriggerOnEvent> triggers);
		~ReactiveSystem();

		auto GetSubsystem() const -> std::shared_ptr<IReactiveExecuteSystem>;
		void Activate();
		void Deactivate();
		void Clear();
		void Execute();

	private:
		std::shared_ptr<IReactiveExecuteSystem> mSubsystem;
		GroupObserver* mObserver;
		Matcher mEnsureComponents;
		Matcher mExcludeComponents;
		bool mClearAfterExecute{false};
		std::vector<EntityPtr> mEntityBuffer;
};
}
