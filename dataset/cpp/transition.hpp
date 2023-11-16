/******************************************************************************
    Copyright (C) 2016-2019 by Streamlabs (General Workings Inc)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once
#include <napi.h>
#include "input.hpp"
#include "isource.hpp"
#include "scene.hpp"
#include "utility-v8.hpp"

namespace osn {
class Transition : public Napi::ObjectWrap<osn::Transition> {
public:
	uint64_t sourceId;

public:
	static Napi::FunctionReference constructor;
	static Napi::Object Init(Napi::Env env, Napi::Object exports);
	Transition(const Napi::CallbackInfo &info);

	static Napi::Value Types(const Napi::CallbackInfo &info);
	static Napi::Value Create(const Napi::CallbackInfo &info);
	static Napi::Value CreatePrivate(const Napi::CallbackInfo &info);
	static Napi::Value FromName(const Napi::CallbackInfo &info);

	Napi::Value GetActiveSource(const Napi::CallbackInfo &info);
	Napi::Value Clear(const Napi::CallbackInfo &info);
	Napi::Value Set(const Napi::CallbackInfo &info);
	Napi::Value Start(const Napi::CallbackInfo &info);

	Napi::Value CallIsConfigurable(const Napi::CallbackInfo &info);
	Napi::Value CallGetProperties(const Napi::CallbackInfo &info);
	Napi::Value CallGetSettings(const Napi::CallbackInfo &info);
	Napi::Value CallGetSlowUncachedSettings(const Napi::CallbackInfo &info);

	Napi::Value CallGetType(const Napi::CallbackInfo &info);
	Napi::Value CallGetName(const Napi::CallbackInfo &info);
	void CallSetName(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value CallGetOutputFlags(const Napi::CallbackInfo &info);
	Napi::Value CallGetFlags(const Napi::CallbackInfo &info);
	void CallSetFlags(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value CallGetStatus(const Napi::CallbackInfo &info);
	Napi::Value CallGetId(const Napi::CallbackInfo &info);
	Napi::Value CallGetMuted(const Napi::CallbackInfo &info);
	void CallSetMuted(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value CallGetEnabled(const Napi::CallbackInfo &info);
	void CallSetEnabled(const Napi::CallbackInfo &info, const Napi::Value &value);

	Napi::Value CallRelease(const Napi::CallbackInfo &info);
	Napi::Value CallRemove(const Napi::CallbackInfo &info);
	Napi::Value CallUpdate(const Napi::CallbackInfo &info);
	Napi::Value CallLoad(const Napi::CallbackInfo &info);
	Napi::Value CallSave(const Napi::CallbackInfo &info);

	Napi::Value CallSendMouseClick(const Napi::CallbackInfo &info);
	Napi::Value CallSendMouseMove(const Napi::CallbackInfo &info);
	Napi::Value CallSendMouseWheel(const Napi::CallbackInfo &info);
	Napi::Value CallSendFocus(const Napi::CallbackInfo &info);
	Napi::Value CallSendKeyClick(const Napi::CallbackInfo &info);

	Napi::Value CallCallHandler(const Napi::CallbackInfo &info);
};
}
