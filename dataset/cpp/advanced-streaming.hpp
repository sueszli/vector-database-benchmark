/******************************************************************************
    Copyright (C) 2016-2022 by Streamlabs (General Workings Inc)

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
#include "streaming.hpp"

namespace osn {
class AdvancedStreaming : public Napi::ObjectWrap<osn::AdvancedStreaming>, public osn::Streaming {
public:
	static Napi::FunctionReference constructor;
	static Napi::Object Init(Napi::Env env, Napi::Object exports);
	AdvancedStreaming(const Napi::CallbackInfo &info);

	static Napi::Value Create(const Napi::CallbackInfo &info);
	static void Destroy(const Napi::CallbackInfo &info);

	Napi::Value GetAudioTrack(const Napi::CallbackInfo &info);
	void SetAudioTrack(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value GetTwitchTrack(const Napi::CallbackInfo &info);
	void SetTwitchTrack(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value GetRescaling(const Napi::CallbackInfo &info);
	void SetRescaling(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value GetOutputWidth(const Napi::CallbackInfo &info);
	void SetOutputWidth(const Napi::CallbackInfo &info, const Napi::Value &value);
	Napi::Value GetOutputHeight(const Napi::CallbackInfo &info);
	void SetOutputHeight(const Napi::CallbackInfo &info, const Napi::Value &value);

	static Napi::Value GetLegacySettings(const Napi::CallbackInfo &info);
	static void SetLegacySettings(const Napi::CallbackInfo &info, const Napi::Value &value);
};
}