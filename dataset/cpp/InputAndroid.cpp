// Copyright (C) 2009-2023, Panagiotis Christopoulos Charitos and contributors.
// All rights reserved.
// Code licensed under the BSD License.
// http://www.anki3d.org/LICENSE

#include <AnKi/Window/InputAndroid.h>
#include <AnKi/Window/NativeWindowAndroid.h>
#include <AnKi/Util/Logger.h>

namespace anki {

template<>
template<>
Input& MakeSingletonPtr<Input>::allocateSingleton<>()
{
	ANKI_ASSERT(m_global == nullptr);
	m_global = new InputAndroid;

#if ANKI_ASSERTIONS_ENABLED
	++g_singletonsAllocated;
#endif

	return *m_global;
}

template<>
void MakeSingletonPtr<Input>::freeSingleton()
{
	if(m_global)
	{
		delete static_cast<InputAndroid*>(m_global);
		m_global = nullptr;
#if ANKI_ASSERTIONS_ENABLED
		--g_singletonsAllocated;
#endif
	}
}

Error Input::init()
{
	return static_cast<InputAndroid*>(this)->initInternal();
}

Error Input::handleEvents()
{
	for(U32& k : m_touchPointers)
	{
		if(k)
		{
			++k;
		}
	}

	int ident;
	int events;
	android_poll_source* source;

	while((ident = ALooper_pollAll(0, nullptr, &events, reinterpret_cast<void**>(&source))) >= 0)
	{
		if(source != nullptr)
		{
			source->process(g_androidApp, source);
		}
	}

	return Error::kNone;
}

void Input::moveCursor(const Vec2& posNdc)
{
	m_mousePosNdc = posNdc;
	m_mousePosWin = UVec2((posNdc * 0.5f + 0.5f) * Vec2(F32(NativeWindow::getSingleton().getWidth()), F32(NativeWindow::getSingleton().getHeight())));
}

void Input::hideCursor([[maybe_unused]] Bool hide)
{
	// do nothing
}

Bool Input::hasTouchDevice() const
{
	return true;
}

Error InputAndroid::initInternal()
{
	g_androidApp->userData = this;

	g_androidApp->onAppCmd = [](android_app* app, int32_t cmd) {
		InputAndroid* self = static_cast<InputAndroid*>(app->userData);
		self->handleAndroidEvents(app, cmd);
	};

	g_androidApp->onInputEvent = [](android_app* app, AInputEvent* event) -> int {
		InputAndroid* self = static_cast<InputAndroid*>(app->userData);
		return self->handleAndroidInput(app, event);
	};

	return Error::kNone;
}

void InputAndroid::handleAndroidEvents([[maybe_unused]] android_app* app, int32_t cmd)
{
	switch(cmd)
	{
	case APP_CMD_TERM_WINDOW:
	case APP_CMD_LOST_FOCUS:
		addEvent(InputEvent::kWindowClosed);
		break;
	}
}

int InputAndroid::handleAndroidInput([[maybe_unused]] android_app* app, AInputEvent* event)
{
	const I32 type = AInputEvent_getType(event);
	const I32 source = AInputEvent_getSource(event);
	I32 handled = 0;

	switch(type)
	{
	case AINPUT_EVENT_TYPE_KEY:
		// TODO
		break;

	case AINPUT_EVENT_TYPE_MOTION:
	{
		const I32 pointer = AMotionEvent_getAction(event);
		const I32 action = pointer & AMOTION_EVENT_ACTION_MASK;
		const I32 index = (pointer & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;

		if(source & AINPUT_SOURCE_JOYSTICK)
		{
			// TODO
		}
		else if(source & AINPUT_SOURCE_TOUCHSCREEN)
		{
			auto update = [event, this](U32 index, U32 pressValue) {
				const F32 x = AMotionEvent_getX(event, index);
				const F32 y = AMotionEvent_getY(event, index);
				const I32 id = AMotionEvent_getPointerId(event, index);

				m_touchPointerPosWin[id] = UVec2(U32(x), U32(y));

				m_touchPointerPosNdc[id].x() = F32(x) / F32(NativeWindow::getSingleton().getWidth()) * 2.0f - 1.0f;
				m_touchPointerPosNdc[id].y() = -(F32(y) / F32(NativeWindow::getSingleton().getHeight()) * 2.0f - 1.0f);

				if(pressValue == 0 || pressValue == 1)
				{
					m_touchPointers[id] = pressValue;
				}
			};

			switch(action)
			{
			case AMOTION_EVENT_ACTION_DOWN:
			case AMOTION_EVENT_ACTION_POINTER_DOWN:
				update(index, 1);
				break;
			case AMOTION_EVENT_ACTION_MOVE:
			{
				const U32 count = U32(AMotionEvent_getPointerCount(event));
				for(U32 i = 0; i < count; i++)
				{
					update(i, 2);
				}
				break;
			}
			case AMOTION_EVENT_ACTION_UP:
			case AMOTION_EVENT_ACTION_POINTER_UP:
				update(index, 0);
				break;

			default:
				break;
			}
		}
		break;
	}

	default:
		break;
	}

	return handled;
}

} // end namespace anki
