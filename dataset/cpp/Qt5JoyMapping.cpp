#if defined(WITH_QT5)

#include "../Input/JoyMapping.h"
#include "../Input/IInputManager.h"
#include "../Input/IInputEventHandler.h"
#include "../Primitives/Vector2.h"
#include "../Base/Algorithms.h"

namespace nCine
{
	const unsigned int JoyMapping::MaxNameLength;

	const char* JoyMapping::AxesStrings[JoyMappedState::NumAxes] = {
		"leftx",
		"lefty",
		"rightx",
		"righty",
		"lefttrigger",
		"righttrigger"
	};

	const char* JoyMapping::ButtonsStrings[JoyMappedState::NumButtons] = {
		"a",
		"b",
		"x",
		"y",
		"back",
		"guide",
		"start",
		"leftstick",
		"rightstick",
		"leftshoulder",
		"rightshoulder",
		"dpup",
		"dpdown",
		"dpleft",
		"dpright"
	};

	JoyMappedStateImpl JoyMapping::nullMappedJoyState_;
	nctl::StaticArray<JoyMappedStateImpl, JoyMapping::MaxNumJoysticks> JoyMapping::mappedJoyStates_(nctl::StaticArrayMode::EXTEND_SIZE);
	JoyMappedButtonEvent JoyMapping::mappedButtonEvent_;
	JoyMappedAxisEvent JoyMapping::mappedAxisEvent_;

	JoyMapping::MappedJoystick::MappedJoystick()
	{
		name[0] = '\0';

		for (unsigned int i = 0; i < MaxNumAxes; i++)
			desc.axes[i].name = AxisName::UNKNOWN;
		for (unsigned int i = 0; i < MaxNumButtons; i++)
			buttons[i] = ButtonName::UNKNOWN;
		for (unsigned int i = 0; i < MaxHatButtons; i++)
			hats[i] = ButtonName::UNKNOWN;
	}

	JoyMapping::MappedJoystick::Guid::Guid()
	{
		for (unsigned int i = 0; i < 4; i++)
			array_[i] = 0;
	}

	JoyMapping::JoyMapping()
		: mappings_(1), inputManager_(nullptr), inputEventHandler_(nullptr)
	{
		mappings_.emplace_back();
		mappings_[0].axes[0].name = AxisName::LX;
		mappings_[0].axes[0].min = -1.0f;
		mappings_[0].axes[0].max = 1.0f;
		mappings_[0].axes[1].name = AxisName::LY;
		mappings_[0].axes[1].min = -1.0f;
		mappings_[0].axes[1].max = 1.0f;
		mappings_[0].axes[2].name = AxisName::RX;
		mappings_[0].axes[2].min = -1.0f;
		mappings_[0].axes[2].max = 1.0f;
		mappings_[0].axes[3].name = AxisName::RY;
		mappings_[0].axes[3].min = -1.0f;
		mappings_[0].axes[3].max = 1.0f;
		mappings_[0].axes[4].name = AxisName::LTRIGGER;
		mappings_[0].axes[4].min = 0.0f;
		mappings_[0].axes[4].max = 1.0f;
		mappings_[0].axes[5].name = AxisName::RTRIGGER;
		mappings_[0].axes[5].min = 0.0f;
		mappings_[0].axes[5].max = 1.0f;

		mappings_[0].buttons[0] = ButtonName::LBUMPER;
		mappings_[0].buttons[1] = ButtonName::LSTICK;
		mappings_[0].buttons[2] = ButtonName::RBUMPER;
		mappings_[0].buttons[3] = ButtonName::RSTICK;
		mappings_[0].buttons[4] = ButtonName::A;
		mappings_[0].buttons[5] = ButtonName::B;
		mappings_[0].buttons[6] = ButtonName::UNKNOWN;
		mappings_[0].buttons[7] = ButtonName::GUIDE;
		mappings_[0].buttons[8] = ButtonName::BACK;
		mappings_[0].buttons[9] = ButtonName::START;
		mappings_[0].buttons[10] = ButtonName::X;
		mappings_[0].buttons[11] = ButtonName::Y;

		mappings_[0].hats[0] = ButtonName::DPAD_UP;
		mappings_[0].hats[1] = ButtonName::DPAD_DOWN;
		mappings_[0].hats[2] = ButtonName::DPAD_RIGHT;
		mappings_[0].hats[3] = ButtonName::DPAD_LEFT;
	}

	void JoyMapping::MappedJoystick::Guid::fromString(const char* string)
	{
	}

	bool JoyMapping::MappedJoystick::Guid::operator==(const Guid& guid) const
	{
		return false;
	}

	void JoyMapping::init(const IInputManager* inputManager)
	{
		//ASSERT(inputManager);
		inputManager_ = inputManager;
	}

	bool JoyMapping::addMappingFromString(const char* mappingString)
	{
		return false;
	}

	void JoyMapping::addMappingsFromStrings(const char** mappingStrings)
	{
	}

	void JoyMapping::addMappingsFromFile(const char* filename)
	{
	}

	void JoyMapping::onJoyButtonPressed(const JoyButtonEvent& event)
	{
		if (inputEventHandler_ == nullptr)
			return;

		const int idToIndex = mappingIndex_[event.joyId];
		if (idToIndex != -1 &&
			event.buttonId >= 0 && event.buttonId < static_cast<int>(MappedJoystick::MaxNumButtons)) {
			mappedButtonEvent_.joyId = event.joyId;
			mappedButtonEvent_.buttonName = mappings_[idToIndex].buttons[event.buttonId];
			if (mappedButtonEvent_.buttonName != ButtonName::UNKNOWN) {
				const int buttonId = static_cast<int>(mappedButtonEvent_.buttonName);
				mappedJoyStates_[event.joyId].buttons_[buttonId] = true;
				inputEventHandler_->OnJoyMappedButtonPressed(mappedButtonEvent_);
			}
		}
	}

	void JoyMapping::onJoyButtonReleased(const JoyButtonEvent& event)
	{
		if (inputEventHandler_ == nullptr)
			return;

		const int idToIndex = mappingIndex_[event.joyId];
		if (idToIndex != -1 &&
			event.buttonId >= 0 && event.buttonId < static_cast<int>(MappedJoystick::MaxNumButtons)) {
			mappedButtonEvent_.joyId = event.joyId;
			mappedButtonEvent_.buttonName = mappings_[idToIndex].buttons[event.buttonId];
			if (mappedButtonEvent_.buttonName != ButtonName::UNKNOWN) {
				const int buttonId = static_cast<int>(mappedButtonEvent_.buttonName);
				mappedJoyStates_[event.joyId].buttons_[buttonId] = false;
				inputEventHandler_->OnJoyMappedButtonReleased(mappedButtonEvent_);
			}
		}
	}

	void JoyMapping::onJoyHatMoved(const JoyHatEvent& event)
	{
		if (inputEventHandler_ == nullptr)
			return;

		const int idToIndex = mappingIndex_[event.joyId];
		// Only the first gamepad hat is mapped
		if (idToIndex != -1 && event.hatId == 0 &&
			mappedJoyStates_[event.joyId].lastHatState_ != event.hatState) {
			mappedButtonEvent_.joyId = event.joyId;

			const unsigned char oldHatState = mappedJoyStates_[event.joyId].lastHatState_;
			const unsigned char newHatState = event.hatState;

			const unsigned char firstHatValue = HatState::UP;
			const unsigned char lastHatValue = HatState::LEFT;
			for (unsigned char hatValue = firstHatValue; hatValue <= lastHatValue; hatValue *= 2) {
				if ((oldHatState & hatValue) != (newHatState & hatValue)) {
					int hatIndex = hatStateToIndex(hatValue);

					mappedButtonEvent_.buttonName = mappings_[idToIndex].hats[hatIndex];
					if (mappedButtonEvent_.buttonName != ButtonName::UNKNOWN) {
						const int buttonId = static_cast<int>(mappedButtonEvent_.buttonName);
						if (newHatState & hatValue) {
							mappedJoyStates_[event.joyId].buttons_[buttonId] = true;
							inputEventHandler_->OnJoyMappedButtonPressed(mappedButtonEvent_);
						} else {
							mappedJoyStates_[event.joyId].buttons_[buttonId] = false;
							inputEventHandler_->OnJoyMappedButtonReleased(mappedButtonEvent_);
						}
					}
				}
				mappedJoyStates_[event.joyId].lastHatState_ = event.hatState;
			}
		}
	}

	void JoyMapping::onJoyAxisMoved(const JoyAxisEvent& event)
	{
		if (inputEventHandler_ == nullptr)
			return;

		const int idToIndex = mappingIndex_[event.joyId];
		if (idToIndex != -1 &&
			event.axisId >= 0 && event.axisId < static_cast<int>(MappedJoystick::MaxNumAxes)) {
			const MappedJoystick::Axis& axis = mappings_[idToIndex].axes[event.axisId];

			mappedAxisEvent_.joyId = event.joyId;
			mappedAxisEvent_.axisName = axis.name;
			if (mappedAxisEvent_.axisName != AxisName::UNKNOWN) {
				const float value = (event.value + 1.0f) * 0.5f;
				mappedAxisEvent_.value = axis.min + value * (axis.max - axis.min);
				mappedJoyStates_[event.joyId].axesValues_[static_cast<int>(axis.name)] = mappedAxisEvent_.value;
				inputEventHandler_->OnJoyMappedAxisMoved(mappedAxisEvent_);
			}
		}
	}

	bool JoyMapping::onJoyConnected(const JoyConnectionEvent& event)
	{
		const char* joyName = inputManager_->joyName(event.joyId);

		// There is only one mapping for QGamepad
		mappingIndex_[event.joyId] = 0;
		LOGI("Joystick mapping found for \"%s\" (%d)", joyName, event.joyId);

		return (mappingIndex_[event.joyId] != -1);
	}

	void JoyMapping::onJoyDisconnected(const JoyConnectionEvent& event)
	{
		mappingIndex_[event.joyId] = -1;
	}

	bool JoyMapping::isJoyMapped(int joyId) const
	{
		return true;
	}

	const JoyMappedStateImpl& JoyMapping::joyMappedState(int joyId) const
	{
		if (joyId < 0 || joyId > MaxNumJoysticks)
			return nullMappedJoyState_;
		else
			return mappedJoyStates_[joyId];
	}

	void JoyMapping::deadZoneNormalize(Vector2f& joyVector, float deadZoneValue) const
	{
		deadZoneValue = nctl::clamp(deadZoneValue, 0.0f, 1.0f);

		if (joyVector.length() <= deadZoneValue)
			joyVector = Vector2f::Zero;
		else {
			float normalizedLength = (joyVector.length() - deadZoneValue) / (1.0f - deadZoneValue);
			normalizedLength = nctl::clamp(normalizedLength, 0.0f, 1.0f);
			joyVector = joyVector.normalize() * normalizedLength;
		}
	}

	void JoyMapping::checkConnectedJoystics()
	{
	}

	int JoyMapping::findMappingByGuid(const MappedJoystick::Guid& guid) const
	{
		return 0;
	}

	int JoyMapping::findMappingByName(const char* name) const
	{
		return 0;
	}

	bool JoyMapping::parseMappingFromString(const char* mappingString, MappedJoystick& map)
	{
		return false;
	}

	bool JoyMapping::parsePlatformKeyword(const char* start, const char* end) const
	{
		return false;
	}

	bool JoyMapping::parsePlatformName(const char* start, const char* end) const
	{
		return false;
	}

	int JoyMapping::parseAxisName(const char* start, const char* end) const
	{
		return -1;
	}

	int JoyMapping::parseButtonName(const char* start, const char* end) const
	{
		return -1;
	}

	int JoyMapping::parseAxisMapping(const char* start, const char* end, MappedJoystick::Axis& axis) const
	{
		return -1;
	}

	int JoyMapping::parseButtonMapping(const char* start, const char* end) const
	{
		return -1;
	}

	int JoyMapping::parseHatMapping(const char* start, const char* end) const
	{
		return -1;
	}

	int JoyMapping::hatStateToIndex(unsigned char hatState) const
	{
		switch (hatState) {
			case HatState::UP: return 0;
			case HatState::DOWN: return 1;
			case HatState::RIGHT: return 2;
			case HatState::LEFT: return 3;
			default: return 0;
		}
	}

	void JoyMapping::trimSpaces(const char** start, const char** end) const
	{
	}
}

#endif