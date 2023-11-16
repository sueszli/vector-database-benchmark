#include <cassert>
#include "InputBindingManager.h"
#include "ThreadUtils.h"
#include "AppConfig.h"
#include "string_format.h"

#define CONFIG_PREFIX ("input")
#define CONFIG_BINDING_TYPE ("bindingtype")

#define CONFIG_BINDINGTARGET_PROVIDERID ("providerId")
#define CONFIG_BINDINGTARGET_DEVICEID ("deviceId")
#define CONFIG_BINDINGTARGET_KEYID ("keyId")
#define CONFIG_BINDINGTARGET_KEYTYPE ("keyType")

#define CONFIG_BINDINGTARGET1 ("bindingtarget1")
#define CONFIG_BINDINGTARGET2 ("bindingtarget2")

#define CONFIG_POVHATBINDING_REFVALUE ("povhatbinding.refvalue")

#define CONFIG_ANALOG_SENSITIVITY ("analog.sensitivity")

#define DEFAULT_ANALOG_SENSITIVITY (1.0f)

// clang-format off
uint32 CInputBindingManager::m_buttonDefaultValue[PS2::CControllerInfo::MAX_BUTTONS] =
{
	0x7F,
	0x7F,
	0x7F,
	0x7F,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0
};

const char* CInputBindingManager::m_padPreferenceName[] =
{
	"pad1",
	"pad2"
};

// clang-format on

static bool TryParseDeviceId(const char* input, DeviceIdType& out)
{
	uint32 bytes[6] = {0};
	if(std::sscanf(input,
	               "%x:%x:%x:%x:%x:%x",
	               &bytes[0], &bytes[1], &bytes[2],
	               &bytes[3], &bytes[4], &bytes[5]) != 6)
	{
		return false;
	}
	for(int i = 0; i < 6; ++i)
	{
		out.at(i) = bytes[i];
	}
	return true;
}

static void RegisterBindingTargetPreference(Framework::CConfig& config, const char* base)
{
	config.RegisterPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_PROVIDERID).c_str(), 0);
	config.RegisterPreferenceString(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_DEVICEID).c_str(), "0:0:0:0:0:0");
	config.RegisterPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_KEYID).c_str(), 0);
	config.RegisterPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_KEYTYPE).c_str(), 0);
}

static BINDINGTARGET LoadBindingTargetPreference(Framework::CConfig& config, const char* base)
{
	BINDINGTARGET result;
	result.providerId = config.GetPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_PROVIDERID).c_str());
	auto deviceIdString = config.GetPreferenceString(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_DEVICEID).c_str());
	FRAMEWORK_MAYBE_UNUSED bool parseResult = TryParseDeviceId(deviceIdString, result.deviceId);
	assert(parseResult);
	result.keyId = config.GetPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_KEYID).c_str());
	result.keyType = static_cast<BINDINGTARGET::KEYTYPE>(config.GetPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_KEYTYPE).c_str()));
	return result;
}

static void SaveBindingTargetPreference(Framework::CConfig& config, const char* base, const BINDINGTARGET& target)
{
	auto deviceIdString = string_format("%x:%x:%x:%x:%x:%x",
	                                    target.deviceId[0], target.deviceId[1], target.deviceId[2],
	                                    target.deviceId[3], target.deviceId[4], target.deviceId[5]);
	config.SetPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_PROVIDERID).c_str(), target.providerId);
	config.SetPreferenceString(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_DEVICEID).c_str(), deviceIdString.c_str());
	config.SetPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_KEYID).c_str(), target.keyId);
	config.SetPreferenceInteger(Framework::CConfig::MakePreferenceName(base, CONFIG_BINDINGTARGET_KEYTYPE).c_str(), static_cast<int32>(target.keyType));
}

CInputBindingManager::CInputBindingManager()
{
	m_analogSensitivity.fill(DEFAULT_ANALOG_SENSITIVITY);
	m_config = CInputConfig::LoadProfile();
	Reload();
}

bool CInputBindingManager::HasBindings() const
{
	for(unsigned int pad = 0; pad < MAX_PADS; pad++)
	{
		for(unsigned int button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			const auto& binding = m_bindings[pad][button];
			if(binding) return true;
		}
	}
	return false;
}

void CInputBindingManager::RegisterInputProvider(const ProviderPtr& provider)
{
	auto connection = provider->OnInput.Connect([this](auto target, auto value) { this->OnInputEventReceived(target, value); });
	m_providers.insert(std::make_pair(provider->GetId(), provider));
	m_providersConnection.insert(std::make_pair(provider->GetId(), connection));
}

CInputBindingManager::ProviderConnectionMap CInputBindingManager::OverrideInputEventHandler(const InputEventFunction& inputEventHandler)
{
	CInputBindingManager::ProviderConnectionMap providersOverrideConnection;
	for(auto& providerPair : m_providers)
	{
		auto& provider = providerPair.second;
		auto connection = provider->OnInput.ConnectOverride(inputEventHandler);
		providersOverrideConnection.insert(std::make_pair(provider->GetId(), connection));
	}
	return providersOverrideConnection;
}

std::string CInputBindingManager::GetTargetDescription(const BINDINGTARGET& target) const
{
	auto providerIterator = m_providers.find(target.providerId);
	if(providerIterator == std::end(m_providers))
	{
		return "Unknown Provider";
	}
	auto provider = providerIterator->second;
	return provider->GetTargetDescription(target);
}

std::vector<DEVICEINFO> CInputBindingManager::GetDevices() const
{
	std::vector<DEVICEINFO> devices;
	for(auto& [_, provider] : m_providers)
	{
		auto providerDevices = provider->GetDevices();
		devices.insert(devices.end(), providerDevices.begin(), providerDevices.end());
	}
	return devices;
}

void CInputBindingManager::OnInputEventReceived(const BINDINGTARGET& target, uint32 value)
{
	for(unsigned int pad = 0; pad < MAX_PADS; pad++)
	{
		for(unsigned int button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			auto binding = m_bindings[pad][button];
			if(!binding) continue;
			uint32 bindingValue = value;
			if(PS2::CControllerInfo::IsAxis(static_cast<PS2::CControllerInfo::BUTTON>(button)))
			{
				float analogSensitivity = m_analogSensitivity[pad];
				if(analogSensitivity != 1.0f)
				{
					int32 biasedValue = static_cast<int32>(bindingValue) - 128;
					biasedValue = static_cast<int32>(static_cast<float>(biasedValue) * analogSensitivity);
					biasedValue = std::clamp(biasedValue, -128, 127);
					bindingValue = biasedValue + 128;
				}
			}
			binding->ProcessEvent(target, bindingValue);
		}
	}
}

void CInputBindingManager::Reload()
{
	for(unsigned int pad = 0; pad < MAX_PADS; pad++)
	{
		m_config->RegisterPreferenceFloat(Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], CONFIG_ANALOG_SENSITIVITY).c_str(), DEFAULT_ANALOG_SENSITIVITY);
		for(unsigned int button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			auto prefBase = Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], PS2::CControllerInfo::m_buttonName[button]);
			m_config->RegisterPreferenceInteger(Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDING_TYPE).c_str(), BINDING_UNBOUND);
			RegisterBindingTargetPreference(*m_config, Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDINGTARGET1).c_str());
			if(PS2::CControllerInfo::IsAxis(static_cast<PS2::CControllerInfo::BUTTON>(button)))
			{
				RegisterBindingTargetPreference(*m_config, Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDINGTARGET2).c_str());
			}
			CPovHatBinding::RegisterPreferences(*m_config, prefBase.c_str());
		}
		{
			auto prefBase = Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], "motor");
			RegisterBindingTargetPreference(*m_config, Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDINGTARGET1).c_str());
		}
	}

	for(unsigned int pad = 0; pad < MAX_PADS; pad++)
	{
		m_analogSensitivity[pad] = m_config->GetPreferenceFloat(Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], CONFIG_ANALOG_SENSITIVITY).c_str());
		for(unsigned int button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			auto prefBase = Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], PS2::CControllerInfo::m_buttonName[button]);
			auto prefBindingType = Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDING_TYPE);
			auto bindingType = static_cast<BINDINGTYPE>(m_config->GetPreferenceInteger(prefBindingType.c_str()));
			BindingPtr binding;
			switch(bindingType)
			{
			default:
				assert(false);
				[[fallthrough]];
			case BINDING_UNBOUND:
				break;
			case BINDING_SIMPLE:
				binding = std::make_shared<CSimpleBinding>();
				break;
			case BINDING_POVHAT:
				binding = std::make_shared<CPovHatBinding>();
				break;
			case BINDING_SIMULATEDAXIS:
				binding = std::make_shared<CSimulatedAxisBinding>();
				break;
			}
			if(binding)
			{
				binding->Load(*m_config, prefBase.c_str());
			}
			m_bindings[pad][button] = binding;
		}
		{
			auto binding = std::make_shared<CMotorBinding>(m_providers);
			auto prefBase = Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], "motor");
			binding->Load(*m_config, prefBase.c_str());
			m_motorBindings[pad] = binding;
		}
	}
	ResetBindingValues();
}

void CInputBindingManager::Load(std::string profile)
{
	m_config = CInputConfig::LoadProfile(profile);
	Reload();
}

void CInputBindingManager::Save()
{
	for(unsigned int pad = 0; pad < MAX_PADS; pad++)
	{
		m_config->SetPreferenceFloat(Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], CONFIG_ANALOG_SENSITIVITY).c_str(), m_analogSensitivity[pad]);
		for(unsigned int button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			auto prefBase = Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], PS2::CControllerInfo::m_buttonName[button]);
			auto prefBindingType = Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDING_TYPE);
			const auto& binding = m_bindings[pad][button];
			if(binding)
			{
				m_config->SetPreferenceInteger(prefBindingType.c_str(), binding->GetBindingType());
				binding->Save(*m_config, prefBase.c_str());
			}
			else
			{
				m_config->SetPreferenceInteger(prefBindingType.c_str(), BINDING_UNBOUND);
			}
		}

		{
			auto prefBase = Framework::CConfig::MakePreferenceName(CONFIG_PREFIX, m_padPreferenceName[pad], "motor");
			auto prefBindingType = Framework::CConfig::MakePreferenceName(prefBase, CONFIG_BINDING_TYPE);
			const auto& binding = m_motorBindings[pad];
			if(binding)
			{
				m_config->SetPreferenceInteger(prefBindingType.c_str(), binding->GetBindingType());
				binding->Save(*m_config, prefBase.c_str());
			}
			else
			{
				m_config->SetPreferenceInteger(prefBindingType.c_str(), BINDING_UNBOUND);
			}
		}
	}
	m_config->Save();
}

const CInputBindingManager::CBinding* CInputBindingManager::GetBinding(uint32 pad, PS2::CControllerInfo::BUTTON button) const
{
	if((pad >= MAX_PADS) || (button >= PS2::CControllerInfo::MAX_BUTTONS))
	{
		throw std::exception();
	}
	return m_bindings[pad][button].get();
}

CInputBindingManager::CMotorBinding* CInputBindingManager::GetMotorBinding(uint32 pad) const
{
	if(pad >= MAX_PADS)
	{
		throw std::exception();
	}
	return m_motorBindings[pad].get();
}

void CInputBindingManager::SetMotorBinding(uint32 pad, const BINDINGTARGET& binding)
{
	m_motorBindings[pad] = std::make_shared<CMotorBinding>(binding, m_providers);
}

float CInputBindingManager::GetAnalogSensitivity(uint32 pad) const
{
	return m_analogSensitivity[pad];
}

void CInputBindingManager::SetAnalogSensitivity(uint32 pad, float value)
{
	m_analogSensitivity[pad] = value;
}

uint32 CInputBindingManager::GetBindingValue(uint32 pad, PS2::CControllerInfo::BUTTON button) const
{
	assert(pad < MAX_PADS);
	assert(button < PS2::CControllerInfo::MAX_BUTTONS);
	const auto& binding = m_bindings[pad][button];
	if(binding)
	{
		return binding->GetValue();
	}
	else
	{
		return m_buttonDefaultValue[button];
	}
}

void CInputBindingManager::ResetBindingValues()
{
	for(unsigned int pad = 0; pad < MAX_PADS; pad++)
	{
		for(unsigned int button = 0; button < PS2::CControllerInfo::MAX_BUTTONS; button++)
		{
			const auto& binding = m_bindings[pad][button];
			if(!binding) continue;
			binding->SetValue(m_buttonDefaultValue[button]);
		}
	}
}

void CInputBindingManager::SetSimpleBinding(uint32 pad, PS2::CControllerInfo::BUTTON button, const BINDINGTARGET& binding)
{
	if((pad >= MAX_PADS) || (button >= PS2::CControllerInfo::MAX_BUTTONS))
	{
		throw std::exception();
	}
	m_bindings[pad][button] = std::make_shared<CSimpleBinding>(binding);
}

void CInputBindingManager::SetPovHatBinding(uint32 pad, PS2::CControllerInfo::BUTTON button, const BINDINGTARGET& binding, uint32 refValue)
{
	if((pad >= MAX_PADS) || (button >= PS2::CControllerInfo::MAX_BUTTONS))
	{
		throw std::exception();
	}
	m_bindings[pad][button] = std::make_shared<CPovHatBinding>(binding, refValue);
}

void CInputBindingManager::SetSimulatedAxisBinding(uint32 pad, PS2::CControllerInfo::BUTTON button, const BINDINGTARGET& binding1, const BINDINGTARGET& binding2)
{
	if((pad >= MAX_PADS) || (button >= PS2::CControllerInfo::MAX_BUTTONS))
	{
		throw std::exception();
	}
	m_bindings[pad][button] = std::make_shared<CSimulatedAxisBinding>(binding1, binding2);
}

void CInputBindingManager::ResetBinding(uint32 pad, PS2::CControllerInfo::BUTTON button)
{
	if((pad >= MAX_PADS) || (button >= PS2::CControllerInfo::MAX_BUTTONS))
	{
		throw std::exception();
	}
	m_bindings[pad][button].reset();
}

////////////////////////////////////////////////
// SimpleBinding
////////////////////////////////////////////////
CInputBindingManager::CSimpleBinding::CSimpleBinding(const BINDINGTARGET& binding)
    : m_binding(binding)
    , m_value(0)
{
}

void CInputBindingManager::CSimpleBinding::ProcessEvent(const BINDINGTARGET& target, uint32 value)
{
	if(m_binding != target) return;
	m_value = value;
}

CInputBindingManager::BINDINGTYPE CInputBindingManager::CSimpleBinding::GetBindingType() const
{
	return BINDING_SIMPLE;
}

const char* CInputBindingManager::CSimpleBinding::GetBindingTypeName() const
{
	return "simplebinding";
}

uint32 CInputBindingManager::CSimpleBinding::GetValue() const
{
	return m_value;
}

void CInputBindingManager::CSimpleBinding::SetValue(uint32 value)
{
	m_value = value;
}

std::string CInputBindingManager::CSimpleBinding::GetDescription(CInputBindingManager* bindingManager) const
{
	return bindingManager->GetTargetDescription(m_binding);
}

void CInputBindingManager::CSimpleBinding::Save(Framework::CConfig& config, const char* buttonBase) const
{
	auto prefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
	SaveBindingTargetPreference(config, prefBase.c_str(), m_binding);
}

void CInputBindingManager::CSimpleBinding::Load(Framework::CConfig& config, const char* buttonBase)
{
	auto prefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
	m_binding = LoadBindingTargetPreference(config, prefBase.c_str());
}

////////////////////////////////////////////////
// PovHatBinding
////////////////////////////////////////////////

CInputBindingManager::CPovHatBinding::CPovHatBinding(const BINDINGTARGET& binding, uint32 refValue)
    : m_binding(binding)
    , m_refValue(refValue)
{
}

CInputBindingManager::BINDINGTYPE CInputBindingManager::CPovHatBinding::GetBindingType() const
{
	return BINDING_POVHAT;
}

const char* CInputBindingManager::CPovHatBinding::GetBindingTypeName() const
{
	return "povhatbinding";
}

void CInputBindingManager::CPovHatBinding::RegisterPreferences(Framework::CConfig& config, const char* buttonBase)
{
	config.RegisterPreferenceInteger(Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_POVHATBINDING_REFVALUE).c_str(), -1);
}

void CInputBindingManager::CPovHatBinding::Save(Framework::CConfig& config, const char* buttonBase) const
{
	{
		auto prefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
		SaveBindingTargetPreference(config, prefBase.c_str(), m_binding);
	}
	config.SetPreferenceInteger(Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_POVHATBINDING_REFVALUE).c_str(), m_refValue);
}

void CInputBindingManager::CPovHatBinding::Load(Framework::CConfig& config, const char* buttonBase)
{
	{
		auto prefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
		m_binding = LoadBindingTargetPreference(config, prefBase.c_str());
	}
	m_refValue = config.GetPreferenceInteger(Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_POVHATBINDING_REFVALUE).c_str());
}

void CInputBindingManager::CPovHatBinding::ProcessEvent(const BINDINGTARGET& target, uint32 value)
{
	if(m_binding != target) return;
	m_value = value;
}

std::string CInputBindingManager::CPovHatBinding::GetDescription(CInputBindingManager* inputBindingManager) const
{
	return string_format("%s - %d",
	                     inputBindingManager->GetTargetDescription(m_binding).c_str(),
	                     m_refValue);
}

uint32 CInputBindingManager::CPovHatBinding::GetValue() const
{
	if(m_value >= BINDINGTARGET::POVHAT_MAX) return 0;
	int32 normalizedRefValue = (m_refValue * 360) / BINDINGTARGET::POVHAT_MAX;
	int32 normalizedValue = (m_value * 360) / BINDINGTARGET::POVHAT_MAX;
	if(GetShortestDistanceBetweenAngles(normalizedValue, normalizedRefValue) <= 45)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void CInputBindingManager::CPovHatBinding::SetValue(uint32 value)
{
	if(value == 0)
	{
		//Using POVHAT_MAX will make GetValue return 0
		m_value = BINDINGTARGET::POVHAT_MAX;
	}
	else
	{
		//Using refValue will make GetValue return 1
		m_value = m_refValue;
	}
}

int32 CInputBindingManager::CPovHatBinding::GetShortestDistanceBetweenAngles(int32 angle1, int32 angle2)
{
	if(angle1 > 180) angle1 -= 360;
	if(angle1 < -180) angle1 += 360;
	if(angle2 > 180) angle2 -= 360;
	if(angle2 < -180) angle2 += 360;
	int32 angle = abs(angle1 - angle2);
	if(angle > 180)
	{
		angle = 360 - angle;
	}
	return angle;
}

////////////////////////////////////////////////
// CSimulatedAxisBinding
////////////////////////////////////////////////
CInputBindingManager::CSimulatedAxisBinding::CSimulatedAxisBinding(const BINDINGTARGET& binding1, const BINDINGTARGET& binding2)
    : m_key1Binding(binding1)
    , m_key2Binding(binding2)
{
}

void CInputBindingManager::CSimulatedAxisBinding::ProcessEvent(const BINDINGTARGET& target, uint32 state)
{
	if(m_key1Binding == target)
	{
		m_key1State = state;
	}
	else if(m_key2Binding == target)
	{
		m_key2State = state;
	}
}

uint32 CInputBindingManager::CSimulatedAxisBinding::GetValue() const
{
	uint32 value = 0x7F;
	if(m_key1State && m_key2State)
	{
		value = 0x7F;
	}
	if(m_key1State)
	{
		value = 0;
	}
	else if(m_key2State)
	{
		value = 0xFF;
	}
	return value;
}

CInputBindingManager::BINDINGTYPE CInputBindingManager::CSimulatedAxisBinding::GetBindingType() const
{
	return BINDING_SIMULATEDAXIS;
}

const char* CInputBindingManager::CSimulatedAxisBinding::GetBindingTypeName() const
{
	return "simulatedaxisbinding";
}

std::string CInputBindingManager::CSimulatedAxisBinding::GetDescription(CInputBindingManager* inputBindingManager) const
{
	return string_format("(%s, %s)",
	                     inputBindingManager->GetTargetDescription(m_key1Binding).c_str(),
	                     inputBindingManager->GetTargetDescription(m_key2Binding).c_str());
}

void CInputBindingManager::CSimulatedAxisBinding::SetValue(uint32 state)
{
	m_key1State = 0;
	m_key2State = 0;
}

void CInputBindingManager::CSimulatedAxisBinding::CSimulatedAxisBinding::Save(Framework::CConfig& config, const char* buttonBase) const
{
	auto key1PrefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
	auto key2PrefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET2);
	SaveBindingTargetPreference(config, key1PrefBase.c_str(), m_key1Binding);
	SaveBindingTargetPreference(config, key2PrefBase.c_str(), m_key2Binding);
}

void CInputBindingManager::CSimulatedAxisBinding::Load(Framework::CConfig& config, const char* buttonBase)
{
	auto key1PrefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
	auto key2PrefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET2);
	m_key1Binding = LoadBindingTargetPreference(config, key1PrefBase.c_str());
	m_key2Binding = LoadBindingTargetPreference(config, key2PrefBase.c_str());
}

////////////////////////////////////////////////
// CMotorBinding, Specialised binding that can communicate back to a provider
////////////////////////////////////////////////
CInputBindingManager::CMotorBinding::CMotorBinding(const BINDINGTARGET& binding, const CInputBindingManager::ProviderMap& providers)
    : m_binding(binding)
    , m_providers(providers)
    , m_running(true)
    , m_nextTimeout(std::chrono::steady_clock::now())
{
	m_thread = std::thread(&CInputBindingManager::CMotorBinding::ThreadProc, this);
	Framework::ThreadUtils::SetThreadName(m_thread, "MotorBinding Thread");
}

CInputBindingManager::CMotorBinding::CMotorBinding(ProviderMap& providers)
    : CMotorBinding(BINDINGTARGET(), providers)
{
}

CInputBindingManager::CMotorBinding::CMotorBinding()
    : CMotorBinding(BINDINGTARGET(), {})
{
}

CInputBindingManager::CMotorBinding::~CMotorBinding()
{
	m_running = false;
	m_cv.notify_all();
	if(m_thread.joinable())
	{
		m_thread.join();
	}
}

void CInputBindingManager::CMotorBinding::ProcessEvent(uint8 largeMotor, uint8 smallMotor)
{
	for(auto& [id, provider] : m_providers)
	{
		if(id == m_binding.providerId)
		{
			provider->SetVibration(m_binding.deviceId, largeMotor, smallMotor);
			if(largeMotor + smallMotor)
			{
				m_nextTimeout = std::chrono::steady_clock::now() + std::chrono::seconds(1);
				m_cv.notify_all();
			}
		}
	}
}

CInputBindingManager::BINDINGTYPE CInputBindingManager::CMotorBinding::GetBindingType() const
{
	return BINDING_MOTOR;
}

BINDINGTARGET CInputBindingManager::CMotorBinding::GetBindingTarget() const
{
	return m_binding;
}

void CInputBindingManager::CMotorBinding::Save(Framework::CConfig& config, const char* buttonBase) const
{
	auto prefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
	SaveBindingTargetPreference(config, prefBase.c_str(), m_binding);
}

void CInputBindingManager::CMotorBinding::Load(Framework::CConfig& config, const char* buttonBase)
{
	auto prefBase = Framework::CConfig::MakePreferenceName(buttonBase, CONFIG_BINDINGTARGET1);
	m_binding = LoadBindingTargetPreference(config, prefBase.c_str());
}

void CInputBindingManager::CMotorBinding::ThreadProc()
{
	while(m_running)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_cv.wait(lock);

		while(m_running && m_nextTimeout.load() > std::chrono::steady_clock::now())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}

		for(auto& [id, provider] : m_providers)
		{
			if(id == m_binding.providerId)
			{
				provider->SetVibration(m_binding.deviceId, 0, 0);
			}
		}
	}
}
