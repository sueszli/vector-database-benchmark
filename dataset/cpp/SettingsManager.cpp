#include <jni.h>
#include "SettingsManager.h"
#include "NativeShared.h"
#include "../AppConfig.h"

void CSettingsManager::Save()
{
	CAppConfig::GetInstance().Save();
}

void CSettingsManager::RegisterPreferenceBoolean(const std::string& name, bool value)
{
	CAppConfig::GetInstance().RegisterPreferenceBoolean(name.c_str(), value);
}

bool CSettingsManager::GetPreferenceBoolean(const std::string& name)
{
	return CAppConfig::GetInstance().GetPreferenceBoolean(name.c_str());
}

void CSettingsManager::SetPreferenceBoolean(const std::string& name, bool value)
{
	CAppConfig::GetInstance().SetPreferenceBoolean(name.c_str(), value);
}

int CSettingsManager::GetPreferenceInteger(const std::string& name)
{
	return CAppConfig::GetInstance().GetPreferenceInteger(name.c_str());
}

void CSettingsManager::SetPreferenceInteger(const std::string& name, int value)
{
	CAppConfig::GetInstance().SetPreferenceInteger(name.c_str(), value);
}

extern "C" JNIEXPORT void JNICALL Java_com_virtualapplications_play_SettingsManager_save(JNIEnv* env, jobject obj)
{
	CSettingsManager::GetInstance().Save();
}

extern "C" JNIEXPORT void JNICALL Java_com_virtualapplications_play_SettingsManager_registerPreferenceBoolean(JNIEnv* env, jobject obj, jstring name, jboolean value)
{
	CSettingsManager::GetInstance().RegisterPreferenceBoolean(GetStringFromJstring(env, name), value == JNI_TRUE);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_virtualapplications_play_SettingsManager_getPreferenceBoolean(JNIEnv* env, jobject obj, jstring name)
{
	return CSettingsManager::GetInstance().GetPreferenceBoolean(GetStringFromJstring(env, name));
}

extern "C" JNIEXPORT void JNICALL Java_com_virtualapplications_play_SettingsManager_setPreferenceBoolean(JNIEnv* env, jobject obj, jstring name, jboolean value)
{
	CSettingsManager::GetInstance().SetPreferenceBoolean(GetStringFromJstring(env, name), value == JNI_TRUE);
}

extern "C" JNIEXPORT jint JNICALL Java_com_virtualapplications_play_SettingsManager_getPreferenceInteger(JNIEnv* env, jobject obj, jstring name)
{
	return CSettingsManager::GetInstance().GetPreferenceInteger(GetStringFromJstring(env, name).c_str());
}

extern "C" JNIEXPORT void JNICALL Java_com_virtualapplications_play_SettingsManager_setPreferenceInteger(JNIEnv* env, jobject obj, jstring name, jint value)
{
	CSettingsManager::GetInstance().SetPreferenceInteger(GetStringFromJstring(env, name).c_str(), value);
}
