#pragma once

#ifdef USE_KEYCHAIN

#include <QString>

class Keychain
{
public:
	static auto getPassword(const QString &user) -> QString;
	static auto setPassword(const QString &user, const QString &password) -> bool;
	static auto clearPassword(const QString &user) -> bool;

private:
	Keychain() = default;
};

#endif
