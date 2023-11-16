////////////////////////////////////////////////////////////////////////////
//
// Copyright 2020 Realm Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or utilied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
////////////////////////////////////////////////////////////////////////////

#include <realm/object-store/sync/app_credentials.hpp>
#include <realm/object-store/util/bson/bson.hpp>

namespace realm::app {

std::string const kAppProviderKey = "provider";

IdentityProvider const IdentityProviderAnonymous = "anon-user";
IdentityProvider const IdentityProviderGoogle = "oauth2-google";
IdentityProvider const IdentityProviderFacebook = "oauth2-facebook";
IdentityProvider const IdentityProviderApple = "oauth2-apple";
IdentityProvider const IdentityProviderUsernamePassword = "local-userpass";
IdentityProvider const IdentityProviderCustom = "custom-token";
IdentityProvider const IdentityProviderFunction = "custom-function";
IdentityProvider const IdentityProviderAPIKey = "api-key";

IdentityProvider provider_type_from_enum(AuthProvider provider)
{
    switch (provider) {
        case AuthProvider::ANONYMOUS:
        case AuthProvider::ANONYMOUS_NO_REUSE:
            return IdentityProviderAnonymous;
        case AuthProvider::APPLE:
            return IdentityProviderApple;
        case AuthProvider::FACEBOOK:
            return IdentityProviderFacebook;
        case AuthProvider::GOOGLE:
            return IdentityProviderGoogle;
        case AuthProvider::CUSTOM:
            return IdentityProviderCustom;
        case AuthProvider::USERNAME_PASSWORD:
            return IdentityProviderUsernamePassword;
        case AuthProvider::FUNCTION:
            return IdentityProviderFunction;
        case AuthProvider::API_KEY:
            return IdentityProviderAPIKey;
    }
    throw InvalidArgument("unknown provider type in provider_type_from_enum");
}

AuthProvider enum_from_provider_type(const IdentityProvider& provider)
{
    if (provider == IdentityProviderAnonymous) {
        return AuthProvider::ANONYMOUS;
    }
    else if (provider == IdentityProviderApple) {
        return AuthProvider::APPLE;
    }
    else if (provider == IdentityProviderFacebook) {
        return AuthProvider::FACEBOOK;
    }
    else if (provider == IdentityProviderGoogle) {
        return AuthProvider::GOOGLE;
    }
    else if (provider == IdentityProviderCustom) {
        return AuthProvider::CUSTOM;
    }
    else if (provider == IdentityProviderUsernamePassword) {
        return AuthProvider::USERNAME_PASSWORD;
    }
    else if (provider == IdentityProviderFunction) {
        return AuthProvider::FUNCTION;
    }
    else if (provider == IdentityProviderAPIKey) {
        return AuthProvider::API_KEY;
    }
    else {
        REALM_UNREACHABLE();
    }
}

AppCredentials::AppCredentials(AuthProvider provider, std::unique_ptr<bson::BsonDocument> payload)
    : m_provider(provider)
    , m_payload(std::move(payload))
{
}

AppCredentials::AppCredentials(AuthProvider provider,
                               std::initializer_list<std::pair<const char*, bson::Bson>> values)
    : m_provider(provider)
    , m_payload(std::make_unique<bson::BsonDocument>())
{
    (*m_payload)[kAppProviderKey] = provider_type_from_enum(provider);
    for (auto& [key, value] : values) {
        (*m_payload)[key] = std::move(value);
    }
}

AuthProvider AppCredentials::provider() const
{
    return m_provider;
}

std::string AppCredentials::provider_as_string() const
{
    return provider_type_from_enum(m_provider);
}

bson::BsonDocument AppCredentials::serialize_as_bson() const
{
    return *m_payload;
}

std::string AppCredentials::serialize_as_json() const
{
    return bson::Bson(*m_payload).to_string();
}

AppCredentials AppCredentials::anonymous(bool reuse_credentials)
{
    return reuse_credentials ? AppCredentials(AuthProvider::ANONYMOUS, {})
                             : AppCredentials(AuthProvider::ANONYMOUS_NO_REUSE, {});
}

AppCredentials AppCredentials::apple(AppCredentialsToken id_token)
{
    return AppCredentials(AuthProvider::APPLE, {{"id_token", id_token}});
}

AppCredentials AppCredentials::facebook(AppCredentialsToken access_token)
{
    return AppCredentials(AuthProvider::FACEBOOK, {{"accessToken", access_token}});
}

AppCredentials AppCredentials::google(AuthCode&& auth_token)
{
    return AppCredentials(AuthProvider::GOOGLE, {{"authCode", bson::Bson(auth_token)}});
}

AppCredentials AppCredentials::google(IdToken&& id_token)
{
    return AppCredentials(AuthProvider::GOOGLE, {{"id_token", bson::Bson(id_token)}});
}

AppCredentials AppCredentials::custom(AppCredentialsToken token)
{
    return AppCredentials(AuthProvider::CUSTOM, {{"token", token}});
}

AppCredentials AppCredentials::username_password(std::string username, std::string password)
{
    return AppCredentials(AuthProvider::USERNAME_PASSWORD, {{"username", username}, {"password", password}});
}

AppCredentials AppCredentials::function(const bson::BsonDocument& payload)
{
    return AppCredentials(AuthProvider::FUNCTION, std::make_unique<bson::BsonDocument>(payload));
}

AppCredentials AppCredentials::function(const std::string& serialized_payload)
{
    return AppCredentials(
        AuthProvider::FUNCTION,
        std::make_unique<bson::BsonDocument>(static_cast<bson::BsonDocument>(bson::parse(serialized_payload))));
}


AppCredentials AppCredentials::api_key(std::string api_key)
{
    return AppCredentials(AuthProvider::API_KEY, {{"key", api_key}});
}

AppCredentials::AppCredentials(const AppCredentials& credentials)
    : m_provider(credentials.m_provider)
    , m_payload(std::make_unique<bson::BsonDocument>(*credentials.m_payload))
{
}

AppCredentials& AppCredentials::operator=(const AppCredentials& credentials)
{
    if (&credentials != this) {
        m_payload = std::make_unique<bson::BsonDocument>(*credentials.m_payload);
        m_provider = credentials.m_provider;
    }
    return *this;
}

} // namespace realm::app
