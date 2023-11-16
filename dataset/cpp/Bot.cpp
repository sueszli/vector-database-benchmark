#include "tgbot/net/BoostHttpOnlySslClient.h"
#include "tgbot/Bot.h"

#include "tgbot/EventBroadcaster.h"

#include <memory>
#include <string>

namespace TgBot {

Bot::Bot(std::string token, const HttpClient& httpClient, const std::string& url)
    : _token(std::move(token))
    , _api(_token, httpClient, url)
    , _eventBroadcaster(std::make_unique<EventBroadcaster>())
    , _eventHandler(getEvents()) {
}

HttpClient& Bot::_getDefaultHttpClient() {
    static BoostHttpOnlySslClient instance;
    return instance;
}

}
