#include "server_component_impl.h"
#include "mavsdk_impl.h"

namespace mavsdk {

ServerComponent::ServerComponent(MavsdkImpl& mavsdk_impl, uint8_t component_id) :
    _impl(std::make_unique<ServerComponentImpl>(mavsdk_impl, component_id))
{}

uint8_t ServerComponent::component_id() const
{
    return _impl->get_own_component_id();
}

} // namespace mavsdk
