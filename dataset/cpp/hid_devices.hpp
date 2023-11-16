// hid_devices.hpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2020 Leigh Johnston.  All Rights Reserved.
  
  This program is free software: you can redistribute it and / or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <neogfx/neogfx.hpp>
#include <neolib/core/vector.hpp>
#include <neolib/core/reference_counted.hpp>
#include <neogfx/hid/i_hid_devices.hpp>

namespace neogfx
{
    class hid_devices : public i_hid_devices
    {
    public:
        define_declared_event(DeviceConnected, device_connected, i_hid_device&)
        define_declared_event(DeviceDisconnected, device_disconnected, i_hid_device&)
    public:
        typedef neolib::vector<neolib::ref_ptr<i_hid_device>> device_list;
    public:
        hid_devices();
        ~hid_devices();
    public:
        void enumerate_devices() override;
        const device_list& devices() const override;
        void add_device(i_hid_device& aDevice) override;
        void remove_device(i_hid_device& aDevice) override;
    public:
        const i_string& product_name(hid_device_class aClass, const hid_device_uuid& aProductId) const override;
    private:
        device_list iDevices;
    };
}