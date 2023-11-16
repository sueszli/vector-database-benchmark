/* Copyright (c) 2022 StoneAtom, Inc. All rights reserved.
   Use is subject to license terms

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1335 USA
*/

#include "system/res_manager.h"

#include "system/configuration.h"
#include "system/mag_memory_policy.h"

namespace Tianmu {
namespace system {

ResourceManager::ResourceManager() {
  // Instantiate policies based on ConfigurationManager settings
  res_manage_policy_ = new MagMemoryPolicy(tianmu_sysvar_servermainheapsize);
}

ResourceManager::~ResourceManager() { delete res_manage_policy_; }

}  // namespace system
}  // namespace Tianmu
