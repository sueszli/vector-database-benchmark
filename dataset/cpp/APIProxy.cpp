#include "utility/String.hpp"

#include "ScriptRunner.hpp"

#include "PluginLoader.hpp"
#include "APIProxy.hpp"

std::shared_ptr<APIProxy>& APIProxy::get() {
    static auto instance = std::make_shared<APIProxy>();
    return instance;
}

bool APIProxy::add_on_lua_state_created(APIProxy::REFLuaStateCreatedCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    m_on_lua_state_created_cbs.push_back(cb);

    auto& state = ScriptRunner::get()->get_state();

    if (state != nullptr && state->lua().lua_state() != nullptr) {
        cb(state->lua());
    }

    return true;
}

bool APIProxy::add_on_lua_state_destroyed(APIProxy::REFLuaStateDestroyedCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    m_on_lua_state_destroyed_cbs.push_back(cb);
    return true;
}

bool APIProxy::add_on_present(APIProxy::REFOnPresentCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    m_on_present_cbs.push_back(cb);
    return true;
}

bool APIProxy::add_on_pre_application_entry(std::string_view name, REFOnPreApplicationEntryCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    if (name.empty()) {
        return false;
    }

    const auto name_hash = utility::hash(name);

    m_on_pre_application_entry_cbs[name_hash].push_back(cb);
    return true;
}

bool APIProxy::add_on_post_application_entry(std::string_view name, REFOnPostApplicationEntryCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    if (name.empty()) {
        return false;
    }

    const auto name_hash = utility::hash(name);

    m_on_post_application_entry_cbs[name_hash].push_back(cb);
    return true;
}

bool APIProxy::add_on_device_reset(REFOnDeviceResetCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    m_on_device_reset_cbs.push_back(cb);
    return true;
}

bool APIProxy::add_on_message(REFOnMessageCb cb) {
    std::unique_lock _{m_api_cb_mtx};

    m_on_message_cbs.push_back(cb);
    return true;
}

void APIProxy::on_lua_state_created(sol::state& state) {
    std::shared_lock _{m_api_cb_mtx};

    for (auto& cb : m_on_lua_state_created_cbs) {
        try {
            cb(state.lua_state());
        } catch(...) {
            spdlog::error("[APIProxy] Exception occurred in on_lua_state_created callback; one of the plugins has an error.");
        }
    }
}

void APIProxy::on_lua_state_destroyed(sol::state& state) {
    std::shared_lock _{m_api_cb_mtx};

    for (auto& cb : m_on_lua_state_destroyed_cbs) {
        try {
            cb(state.lua_state());
        } catch(...) {
            spdlog::error("[APIProxy] Exception occurred in on_lua_state_destroyed callback; one of the plugins has an error.");
        }
    }
}

void APIProxy::on_present() {
    std::shared_lock _{m_api_cb_mtx};

    reframework::g_renderer_data.renderer_type = (int)g_framework->get_renderer_type();
    
    if (reframework::g_renderer_data.renderer_type == REFRAMEWORK_RENDERER_D3D11) {
        auto& d3d11 = g_framework->get_d3d11_hook();

        reframework::g_renderer_data.device = d3d11->get_device();
        reframework::g_renderer_data.swapchain = d3d11->get_swap_chain();
    } else if (reframework::g_renderer_data.renderer_type == REFRAMEWORK_RENDERER_D3D12) {
        auto& d3d12 = g_framework->get_d3d12_hook();

        reframework::g_renderer_data.device = d3d12->get_device();
        reframework::g_renderer_data.swapchain = d3d12->get_swap_chain();
        reframework::g_renderer_data.command_queue = d3d12->get_command_queue();
    }

    for (auto&& cb : m_on_present_cbs) {
        try {
            cb();
        } catch(...) {
            spdlog::error("[APIProxy] Exception occurred in on_present callback; one of the plugins has an error.");
        }
    }
}

void APIProxy::on_pre_application_entry(void* entry, const char* name, size_t hash) {
    std::shared_lock _{m_api_cb_mtx};

    if (auto it = m_on_pre_application_entry_cbs.find(hash); it != m_on_pre_application_entry_cbs.end()) {
        for (auto&& cb : it->second) {
            try {
                cb();
            } catch(...) {
                spdlog::error("[APIProxy] Exception occurred in on_pre_application_entry callback ({}); one of the plugins has an error.", name);
            }
        }
    }
}

void APIProxy::on_application_entry(void* entry, const char* name, size_t hash) {
    std::shared_lock _{m_api_cb_mtx};

    if (auto it = m_on_post_application_entry_cbs.find(hash); it != m_on_post_application_entry_cbs.end()) {
        for (auto&& cb : it->second) {
            try {
                cb();
            } catch(...) {
                spdlog::error("[APIProxy] Exception occurred in on_post_application_entry callback ({}); one of the plugins has an error.", name);
            }
        }
    }
}

void APIProxy::on_device_reset() {
    std::shared_lock _{m_api_cb_mtx};

    for (auto&& cb : m_on_device_reset_cbs) {
        try {
            cb();
        } catch(...) {
            spdlog::error("[APIProxy] Exception occurred in on_device_reset callback; one of the plugins has an error.");
        }
    }
}

bool APIProxy::on_message(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    std::shared_lock _{m_api_cb_mtx};

    for (auto&& cb : m_on_message_cbs) {
        try {
            if (!cb(hwnd, msg, wparam, lparam)) {
                return false;
            }
        } catch(...) {
            spdlog::error("[APIProxy] Exception occurred in on_message callback; one of the plugins has an error.");
            continue;
        }
    }

    return true;
}