/*
 * linux_cursor_settings.c
 * Copyright (C) 2021 Kovid Goyal <kovid at kovidgoyal.net>
 *
 * Distributed under terms of the GPL3 license.
 */

#include "linux_desktop_settings.h"
#include <stdlib.h>
#include <strings.h>
#include <string.h>

#define DESKTOP_SERVICE "org.freedesktop.portal.Desktop"
#define DESKTOP_PATH "/org/freedesktop/portal/desktop"
#define DESKTOP_INTERFACE "org.freedesktop.portal.Settings"
#define GNOME_DESKTOP_NAMESPACE "org.gnome.desktop.interface"
#define FDO_DESKTOP_NAMESPACE "org.freedesktop.appearance"
#define FDO_APPEARANCE_KEY "color-scheme"


static char theme_name[128] = {0};
static int theme_size = -1;
static uint32_t appearance = 0;
static bool is_gnome = false;
static bool cursor_theme_changed = false;

int
glfw_current_system_color_theme(void) {
    return appearance;
}

#define HANDLER(name) static void name(DBusMessage *msg, const char* errmsg, void *data) { \
    (void)data; \
    if (errmsg) { \
        _glfwInputError(GLFW_PLATFORM_ERROR, "%s: failed with error: %s", #name, errmsg); \
        return; \
    }

static void
process_fdo_setting(const char *key, DBusMessageIter *value) {
    if (strcmp(key, FDO_APPEARANCE_KEY) == 0) {
        if (dbus_message_iter_get_arg_type(value) == DBUS_TYPE_UINT32) {
            dbus_message_iter_get_basic(value, &appearance);
            if (appearance > 2) appearance = 0;
        }
    }
}

static void
process_gnome_setting(const char *key, DBusMessageIter *value) {
    if (strcmp(key, "cursor-size") == 0) {
        if (dbus_message_iter_get_arg_type(value) == DBUS_TYPE_INT32) {
            int32_t sz;
            dbus_message_iter_get_basic(value, &sz);
            if (sz > 0 && sz != theme_size) {
                theme_size = sz;
                cursor_theme_changed = true;
            }
        }
    } else if (strcmp(key, "cursor-theme") == 0) {
        if (dbus_message_iter_get_arg_type(value) == DBUS_TYPE_STRING) {
            const char *name;
            dbus_message_iter_get_basic(value, &name);
            if (name) {
                strncpy(theme_name, name, sizeof(theme_name) - 1);
                cursor_theme_changed = true;
            }
        }
    }
}

static void
process_settings_dict(DBusMessageIter *array_iter, void(process_setting)(const char *, DBusMessageIter*)) {
    DBusMessageIter item_iter, value_iter;
    while (dbus_message_iter_get_arg_type(array_iter) == DBUS_TYPE_DICT_ENTRY) {
        dbus_message_iter_recurse(array_iter, &item_iter);
        if (dbus_message_iter_get_arg_type(&item_iter) == DBUS_TYPE_STRING) {
            const char *key;
            dbus_message_iter_get_basic(&item_iter, &key);
            if (dbus_message_iter_next(&item_iter) && dbus_message_iter_get_arg_type(&item_iter) == DBUS_TYPE_VARIANT) {
                dbus_message_iter_recurse(&item_iter, &value_iter);
                process_setting(key, &value_iter);
            }
        }
        if (!dbus_message_iter_next(array_iter)) break;
    }
}

HANDLER(process_desktop_settings)
    cursor_theme_changed = false;
    DBusMessageIter root, array, item, settings;
    dbus_message_iter_init(msg, &root);
#define die(...) { _glfwInputError(GLFW_PLATFORM_ERROR, __VA_ARGS__); return; }
    if (dbus_message_iter_get_arg_type(&root) != DBUS_TYPE_ARRAY) die("Reply to request for desktop settings is not an array");
    dbus_message_iter_recurse(&root, &array);
    while (dbus_message_iter_get_arg_type(&array) == DBUS_TYPE_DICT_ENTRY) {
        dbus_message_iter_recurse(&array, &item);
        if (dbus_message_iter_get_arg_type(&item) == DBUS_TYPE_STRING) {
            const char *namespace;
            dbus_message_iter_get_basic(&item, &namespace);
            if (dbus_message_iter_next(&item) && dbus_message_iter_get_arg_type(&item) == DBUS_TYPE_ARRAY) {
                dbus_message_iter_recurse(&item, &settings);
                if (strcmp(namespace, FDO_DESKTOP_NAMESPACE) == 0) {
                    process_settings_dict(&settings, process_fdo_setting);
                } else if (is_gnome && strcmp(namespace, GNOME_DESKTOP_NAMESPACE) == 0) {
                    process_settings_dict(&settings, process_gnome_setting);
                }
            }
        }
        if (!dbus_message_iter_next(&array)) break;
    }
#undef die
    if (cursor_theme_changed) _glfwPlatformChangeCursorTheme();
}

#undef HANDLER

static bool
read_desktop_settings(DBusConnection *session_bus) {
    DBusMessage *msg = dbus_message_new_method_call(DESKTOP_SERVICE, DESKTOP_PATH, DESKTOP_INTERFACE, "ReadAll");
    if (!msg) return false;
    DBusMessageIter iter, array_iter;
    dbus_message_iter_init_append(msg, &iter);
    if (!dbus_message_iter_open_container(&iter, DBUS_TYPE_ARRAY, "s", &array_iter)) { dbus_message_unref(msg); return false; }
    if (!dbus_message_iter_close_container(&iter, &array_iter)) { dbus_message_unref(msg); return false; }
    bool ok = call_method_with_msg(session_bus, msg, DBUS_TIMEOUT_USE_DEFAULT, process_desktop_settings, NULL);
    dbus_message_unref(msg);
    return ok;
}

void
glfw_current_cursor_theme(const char **theme, int *size) {
    *theme = theme_name[0] ? theme_name : NULL;
    *size =  (theme_size > 0 && theme_size < 2048) ? theme_size : 32;
}

static void
get_cursor_theme_from_env(void) {
    const char *q = getenv("XCURSOR_THEME");
    if (q) strncpy(theme_name, q, sizeof(theme_name)-1);
    const char *env = getenv("XCURSOR_SIZE");
    theme_size = 32;
    if (env) {
        const int retval = atoi(env);
        if (retval > 0 && retval < 2048) theme_size = retval;
    }
}

static void
on_color_scheme_change(DBusMessage *message) {
    DBusMessageIter iter[2];
    dbus_message_iter_init (message, &iter[0]);
    int current_type;
    while ((current_type = dbus_message_iter_get_arg_type (&iter[0])) != DBUS_TYPE_INVALID) {
        if (current_type == DBUS_TYPE_VARIANT) {
            dbus_message_iter_recurse(&iter[0], &iter[1]);
            if (dbus_message_iter_get_arg_type(&iter[1]) == DBUS_TYPE_UINT32) {
                uint32_t val = 0;
                dbus_message_iter_get_basic(&iter[1], &val);
                if (val > 2) val = 0;
                if (val != appearance) {
                    appearance = val;
                    if (_glfw.callbacks.system_color_theme_change) {
                        _glfw.callbacks.system_color_theme_change(appearance);
                    }
                }
            }
            break;
        }
        dbus_message_iter_next(&iter[0]);
    }
}

static DBusHandlerResult
setting_changed(DBusConnection *conn UNUSED, DBusMessage *msg, void *user_data UNUSED) {
    /* printf("session_bus settings_changed invoked interface: %s member: %s\n", dbus_message_get_interface(msg), dbus_message_get_member(msg)); */
    if (dbus_message_is_signal(msg, DESKTOP_INTERFACE, "SettingChanged")) {
        const char *namespace = NULL, *key = NULL;
        if (glfw_dbus_get_args(msg, "Failed to get namespace and key from SettingChanged notification signal", DBUS_TYPE_STRING, &namespace, DBUS_TYPE_STRING, &key, DBUS_TYPE_INVALID)) {
            if (strcmp(namespace, FDO_DESKTOP_NAMESPACE) == 0) {
                if (strcmp(key, FDO_APPEARANCE_KEY) == 0) {
                    on_color_scheme_change(msg);
                }
            }
        }

    }
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}


void
glfw_initialize_desktop_settings(void) {
    get_cursor_theme_from_env();
    const char *desktop = getenv("XDG_CURRENT_DESKTOP");
    is_gnome = desktop && strstr(desktop, "GNOME");
    DBusConnection *session_bus = glfw_dbus_session_bus();
    if (session_bus) {
        if (!read_desktop_settings(session_bus)) _glfwInputError(GLFW_PLATFORM_ERROR, "Failed to read desktop settings, make sure you have the desktop portal running.");
        dbus_bus_add_match(session_bus, "type='signal',interface='" DESKTOP_INTERFACE "',member='SettingChanged'", NULL);
        dbus_connection_add_filter(session_bus, setting_changed, NULL, NULL);
    }
}
