#pragma once

// TODO: Rename folder to data
#define ROOT_LOCATION								"../../../FlexEngine/"

// Game resource files
#define RESOURCE_DIRECTORY							ROOT_LOCATION "resources/"
// Config files which are included in shipping builds
#define CONFIG_DIRECTORY							ROOT_LOCATION "config/"
// Cached files which are not shipped in builds
#define SAVED_DIRECTORY								ROOT_LOCATION "saved/"

#define MESH_DIRECTORY								RESOURCE_DIRECTORY "meshes/"
#define TEXTURE_DIRECTORY							RESOURCE_DIRECTORY "textures/"
#define ICON_DIRECTORY								TEXTURE_DIRECTORY "icons/"
#define SFX_DIRECTORY								RESOURCE_DIRECTORY "audio/"
#define FONT_DIRECTORY								RESOURCE_DIRECTORY "fonts/"
#define SHADER_SOURCE_DIRECTORY						RESOURCE_DIRECTORY "shaders/"
#define PREFAB_DIRECTORY							RESOURCE_DIRECTORY "prefabs/"
#define SCENE_DEFAULT_DIRECTORY						RESOURCE_DIRECTORY "scenes/default/"
#define SCENE_SAVED_DIRECTORY						RESOURCE_DIRECTORY "scenes/saved/"
#define SCRIPTS_DIRECTORY							RESOURCE_DIRECTORY "scripts/"
#define MATERIALS_DIRECTORY							RESOURCE_DIRECTORY "materials/"
#define PARTICLE_SYSTEMS_DIRECTORY					RESOURCE_DIRECTORY "particle_systems/"

#define FONT_SDF_DIRECTORY							SAVED_DIRECTORY "fonts/"
#define SCREENSHOT_DIRECTORY						SAVED_DIRECTORY "screenshots/"
#define COMPILED_SHADERS_DIRECTORY					SAVED_DIRECTORY "spv/"
#define SAVE_FILE_DIRECTORY							SAVED_DIRECTORY "save_files/"

#define DEBUG_OVERLAY_NAMES_LOCATION				CONFIG_DIRECTORY "debug_overlay_names.json"
#define FONT_DEFINITION_LOCATION					CONFIG_DIRECTORY "fonts.json"
#define GAME_OBJECT_TYPES_LOCATION					CONFIG_DIRECTORY "game_object_types.txt"
#define INPUT_BINDINGS_LOCATION						CONFIG_DIRECTORY "input_bindings.json"
#define UI_PLAYER_QUICK_ACCESS_LOCATION				CONFIG_DIRECTORY "player_quick_access_ui.json"
#define UI_PLAYER_INVENTORY_LOCATION				CONFIG_DIRECTORY "player_inventory_ui.json"
#define UI_WEARABLES_INVENTORY_LOCATION				CONFIG_DIRECTORY "wearables_inventory_ui.json"
#define UI_MINER_INVENTORY_LOCATION					CONFIG_DIRECTORY "miner_inventory_ui.json"
#define RENDERER_SETTINGS_LOCATION					CONFIG_DIRECTORY "renderer_settings.json"
#define SHADER_SPECIALIZATION_CONSTANTS_LOCATION	CONFIG_DIRECTORY "shader_specialization_constants.json"
#define SPECIALIZATION_CONSTANTS_LOCATION			CONFIG_DIRECTORY "specialization_constants.json"
#define UI_SETTINGS_LOCATION						CONFIG_DIRECTORY "ui_settings.json"
#define PARTICLE_PARAMETER_TYPES_LOCATION			CONFIG_DIRECTORY "particle_parameter_types.json"

#define COMMON_CONFIG_LOCATION						SAVED_DIRECTORY "common.json"
#define BOOTUP_TIMES_LOCATION						SAVED_DIRECTORY "bootup_times.log"
#define RENDERDOC_LOCATION							SAVED_DIRECTORY "renderdoc.json"
#define IMGUI_INI_LOCATION							SAVED_DIRECTORY "imgui.ini"
#define IMGUI_LOG_LOCATION							SAVED_DIRECTORY "imgui.log"
#define UI_WINDOW_CACHE_LOCATION					SAVED_DIRECTORY "ui_window_cache.json"
#define SHADER_CHECKSUM_LOCATION					SAVED_DIRECTORY "vk_shader_checksum.dat"
#define WINDOW_CONFIG_LOCATION						SAVED_DIRECTORY "window_settings.json"
#define PLAYER_CONFIG_LOCATION						SAVED_DIRECTORY "player_settings.json"

#define USER_INVENTORY_LOCATION						SAVE_FILE_DIRECTORY "user_inventory.json"
