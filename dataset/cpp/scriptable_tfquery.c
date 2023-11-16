/*
    DeaDBeeF -- the music player
    Copyright (C) 2009-2023 Oleksiy Yakovenko and other contributors

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

    3. This notice may not be removed or altered from any source distribution.
*/

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <jansson.h>
#include "scriptable_tfquery.h"

#define trace(...) \
    { deadbeef->log_detailed (&plugin->plugin, 0, __VA_ARGS__); }

static DB_functions_t *deadbeef;
static DB_mediasource_t *plugin;

static const char _default_config[] =
    "{\"queries\":["
    "{\"name\":\"Albums\",\"items\":["
    "\"$if2(%album artist%,\\\\<?\\\\>) - [\\\\[Disc %disc%\\\\] ]$if2(%album%,\\\\<?\\\\>)\","
    "\"[%tracknumber%. ]%title%\""
    "]},"
    "{\"name\":\"Artists\",\"items\":["
    "\"$if2(%album artist%,\\\\<?\\\\>)\","
    "\"$if2(%album artist%,\\\\<?\\\\>) - [\\\\[Disc %disc%\\\\] ]$if2(%album%,\\\\<?\\\\>)\","
    "\"[%tracknumber%. ]%title%\""
    "]},"
    "{\"name\":\"Genres\",\"items\":["
    "\"$if2(%genre%,\\\\<?\\\\>)\","
    "\"$if2(%album artist%,\\\\<?\\\\>) - [\\\\[Disc %disc%\\\\] ]$if2(%album%,\\\\<?\\\\>)\","
    "\"[%tracknumber%. ]%title%\""
    "]},"
    "{\"name\":\"Folders\",\"items\":["
    "\"%folder_tree%\","
    "\"[%tracknumber%. ]%title%\""
    "]}"
    "]}";

// structure:
// - presetRoot
//    - presets: List<Preset>
//       - preset: {
//             name: String
//             tfStrings: List<String>
//       }

static scriptableStringListItem_t *
_itemNames (scriptableItem_t *item) {
    scriptableStringListItem_t *newPreset = scriptableStringListItemAlloc ();
    newPreset->str = strdup ("New");

    scriptableStringListItem_t *albums = scriptableStringListItemAlloc ();
    albums->str = strdup ("Albums");

    scriptableStringListItem_t *artists = scriptableStringListItemAlloc ();
    artists->str = strdup ("Artists");

    scriptableStringListItem_t *genres = scriptableStringListItemAlloc ();
    genres->str = strdup ("Genres");

    scriptableStringListItem_t *folders = scriptableStringListItemAlloc ();
    folders->str = strdup ("Folders");

    newPreset->next = albums;
    albums->next = artists;
    artists->next = genres;
    genres->next = folders;

    return newPreset;
}

static scriptableStringListItem_t *
_itemTypes (scriptableItem_t *item) {
    return _itemNames (item);
}

static int
_presetSave (scriptableItem_t *item) {
    return 0;
}

static int
_presetUpdateItem (scriptableItem_t *item) {
    return scriptableItemSave (item);
}

static scriptableStringListItem_t *
_presetItemNames (scriptableItem_t *item) {
    scriptableStringListItem_t *s = scriptableStringListItemAlloc ();
    s->str = strdup ("TFQueryPresetItem");
    return s;
}

static scriptableStringListItem_t *
_presetItemTypes (scriptableItem_t *item) {
    scriptableStringListItem_t *s = scriptableStringListItemAlloc ();
    s->str = strdup ("TFQueryPresetItem");
    return s;
}

static scriptableItem_t *
_presetCreateItemOfType (scriptableItem_t *root, const char *type) {
    // type is ignored, since there's only one item type
    scriptableItem_t *item = scriptableItemAlloc ();
    scriptableItemSetPropertyValueForKey (item, "", "name");
    return item;
}

static const char *
_readonlyPrefix (scriptableItem_t *item) {
    return "[Built-in] ";
}

static const char *
_presetPbIdentifier (scriptableItem_t *item) {
    return "deadbeef.medialib.tfstring";
}

static int
_loadPreset (scriptableItem_t *scriptableQuery, json_t *query, scriptableItem_t *root) {
    json_t *name = json_object_get (query, "name");
    if (!json_is_string (name)) {
        return -1;
    }

    json_t *items = json_object_get (query, "items");
    if (!json_is_array (items)) {
        return -1;
    }

    size_t item_count = json_array_size (items);

    // validate
    for (size_t item_index = 0; item_index < item_count; item_index++) {
        json_t *item = json_array_get (items, item_index);
        if (!json_is_string (item)) {
            return -1;
        }
    }

    scriptableItemSetPropertyValueForKey (scriptableQuery, json_string_value (name), "name");

    for (size_t item_index = 0; item_index < item_count; item_index++) {
        json_t *item = json_array_get (items, item_index);

        scriptableItem_t *scriptableItem = scriptableItemAlloc ();
        scriptableItemSetPropertyValueForKey (scriptableItem, json_string_value (item), "name");
        scriptableItemAddSubItem (scriptableQuery, scriptableItem);
    }

    return 0;
}

static int
_resetPresetNamed (scriptableItem_t *item, scriptableItem_t *root, const char *presetName) {
    int res = -1;

    json_error_t error;
    json_t *json = json_loads (_default_config, 0, &error);

    if (json == NULL) {
        return -1;
    }

    json_t *queries = json_object_get (json, "queries");
    if (queries == NULL) {
        goto error;
    }

    if (!json_is_array (queries)) {
        goto error;
    }

    scriptableItemFlagsAdd (root, SCRIPTABLE_FLAG_IS_LOADING);

    size_t count = json_array_size (queries);

    for (size_t i = 0; i < count; i++) {
        json_t *query = json_array_get (queries, i);
        if (!json_is_object (query)) {
            goto error;
        }

        json_t *name = json_object_get (query, "name");
        if (!json_is_string (name)) {
            goto error;
        }

        if (strcmp (json_string_value (name), presetName)) {
            continue;
        }

        scriptableItemFlagsAdd (item, SCRIPTABLE_FLAG_IS_LOADING);

        for (;;) {
            scriptableItem_t *child = scriptableItemChildren (item);
            if (child == NULL) {
                break;
            }
            scriptableItemRemoveSubItem (item, child);
        }

        if (-1 == _loadPreset (item, query, root)) {
            scriptableItemFlagsRemove (item, SCRIPTABLE_FLAG_IS_LOADING);
            goto error;
        }

        scriptableItemFlagsRemove (item, SCRIPTABLE_FLAG_IS_LOADING);

        break;
    }

    res = 0;

error:
    scriptableItemFlagsRemove (root, SCRIPTABLE_FLAG_IS_LOADING);

    if (json != NULL) {
        json_delete (json);
    }

    return res;
}

static int
_resetPreset (scriptableItem_t *item) {
    scriptableItem_t *root = scriptableItemParent (item);
    return _resetPresetNamed (item, root, scriptableItemPropertyValueForKey (item, "name"));
}

static scriptableOverrides_t _presetCallbacks = {
    .readonlyPrefix = _readonlyPrefix,
    .save = _presetSave,
    .didUpdateItem = _presetUpdateItem,
    .pasteboardItemIdentifier = _presetPbIdentifier,
    .factoryItemNames = _presetItemNames,
    .factoryItemTypes = _presetItemTypes,
    .createItemOfType = _presetCreateItemOfType,
    .reset = _resetPreset,
};

static scriptableItem_t *
_createBlankPreset (void) {
    scriptableItem_t *item = scriptableItemAlloc ();
    scriptableItemFlagsSet (item, SCRIPTABLE_FLAG_IS_LIST | SCRIPTABLE_FLAG_IS_REORDABLE | SCRIPTABLE_FLAG_CAN_RENAME | SCRIPTABLE_FLAG_CAN_RESET | SCRIPTABLE_FLAG_ALLOW_NON_UNIQUE_KEYS);
    scriptableItemSetOverrides (item, &_presetCallbacks);
    return item;
}

static scriptableItem_t *
_createPreset (scriptableItem_t *root, const char *type) {
    // type is ignored, since there's only one preset type
    scriptableItem_t *item = _createBlankPreset ();
    if (!strcmp (type, "New")) {
        scriptableItemSetUniqueNameUsingPrefixAndRoot (item, "New Preset", root);
    }
    else {
        _resetPresetNamed (item, root, type);
        scriptableItemSetUniqueNameUsingPrefixAndRoot (item, type, root);
    }
    return item;
}

static const char *
_rootPbIdentifier (scriptableItem_t *item) {
    return "deadbeef.medialib.tfquery";
}

static int
_saveRoot (scriptableItem_t *root) {
    int res = -1;

    json_t *json = json_object ();

    json_t *queries = json_array ();
    for (scriptableItem_t *query = scriptableItemChildren (root); query; query = scriptableItemNext (query)) {
        json_t *jsonQuery = json_object ();
        const char *name = scriptableItemPropertyValueForKey (query, "name");
        json_object_set (jsonQuery, "name", json_string (name));

        json_t *jsonItems = json_array ();
        for (scriptableItem_t *item = scriptableItemChildren (query); item; item = scriptableItemNext (item)) {
            const char *itemName = scriptableItemPropertyValueForKey (item, "name");
            json_array_append (jsonItems, json_string (itemName));
        }

        json_object_set (jsonQuery, "items", jsonItems);
        json_array_append (queries, jsonQuery);
    }

    json_object_set (json, "queries", queries);

    char *data = json_dumps (json, JSON_COMPACT);
    if (data == NULL) {
        goto error;
    }
    deadbeef->conf_set_str ("medialib.tfqueries", data);
    deadbeef->conf_save ();
    free (data);

    res = 0;
error:

    json_delete (json);

    return res;
}

static int
_resetRoot (scriptableItem_t *root) {
    deadbeef->conf_remove_items ("medialib.tfqueries");
    scriptableTFQueryLoadPresets (root);
    return 0;
}

static scriptableOverrides_t _rootCallbacks = {
    .pasteboardItemIdentifier = _rootPbIdentifier,
    .factoryItemNames = _itemNames,
    .factoryItemTypes = _itemTypes,
    .createItemOfType = _createPreset,
    .save = _saveRoot,
    .reset = _resetRoot,
};

scriptableItem_t *
scriptableTFQueryRootCreate (void) {
    // top level node: list of presets
    scriptableItem_t *root = scriptableItemAlloc ();
    scriptableItemSetOverrides (root, &_rootCallbacks);
    scriptableItemFlagsSet (root, SCRIPTABLE_FLAG_IS_REORDABLE | SCRIPTABLE_FLAG_CAN_RENAME | SCRIPTABLE_FLAG_CAN_RESET | SCRIPTABLE_FLAG_IS_LIST);

    scriptableItemSetPropertyValueForKey (root, "Medialib Query Presets", "name");
    return root;
}

int
scriptableTFQueryLoadPresets (scriptableItem_t *root) {
    int res = -1;
    char *buffer = calloc (1, 20000);
    deadbeef->conf_get_str ("medialib.tfqueries", NULL, buffer, 20000);

    json_error_t error;
    json_t *json = json_loads (buffer, 0, &error);
    free (buffer);

    if (json == NULL) {
        json = json_loads (_default_config, 0, &error);
    }

    if (json == NULL) {
        return -1;
    }

    json_t *queries = json_object_get (json, "queries");
    if (queries == NULL) {
        goto error;
    }

    if (!json_is_array (queries)) {
        goto error;
    }

    scriptableItemFlagsAdd (root, SCRIPTABLE_FLAG_IS_LOADING);

    scriptableItem_t *item;
    while ((item = scriptableItemChildren (root))) {
        scriptableItemRemoveSubItem (root, item);
    }

    size_t count = json_array_size (queries);

    for (size_t i = 0; i < count; i++) {
        json_t *query = json_array_get (queries, i);
        if (!json_is_object (query)) {
            goto error;
        }

        // create
        scriptableItem_t *scriptableQuery = _createBlankPreset ();
        scriptableItemFlagsAdd (scriptableQuery, SCRIPTABLE_FLAG_IS_LOADING);
        int res = _loadPreset (scriptableQuery, query, root);
        if (res == -1) {
            scriptableItemFree (scriptableQuery);
            scriptableItemFlagsRemove (scriptableQuery, SCRIPTABLE_FLAG_IS_LOADING);
            goto error;
        }
        scriptableItemFlagsRemove (scriptableQuery, SCRIPTABLE_FLAG_IS_LOADING);
        scriptableItemAddSubItem (root, scriptableQuery);
    }

    res = 0;
error:
    scriptableItemFlagsRemove (root, SCRIPTABLE_FLAG_IS_LOADING);

    if (json != NULL) {
        json_delete (json);
    }

    return res;
}

void
ml_scriptable_init (DB_functions_t *_deadbeef, DB_mediasource_t *_plugin, scriptableItem_t *root) {
    deadbeef = _deadbeef;
    plugin = _plugin;
    int tf_query_result = scriptableTFQueryLoadPresets (root);
    if (tf_query_result == -1) {
        trace ("medialib: Failed to load tf query presets");
    }
}

void
ml_scriptable_deinit (void) {
}
