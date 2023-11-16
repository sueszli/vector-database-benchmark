// Palanteer viewer
// Copyright (C) 2021, Damien Feneyrou <dfeneyrou@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


// This file implements the core of the viewer application.

// System
#include <cinttypes>
#include <cmath>
#include <algorithm>
#include <ctime>

// External
#include "imgui.h"
#include "imgui_internal.h" // For the DockBuilder API (alpha) + title bar tooltip
#include "palanteer.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ASSERT(x) plAssert(x)
#include "stb_image.h"      // To load the icon

// Internal
#include "bsOs.h"
#include "cmCnx.h"
#include "cmLiveControl.h"
#include "cmRecording.h"
#include "vwConst.h"
#include "vwMain.h"
#include "vwPlatform.h"
#include "vwFileDialog.h"
#include "vwConfig.h"


#define DOCKSPACE_FLAGS  ImGuiDockNodeFlags_PassthruCentralNode

// Exported Palanteer icon file (2387 bytes) using binary_to_compressed_c.cpp  from Dear ImGui
static const unsigned int icon_size = 2387;
static const unsigned int icon_data[2388/4] =
{
    0x474e5089, 0x0a1a0a0d, 0x0d000000, 0x52444849, 0x20000000, 0x20000000, 0x00000608, 0x7a7a7300, 0x090000f4, 0x4144491a, 0xedc35854, 0x5c6c7997,
    0xbf8715d5, 0xecde66f7, 0xb1e3c59e, 0x8c99db1d, 0xd8b3b1f7, 0x821b8921, 0x88201642, 0x942069b4, 0x689155a6, 0x542852d5, 0x5a1a8a95, 0x29a54ada,
    0x4a8454aa, 0x22806d0b, 0xaca2a952, 0xc852a149, 0xc1205046, 0x89212076, 0xd893c49d, 0xc7631c71, 0xf664f19e, 0xf7bcde7d, 0x9484c7fa, 0x8a82e842,
    0xf48f4ffa, 0xaef3aba4, 0xbbf4f9ee, 0xae7b9ee7, 0xb9b0a7c0, 0xe035cc56, 0xc712aab8, 0x56ef2b9d, 0x8b6fbadb, 0x149ed953, 0xf850d1be, 0xfe12d74f,
    0x765d89dd, 0x6e563dda, 0x6b4a70eb, 0xcec615bc, 0xa55316d9, 0x5a14e1df, 0xcd92369a, 0xdad15527, 0xaaace33a, 0xb56d1019, 0xeeda84bb, 0x1c97c26d,
    0x002bfe17, 0x7a160e3d, 0xf3cd86dc, 0xec6e6beb, 0x35275d9e, 0x3ad42dbb, 0xf3526166, 0x5614f968, 0xab2e4acd, 0x611e1d30, 0x9a251234, 0xf4f5a951,
    0x96aedce4, 0xadb41617, 0xf07e6cde, 0x0a9ff3e6, 0x531d9d60, 0xf9fba6c1, 0xef86a6c2, 0x4bb47477, 0xf42d8dd5, 0xb2e256d4, 0xd71d105a, 0x8434f440,
    0x26844906, 0x18870a99, 0x89114d0e, 0x7365c987, 0x64881198, 0x7f566c35, 0xfd61d5aa, 0x6fb3a889, 0xfdf5eeb9, 0xe8e295ca, 0xa87ce4af, 0x9439e8dd,
    0x9dadb6d7, 0xdfd6bfef, 0xe6ee1575, 0x6df5fc46, 0x602d5cd8, 0x0ac5046d, 0xcd1029a9, 0xa1489a88, 0x2a2e92a4, 0xa2856524, 0xd1913022, 0x72201748,
    0x3f4574f1, 0xd1d4663c, 0xee5a9df3, 0xfe48227c, 0x1f74b1a3, 0xd2b9bc75, 0xe6c81ee7, 0x35b676f6, 0xd05a6d2e, 0x6696eccd, 0x31f835e6, 0xc707aeb7,
    0x605bd01c, 0x22a90126, 0xcccac16a, 0x78b120c4, 0x8868b19c, 0x632cf173, 0x40ca9809, 0x11660734, 0x56d01d5a, 0x502058e9, 0x9b9d7f9c, 0x5147f657,
    0xa981513e, 0xdb48d6db, 0x5cfefa47, 0x585aed55, 0x2a89a735, 0x4f0e9bb7, 0x8ddf0637, 0x114137a0, 0x1d9a0028, 0x43423e0c, 0xdce8714f, 0x30b11320,
    0x3a4f214a, 0x0d029203, 0x76f4f2a6, 0xd7f5c4a7, 0x757d403c, 0xeb12a53d, 0x23f0c933, 0xdf7d229b, 0x53bf3d33, 0x70c022bb, 0xa33ef274, 0xac637759,
    0x64d10eb2, 0x0da2a3a1, 0x505add43, 0xca02c825, 0x848b0025, 0xc097b87c, 0x842e41e8, 0x328b90a7, 0x256a3a12, 0x64d02988, 0x5fed9583, 0x26c57da0,
    0x01ae8bac, 0x2fe66455, 0x7899110f, 0xd34f1eeb, 0xf3b07df2, 0x14c41faf, 0xee84183f, 0xa3273cd9, 0xd51a7e19, 0x3a0a2819, 0x2a651566, 0x8f8c0b15,
    0xc0328e43, 0xb7d053c9, 0xd106861f, 0x4a706a5b, 0x4c92a43e, 0xbb12b593, 0xe5d3a54e, 0x644d8e60, 0x2d156fa9, 0x06ac6aee, 0x5219d209, 0xc98cbd6f,
    0x68eac360, 0xdf262312, 0x6d9fb5af, 0x814c7cb5, 0x377b0277, 0x5b258b85, 0xc9e13640, 0x21946866, 0x6b17b237, 0x81900af5, 0xbd49a7b3, 0xe3c60bf9,
    0x49b98a39, 0xc7ae1aaa, 0xc86ee4bf, 0x7bdbb007, 0x41dd1fe8, 0x2c2bff97, 0x4d495937, 0x5690a1ef, 0xea12f4d1, 0x5e4867c1, 0xdce86018, 0x1335142b,
    0x36af13c1, 0xf369bc6d, 0x972d81d7, 0xe07d5093, 0xf8d0c749, 0x0b5e67a1, 0xa24b998e, 0xa7a4874a, 0x6150285e, 0xb9dcc669, 0x9a7a6d01, 0xd8fb9ef8,
    0xbb38e675, 0x34310f86, 0xba684d33, 0x23670697, 0x630fb5fe, 0xeb6a5591, 0xc6bea6ae, 0xe2c9518f, 0xd5c6aef0, 0xa3e11773, 0x05c9a3cf, 0xf22f068e,
    0xea31c8c6, 0xca2873ec, 0xaef1ccd3, 0xec2b2ef2, 0x69c5ea7e, 0xf4e01134, 0x27eb99d4, 0xa9aea3e3, 0x13310a6c, 0x23a0b663, 0xecc09f1a, 0x18d1055d,
    0x67bf83e1, 0x5797a067, 0x2671cf86, 0x50b31702, 0xc5cb4050, 0xc29c4dc8, 0xea4ee2bc, 0x8308fdd7, 0xbeb58ad1, 0xafb2db1b, 0xe5ed1b11, 0xe4b3d0d4,
    0x81c39115, 0xa9891817, 0xe34e860c, 0x5cc76cab, 0x543738d3, 0xde5e74f2, 0xae22a882, 0xe412660e, 0x34ea208b, 0x386decea, 0x3c18e395, 0xfac7e49d,
    0x09fc7439, 0x99a58d76, 0x81a8814e, 0x0cc100db, 0x15ba3cec, 0xac6df9ff, 0x84c9b9b0, 0x1496a0cf, 0xfa00a1e2, 0x9e1957a1, 0x2b2b27ec, 0xc55e060c,
    0x70a1f2eb, 0x5f190101, 0x6ccd1795, 0x3b2a1b13, 0xa4f84056, 0x55b4401e, 0x04678d74, 0x5d868a72, 0x43cdbb00, 0xa22561b5, 0xff68c7db, 0x98bc9c4b,
    0x3964fde5, 0xc41a4d17, 0x402f10f2, 0xabe1a0ac, 0xc4ec676a, 0x781dde1e, 0x12ac8c9c, 0x8cb8682b, 0xd383c910, 0xcca8a7a8, 0x1f8afb70, 0x338e8a12,
    0x778c3336, 0xa7780ff9, 0x477b1b6f, 0xe801770f, 0x5c7107df, 0x6427d4b1, 0x28d15501, 0x69ed50c8, 0x89d0d5c5, 0x9c09a626, 0x93e713f8, 0x1640592a,
    0xb242acc1, 0x56e1ae58, 0x37763c09, 0x8e7c9def, 0xa242a4b1, 0xd2054500, 0x41cc5c32, 0x8e82adad, 0x96b8ed59, 0x93a2c46c, 0x34f1bfc8, 0x3b24f8a9,
    0xb1c93d0e, 0x80366978, 0xdbe81f7e, 0x5153090a, 0x881510d6, 0xac6221e7, 0x06bdd3c5, 0xa68a3439, 0x1398b8a4, 0x550d0450, 0x59081f28, 0x9a009303,
    0x07339306, 0x64d36689, 0xa096c864, 0x832482a4, 0xbbd78028, 0x5f459e86, 0xdfada5e4, 0xd917a6c4, 0x0b28ee7f, 0x14336686, 0xba8a2129, 0x7f10500a,
    0x24956a07, 0x868a7496, 0x02a904b1, 0x28fdbc0c, 0x9208e224, 0x8e2e488e, 0x6970b940, 0x41870a97, 0xfcb1a700, 0xec193285, 0x936e1d46, 0x6ae97888,
    0xcaa1e933, 0x35fe7504, 0xc7d33f1c, 0xf18f27cb, 0x11ef23db, 0x3bec6acc, 0x00a8413f, 0x22caf6cd, 0x4a147880, 0x7513d5a2, 0x41acc018, 0xe08104be,
    0xfb039209, 0x7a2ab210, 0x2a4580d1, 0x6569d607, 0x720d0405, 0x2d3a3432, 0xbc084b9c, 0x70c2befe, 0x3767c55c, 0x03767cb1, 0xeb3d6e9d, 0x36a95a5a,
    0xbbd1d8ea, 0x260e5085, 0x41cc1835, 0x3761a0a4, 0xbf986c88, 0x12b9743a, 0x81d51c1a, 0x551e8b2a, 0xce101283, 0x781d9c69, 0x9791c48e, 0x62173970,
    0x054983d5, 0x829f2d49, 0x38965692, 0x0a22cb92, 0x3849932a, 0xc1c38717, 0x9fa72e3c, 0x382ddf4e, 0x7465ae6a, 0xd2a65de4, 0xacb1291a, 0x4ac4029e,
    0x017fab70, 0x08ee7a4b, 0x9970065c, 0x0d757d5c, 0x2089d12b, 0x849e300b, 0xde31d4c0, 0x3695b63e, 0x461c12ef, 0x50604a70, 0xf4d804c1, 0xef2aecd0,
    0x1608aa77, 0x68ac7943, 0xd3f44990, 0x9a76f8be, 0x8c5036ba, 0x432d261c, 0xd5cccda7, 0x02952b0e, 0x1d5045a4, 0xb90a949d, 0x4600cbbe, 0x56d5d941,
    0x0bd25b55, 0x21211450, 0x9070a9c3, 0x4a0989de, 0xea8e99a2, 0x609c5a56, 0x03a343d7, 0x1241abba, 0xc8594245, 0xb464e529, 0x70892048, 0x67219a26,
    0x2eb62074, 0xe790c6a2, 0x64768df3, 0x325f4735, 0x0a92f843, 0x1783e303, 0xba7deddf, 0xe98032ff, 0x5c476bc7, 0x59a0f072, 0xac804502, 0x98496300,
    0xf1ef41c9, 0xfd79b997, 0x6daeb92f, 0xa82ead19, 0x7d4f42b3, 0x4a92853b, 0x917b52ec, 0x2e464229, 0x5eb3411f, 0x80b76cc4, 0x77af02b9, 0x0e8ad635,
    0x6f7fb346, 0x1c2a4cbf, 0xe2f93c99, 0x33bb0386, 0xce1b70ff, 0xd5fcefad, 0x26976f02, 0x41d79069, 0xf7965b28, 0xa38c646c, 0x6f697b03, 0xa547435b,
    0x78804e88, 0x00410276, 0xda2f2b55, 0x58f116e6, 0x9f06cea0, 0xc98adff3, 0x290a1737, 0x0e7ec142, 0xb973be8c, 0x5500d103, 0xe664df24, 0x8ebb1f37,
    0xdaef8b3d, 0xf8b8dddb, 0x15054935, 0x217068c8, 0x1ac2870f, 0x9a7eed8f, 0x8d8264a9, 0xa4b0b6eb, 0xacc74cd6, 0xa1da3445, 0x6c8650c2, 0xeaf9e876,
    0xeb7cf9bc, 0x7a58fb86, 0xea20c3c7, 0x062f7060, 0x92b91747, 0x9d43ed0e, 0xf1e043a7, 0xfd9c2317, 0xdeee8018, 0x18935ab9, 0xd2abb7bd, 0x3203a334,
    0x20eb483a, 0x0958c121, 0x28fdd89e, 0x5b2c12b1, 0x9af33f7c, 0x3573e296, 0xc9ca13a0, 0x64552177, 0xaf02763a, 0xe77a116f, 0x4f3de42c, 0x59ec7a32,
    0x33f70c0e, 0x5d28519d, 0x60c3b002, 0x4fe6c495, 0x70ff09ec, 0xfd6274e6, 0x0bd1e32c, 0x3397d2e9, 0xf0972920, 0x25121114, 0xd4bd82d0, 0xc34659cd,
    0x4845df0d, 0xbf4e08e5, 0x2cc0288d, 0x2e1cf82b, 0x206de75a, 0x0e11f3b0, 0xb1cedf1c, 0x0a9cc389, 0x5f3670cf, 0x468329ce, 0x76b023d0, 0x7fe99247,
    0xacdb96da, 0xebe6dab7, 0xa5629b23, 0x4105fdc8, 0x5ff87d33, 0x46f50695, 0x5bbaf6f8, 0x137c5968, 0x887b23a1, 0xf0a84785, 0x38b76334, 0xa3e81138,
    0x74986477, 0x029d5a11, 0xe31a4307, 0x3134182a, 0x0f356b2a, 0x1ffb0486, 0xe9ed99d3, 0x09d0057f, 0xb1b3e29c, 0xf6cce9b9, 0x443b0fa1, 0x7ff6c7fe,
    0xcd3d2b80, 0x3f9b8026, 0x1f1378a3, 0x510dfc75, 0x20cdea25, 0x00107d79, 0x49000000, 0xae444e45, 0x00826042,
};


// ==============================================================================================
// Main application
// ==============================================================================================

vwMain::vwMain(vwPlatform* platform, int rxPort, const bsString& overrideStoragePath) :
    _platform(platform)
{
    // Initialise some fields
    _idPool.reserve(128);
    _search        .uniqueId = 0x10001; // Fixed IDs non overlapping other windows
    _recordWindow  .uniqueId = 0x10002;
    _catalogWindow .uniqueId = 0x10003;
    _logConsole    .uniqueId = 0x10004;
    _settingsWindow.uniqueId = 0x10005;
    _logConsole.logs.reserve(128);

    // Internals
    _config      = new vwConfig(this, osGetProgramDataPath());
    _storagePath = getConfig().getRecordStoragePath();
    if(!overrideStoragePath.empty()) {
        _storagePath = overrideStoragePath;
        if(_storagePath.back()!=PL_DIR_SEP_CHAR) _storagePath.push_back(PL_DIR_SEP_CHAR);
    }
    _recording = new cmRecording(this, _storagePath, false);
    _clientCnx = new cmCnx(this, rxPort);
    _live      = new cmLiveControl(this, _clientCnx);
    _fileDialogExtStrings       = new vwFileDialog("Update external strings from file",  vwFileDialog::OPEN_FILE,  {"*.txt", "*.*"});
    _fileDialogImport           = new vwFileDialog("Import a record as a file",          vwFileDialog::OPEN_FILE,  {"*.pltraw", "*.*"});
    _fileDialogExportChromeTF   = new vwFileDialog("Export a record as Chrome Trace Format (JSON)", vwFileDialog::SAVE_FILE,  {"*.json", "*.*"});
    _fileDialogExportText       = new vwFileDialog("Export a thread in text", vwFileDialog::SAVE_FILE,  {"*.txt", "*.*"});
    _fileDialogExportLog        = new vwFileDialog("Export logs in text", vwFileDialog::SAVE_FILE,  {"*.log", "*.*"});
    _fileDialogExportPlot       = new vwFileDialog("Export values of a curve (CSV)", vwFileDialog::SAVE_FILE,  {"*.csv", "*.*"});
    _fileDialogExportScreenshot = new vwFileDialog("Export a screenshot as PNG image", vwFileDialog::SAVE_FILE,  {"*.png", "*.*"});
    _fileDialogSelectRecord     = new vwFileDialog("Select the new record storage path", vwFileDialog::SELECT_DIR, {"*.*"});
    vwMain::logToConsole(LOG_INFO, "Record storage path is: %s", _storagePath.toChar());

    // Ensure configuration path exists
    if(!osDirectoryExists(getConfig().getConfigPath())) {
        if(osMakeDir(getConfig().getConfigPath())!=bsDirStatusCode::OK) {
            vwMain::logToConsole(LOG_ERROR, "Unable to create the configuration folder %s", getConfig().getConfigPath().toChar());
        }
    }

    // Ensure record storage path exists
    if(!osDirectoryExists(_storagePath)) {
        if(osMakeDir(_storagePath)!=bsDirStatusCode::OK) {
            vwMain::logToConsole(LOG_ERROR, "Unable to create the record storage folder %s", _storagePath.toChar());
        }
    }

    // Install the icon
    int width=0, height=0;
    u8* pixels = stbi_load_from_memory((const stbi_uc*)icon_data, icon_size, &width, &height, 0, 4);
    osSetIcon(width, height, pixels);  // The array is owned by the OS layer now
    free(pixels);
}


vwMain::~vwMain(void)
{
    plScope("~vwMain");
    delete _clientCnx; // This stops the on-going record, so shall be done before record clearing.
    clearRecord();
    delete _fileDialogExtStrings;
    delete _fileDialogImport;
    delete _fileDialogExportChromeTF;
    delete _fileDialogExportText;
    delete _fileDialogExportLog;
    delete _fileDialogExportPlot;
    delete _fileDialogExportScreenshot;
    delete _fileDialogSelectRecord;
    delete _live;
    delete _recording;
    delete _config;
}


void
vwMain::notifyStart(bool doLoadLastFile)
{
    updateRecordList();

    // Load the last record
    if(doLoadLastFile && !getConfig().getLastLoadedRecordPath().empty()) {
        _msgRecordLoad.t1GetFreeMsg()->recordPath = getConfig().getLastLoadedRecordPath();
        _msgRecordLoad.t1Send();
    }
}


int
vwMain::getDisplayWidth(void)
{
    return _platform->getDisplayWidth();
}


int
vwMain::getDisplayHeight(void)
{
    return _platform->getDisplayHeight();
}


void
vwMain::dirty(void)
{
    _platform->notifyDrawDirty();
}


void
vwMain::beforeDraw(bool doSaveLayout)
{
    _doEnterFullScreen = false;
    _doCreateNewViews  = false;

    // Remove some records if required
    if(!_recordsToDelete.empty()) {
        removeSomeRecords(_recordsToDelete);
        _recordsToDelete.clear();
    }

    // Full screen layout management: save and restore ImGui dockspaces
    if(_nextUniqueIdFullScreen>=-1) {
        if(_uniqueIdFullScreen>=0) { // Fullscreen to normal
            _uniqueIdFullScreen = -1;
            ImGui::LoadIniSettingsFromMemory(_fullScreenLayoutDescr.toChar(), _fullScreenLayoutDescr.size());
        } else if(_nextUniqueIdFullScreen>=0) { // Normal to fullscreen
            _uniqueIdFullScreen = _nextUniqueIdFullScreen;
            _fullScreenLayoutDescr = ImGui::SaveIniSettingsToMemory();
            _doEnterFullScreen = true;
        }
        _nextUniqueIdFullScreen = -2; // Reset the update automata
    }

    else if(!_screenLayoutToApply.windows.empty()) {
        ImGui::LoadIniSettingsFromMemory(_screenLayoutToApply.windows.toChar()); //, _screenLayoutToApply.windows.size());
        _doCreateNewViews = true;
    }

    // Save layout just before exiting or clearing the record
    if(doSaveLayout || _doClearRecord) {
        copyCurrentLayout(getConfig().getCurrentLayout(), (_uniqueIdFullScreen>=0)? _fullScreenLayoutDescr:ImGui::SaveIniSettingsToMemory());
    }

    // Clear current record if required
    if(!_doCreateNewViews && _doClearRecord && !_backgroundComputationInUse) {
        clearRecord();
        _doClearRecord = false;
        if(_actionMode!=LOAD_RECORD) { // No need to wait for loading record. Also we want to keep the state as LOAD_RECORD
            _waitForDisplayRefresh = 2;
        }
    }
    if(_waitForDisplayRefresh>0) {
        _waitForDisplayRefresh--;
        // Waiting for some display is always the end of actions, unless a layout shall be applied
        if(_waitForDisplayRefresh==0 && _screenLayoutToApply.windows.empty()) {
            _actionMode = READY;
            plData("Action mode", plMakeString("Ready"));
        }
    }

    // Snapshot the workspace layout now (ImGui constraint)
    if(!_doSaveTemplateLayoutName.empty()) {
        // Update the template if such name is already in the list
        for(vwConfig::ScreenLayout& tl : getConfig().getTemplateLayouts()) {
            if(tl.name!=_doSaveTemplateLayoutName) continue;
            copyCurrentLayout(tl, (_uniqueIdFullScreen>=0)? _fullScreenLayoutDescr:ImGui::SaveIniSettingsToMemory());
            _doSaveTemplateLayoutName.clear();
            break;
        }
        // Else add a new one
        if(!_doSaveTemplateLayoutName.empty()) {
            getConfig().getTemplateLayouts().push_back({});
            getConfig().getTemplateLayouts().back().name = _doSaveTemplateLayoutName;
            copyCurrentLayout(getConfig().getTemplateLayouts().back(), (_uniqueIdFullScreen>=0)? _fullScreenLayoutDescr:ImGui::SaveIniSettingsToMemory());
            _doSaveTemplateLayoutName.clear();
        }
    }
}


void
vwMain::draw(void)
{
    plScope("draw");

    // Some caching
    _lastMouseMoveDurationUs = _platform->getLastMouseMoveDurationUs();

    // Create the global window
    ImGuiIO& io = ImGui::GetIO();
    const ImU32 flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    if(!ImGui::Begin("App window", NULL, flags | ((_uniqueIdFullScreen>=0)?0:ImGuiWindowFlags_MenuBar))) {
        ImGui::End();
        return;
    }

    // Docking
    ImGuiID mainDockspaceId = ImGui::GetID("MainDockSpace");
    if(!ImGui::DockBuilderGetNode(mainDockspaceId)) {
        ImGui::DockBuilderAddNode(mainDockspaceId, ImGuiDockNodeFlags_DockSpace); // Add root node
        ImGui::DockSpace(mainDockspaceId, ImVec2(0., 0.), DOCKSPACE_FLAGS);
        ImGui::DockBuilderSetNodeSize(mainDockspaceId, ImGui::GetIO().DisplaySize);
    }
    else if(_doEnterFullScreen) {
        // Create the "full screen" layout
        ImGui::DockBuilderRemoveNodeChildNodes(mainDockspaceId); // Remove root and all children
        ImGui::DockSpace(mainDockspaceId, ImVec2(0., 0.), DOCKSPACE_FLAGS);
        ImGui::DockBuilderSetNodeSize(mainDockspaceId, ImGui::GetIO().DisplaySize);
    }
    else {
        // Case no change
        ImGui::DockSpace(mainDockspaceId, ImVec2(0.0f, 0.0f), DOCKSPACE_FLAGS);
    }

    if(_doCreateNewViews) {
        plAssert(!_screenLayoutToApply.windows.empty());
        createLayoutViews(_screenLayoutToApply);
        _screenLayoutToApply.windows.clear();
        _waitForDisplayRefresh = 2; // Will clear the action
    }

    // Messages from other threads
    // ===========================

    // Display an error message (not when a new window will be displayed as this closes modal popups)
    MsgError* error = 0;
    if(_actionMode==READY && (error=_msgRecordErrorDisplay.getReceivedMsg())) {
        if     (error->kind==ERROR_LOAD)   ImGui::OpenPopup("Load error");
        else if(error->kind==ERROR_IMPORT) ImGui::OpenPopup("Import error");
        else                               ImGui::OpenPopup("Error");
        _safeErrorMsg = *error;
        _msgRecordErrorDisplay.releaseMsg();
        _actionMode = ERROR_DISPLAY;
        plData("Action mode", plMakeString("Error display"));
    }

    // Start a record
    cmRecord** recordPtr = 0;
    if(_actionMode==READY && (recordPtr=_msgRecordStarted.getReceivedMsg())) {
        _actionMode = START_RECORD; // Cleared when the new layout is applied and some frames are displayed
        plData("Action mode", plMakeString("Start of record"));
        updateRecordList(); // Populate with updated record list
        findRecord((*recordPtr)->recordPath, _underRecordAppIdx, _underRecordRecIdx);
        if(_underRecordRecIdx==-1) {
            logToConsole(LOG_ERROR, "BUG: file %s (current record) not found...\n", (*recordPtr)->recordPath.toChar());
            _clientCnx->disconnect();
        }
        _forceOpenAppIdx = _underRecordAppIdx;
        _msgRecordStarted.releaseMsg();
        osSetWindowTitle("Palanteer - RECORDING");
        _record = *recordPtr;
        getConfig().notifyNewRecord(_record);
        _screenLayoutToApply = getConfig().getCurrentLayout();
        _recordWindow.isWindowSelected = true;
        _recordWindow.doForceShowLive  = true;
        setFullScreenView(-1);
    }

    // Unfreeze new streams
    int newStreamQty = _newStreamQty;
    if(_record && _underRecordRecIdx>=0 && newStreamQty>_streamQty) {
        for(int streamId=_streamQty; streamId<newStreamQty; ++streamId) {
            _live->remoteSetFreezeMode(streamId, getConfig().getFreezePointEnabled());
        }
        _streamQty = newStreamQty;
    }

    // Live-update a record
    cmRecord::Delta* deltaRecord = 0;
    _liveRecordUpdated = false; // True would invalidate all drawing cache
    if(_actionMode==READY && (deltaRecord=_msgRecordDelta.getReceivedMsg())) {
        plData("Action mode", plMakeString("Delta record"));
        plAssert(_record);
        if(_record->updateFromDelta(deltaRecord)) {
            getConfig().notifyUpdatedRecord(_record);
        }
        _liveRecordUpdated = true;
        _msgRecordDelta.releaseMsg();
        plData("Action mode", plMakeString("Ready"));
    }

    // End a record
    bool* isEndedRecordOkPtr = 0;
    if(_actionMode==READY && (isEndedRecordOkPtr=_msgRecordEnded.getReceivedMsg())) {
        plData("Action mode", plMakeString("End of record"));
        bool isEndedRecordOk = *isEndedRecordOkPtr;
        bsString recordPath = (_underRecordRecIdx>=0)? _cmRecordInfos[_underRecordAppIdx].records[_underRecordRecIdx].path : "";
        updateRecordList(); // Populate with updated record list
        _underRecordAppIdx = -1;
        _underRecordRecIdx = -1;
        _msgRecordEnded.releaseMsg();
        // Reset the delta record
        _msgRecordDelta.getRawData()->reset();
        osSetWindowTitle("Palanteer");
        // Enforce the keeping of only the last N records
        int appIdx, recIdx;
        findRecord(recordPath, appIdx, recIdx);
        if(appIdx>=0) {
            bool keepOnlyLastRecordState;
            int  keepOnlyLastRecordQty;
            getConfig().getKeepOnlyLastNRecord(_cmRecordInfos[appIdx].name, keepOnlyLastRecordState, keepOnlyLastRecordQty);
            if(keepOnlyLastRecordState && keepOnlyLastRecordQty>0) {
                bsVec<bsString> recordsToDelete;
                for(RecordInfos& ri : _cmRecordInfos[appIdx].records)
                    if(ri.nickname[0]==0 && --keepOnlyLastRecordQty<0) _recordsToDelete.push_back(ri.path);
            }
        }
        // Display an error message if needed
        if(!isEndedRecordOk) {
            notifyErrorForDisplay(ERROR_GENERIC, bsString("The recording was interrupted due to detected stream data corruption."));
        }
        plData("Action mode", plMakeString("Ready"));
    }

    // Request for loading (done after "end of record")
    if(_actionMode==READY) {
        _recordLoadSavedMsg = _msgRecordLoad.getReceivedMsg();
        if(_recordLoadSavedMsg) {
            if(_record) _doClearRecord = true;
            _actionMode = LOAD_RECORD;
            plData("Action mode", plMakeString("Load record initiated"));
        }
    }
    if(_actionMode==LOAD_RECORD && _recordLoadSavedMsg && !_record) {
        plData("Action mode", plMakeString("Load record"));
        findRecord(_recordLoadSavedMsg->recordPath, _underDisplayAppIdx, _underDisplayRecIdx);
        if(_underDisplayRecIdx>=0) {
            _forceOpenAppIdx = _underDisplayAppIdx;
            if(loadRecord(_recordLoadSavedMsg->recordPath, _underDisplayAppIdx, _underDisplayRecIdx)) {
                _liveRecordUpdated  = true;
                _recordWindow.isWindowSelected = true;
                // The _actionMode will be cleared after setting the layout and waiting for some frame
            } else {
                _actionMode = READY; // Unable to load the record
            }
        } else {
            _actionMode = READY; // File to load was not found
        }
        _recordLoadSavedMsg = 0;
        _msgRecordLoad.releaseMsg();
    }

    // Handle export automata and per chunk computations
    handleExports();

    // Global record precomputations
    precomputeRecordDisplay();

    // Draw all display components
    _hlHasBeenSet = false;
    if(getConfig().getWindowCatalogVisibility() && _underRecordRecIdx<0) { // No catalog when recording
        drawCatalog();
    }
    if(getConfig().getWindowRecordVisibility() || _underRecordRecIdx>=0)  { // Forced record window when recording
        drawRecord();
    }
    drawMainMenuBar();
    drawTimelines();
    drawMemoryTimelines();
    drawProfiles();
    drawLogs();
    drawTexts();
    drawPlots();
    drawHistograms();
    drawSearch();
    drawAbout();
    drawHelp();
    drawLogConsole();
    drawSettings();
    drawErrorMsg();

    // Handle the font size hotkeys globally
    if(ImGui::GetIO().KeyCtrl) {
        if(ImGui::IsKeyPressed(KC_Add)) {
            getConfig().setFontSize(getConfig().getFontSize()+1);
            _platform->setNewFontSize(getConfig().getFontSize());
            allIsDirty();
        }
        if(ImGui::IsKeyPressed(KC_Subtract)) {
            getConfig().setFontSize(getConfig().getFontSize()-1);
            _platform->setNewFontSize(getConfig().getFontSize());
            allIsDirty();
        }
    }

    if(!_hlHasBeenSet) _hlThreadId = cmConst::MAX_THREAD_QTY; // Reset the highlight if not set anywhere
    ImGui::End(); // End of global window
}


void
vwMain::setScopeHighlight(int threadId, s64 startTimeNs, s64 endTimeNs, int eventFlags, int nestingLevel, u32 nameIdx, bool isMultiple)
{
    _hlHasBeenSet   = true;
    _hlThreadId     = threadId;
    _hlStartTimeNs  = startTimeNs;
    _hlEndTimeNs    = endTimeNs;
    _hlEventFlags   = eventFlags;
    _hlNestingLevel = nestingLevel;
    _hlNameIdx      = nameIdx;
    _hlIsMultiple   = isMultiple;
}


void
vwMain::setScopeHighlight(int threadId, s64 punctualTimeNs, int eventFlags, int nestingLevel, u32 nameIdx)
{
    _hlHasBeenSet   = true;
    _hlThreadId     = threadId;
    _hlStartTimeNs  = punctualTimeNs;
    _hlEndTimeNs    = punctualTimeNs+1;
    _hlEventFlags   = eventFlags;
    _hlNestingLevel = nestingLevel;
    _hlNameIdx      = nameIdx;
    _hlIsMultiple   = false;
}



// =====================================
// Interaction with the client reception
// =====================================

// Called by client reception thread
void
vwMain::notifyNewRemoteBuffer(int streamId, bsVec<u8>& buffer)
{
    _live->storeNewRemoteBuffer(streamId, buffer);
}


// Called by the remote control
void
vwMain::notifyNewFrozenThreadState(int streamId, u64 frozenThreadBitmap)
{
    (void)streamId;
    _frozenThreadBitmap = frozenThreadBitmap;
}


// Called by the remote control
void
vwMain::notifyCommandAnswer(int streamId, plPriv::plRemoteStatus status, const bsString& answer)
{
    (void)streamId; (void)status; (void)answer;
}


// Called by client reception thread
bool
vwMain::notifyRecordStarted(const cmStreamInfo& infos, s64 timeTickOrigin, double tickToNs)
{
    plData("Subaction", plMakeString("Notif record started"));

    // Multistream configuration and record naming
    int isMultiStream = getConfig().isMultiStream();
    bsString finalName = isMultiStream? getConfig().getMultiStreamAppName() : infos.appName;
    finalName.strip();
    if(finalName.empty()) finalName = "Default";

    // Ensure that the record storage repository exists
    if(!osDirectoryExists(_storagePath+finalName)) {
        if(osMakeDir(_storagePath+finalName)!=bsDirStatusCode::OK) {
            plLogError("Error", "unable to create the folder for storing all records");
            notifyErrorForDisplay(ERROR_GENERIC, bsString("Unable to create the folder ")+_storagePath+finalName+
                                  "\nPlease check the write permissions");
            return false;
        }
    }

    // Build the record filename
    char   recordName[256];
    time_t now = time(0);
    tm*    t   = localtime(&now);
    plAssert(t);
    snprintf(recordName, sizeof(recordName), PL_DIR_SEP "rec_%04d-%02d-%02d_%02dh%02dm%02ds",
             1900+t->tm_year, 1+t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    bsString recordFilename = _storagePath + finalName + bsString(recordName) + ".plt";

    // Copy the external string file, if any
    bsString appExtStringsPath;
    getConfig().getExtStringsPath(finalName, appExtStringsPath);
    if(!appExtStringsPath.empty()) {
        osCopyFile(appExtStringsPath, _storagePath + finalName + bsString(recordName) + "_externalStrings");
    }

    // Notify the recording
    bsString errorMsg;
    cmRecord* record = _recording->beginRecord(finalName, infos, timeTickOrigin, tickToNs, isMultiStream,
                                               getConfig().getCacheMBytes(), recordFilename, true, errorMsg);
    if(!record) {
        notifyErrorForDisplay(ERROR_GENERIC, errorMsg);
        osRemoveFile(_storagePath + finalName + bsString(recordName) + "_externalStrings");
        return false;
    }

    // Notify the GUI
    cmRecord** recordPtr = _msgRecordStarted.t1GetFreeMsg();
    if(!recordPtr) {
        delete record;
        return false;  // No message available (which would be very weird...)
    }

    _doClearRecord = true;
    *recordPtr    = record;
    _newStreamQty = 1;
    _streamQty    = 0;
    _msgRecordStarted.t1Send();

    dirty();
    return true;
}


void
vwMain::notifyNewCollectionTick(int streamId)
{
    // Used only in the dynamic library
    (void)streamId;
}


void
vwMain::notifyNewStream(const cmStreamInfo& infos)
{
    _recording->notifyNewStream(infos);
    ++_newStreamQty;
}


void
vwMain::notifyNewThread(int threadId, u64 nameHash)
{
    // Used only in the dynamic library
    (void)threadId; (void)nameHash;
}


void
vwMain::notifyNewElem(u64 nameHash, int elemIdx, int prevElemIdx, int threadId, int flags)
{
    // Used only in the dynamic library
    (void)nameHash; (void)elemIdx; (void)prevElemIdx; (void)threadId; (void)flags;
}


void
vwMain::notifyFilteredEvent(int elemIdx, int flags, u64 nameHash, s64 dateNs, u64 value)
{
    // Used only in the dynamic library
    (void)elemIdx; (void)flags; (void)nameHash; (void)dateNs; (void)value;
}


void
vwMain::notifyInstrumentationError(cmRecord::RecErrorType type, int threadId, u32 filenameIdx, int lineNbr, u32 nameIdx)
{
    // Used only in the dynamic library
    (void)type; (void)threadId; (void)filenameIdx; (void)lineNbr; (void)nameIdx;
}


void
vwMain::notifyNewCli(int streamId, u32 nameIdx, int paramSpecIdx, int descriptionIdx)
{
    (void)streamId; (void)nameIdx; (void)paramSpecIdx; (void)descriptionIdx;
}


void
vwMain::notifyNewString(int streamId, const bsString& newString, u64 hash)
{
    _recording->storeNewString(streamId, newString, hash);
}


bool
vwMain::notifyNewEvents(int streamId, plPriv::EventExt* events, int eventQty, s64 shortDateSyncTick)
{
    return _recording->storeNewEvents(streamId, events, eventQty, shortDateSyncTick);
}


// Called by client reception thread
bool
vwMain::createDeltaRecord(void)
{
    // Merge the new record parts
    cmRecord::Delta* delta = _msgRecordDelta.t1GetFreeMsg();
    if(!delta) return false;
    plData("Subaction", plMakeString("Notif delta record creation"));
    _recording->createDeltaRecord(delta);
    _msgRecordDelta.t1Send();
    dirty();
    return true;
}


// Called by client reception thread
void
vwMain::notifyRecordEnded(bool isRecordOk)
{
    plData("Subaction", plMakeString("Notif record ended"));
    // Stop the recording
    _recording->endRecord();

    // Request consecutive load after end of record
    _msgRecordLoad.t1GetFreeMsg()->recordPath = _recording->getRecordPath(); // The path of the last recorded record

     // Send the end record message, with the status as a parameter
    *(_msgRecordEnded.t1GetFreeMsg()) = isRecordOk;
    _msgRecordEnded.t1Send();

    if(isRecordOk) {
        _msgRecordLoad.t1Send();  // Send load request
    }
    dirty();
}


// Called by any thread
void
vwMain::notifyErrorForDisplay(cmErrorKind kind, const bsString& errorMsg)
{
    MsgError* error = _msgRecordErrorDisplay.t1GetFreeMsg();
    if(!error) return;
    logToConsole(LOG_ERROR, errorMsg);
    error->kind = kind;
    error->msg  = errorMsg;
    _msgRecordErrorDisplay.t1Send();
    dirty();
}


// ===========================
// View record API
// ===========================

bool
vwMain::loadRecord(const bsString& recordPath, int appIdx, int recIdx)
{
    bsString errorMsg;

    logToConsole(LOG_INFO, "Loading record %s", recordPath.toChar());
    cmRecord* record = cmLoadRecord(recordPath, getConfig().getCacheMBytes(), errorMsg);
    if(!record) {
        notifyErrorForDisplay(ERROR_LOAD, errorMsg);
        _underDisplayAppIdx = -1;
        _underDisplayRecIdx = -1;
        return false;
    }

    _underDisplayAppIdx = appIdx;
    _underDisplayRecIdx = recIdx;
    _record             = record;
    getConfig().notifyNewRecord(record);
    bsString nickname = _cmRecordInfos[appIdx].records[recIdx].nickname;
    if(!nickname.empty()) nickname = bsString(" - ")+nickname;
    osSetWindowTitle(bsString("Palanteer - ")+_record->appName + nickname + " - " + getNiceDate(_record->recordDate, osGetDate()));
    getConfig().setLastLoadedRecordPath(recordPath);

    // Apply the last workspace
    _screenLayoutToApply = getConfig().getCurrentLayout();

    // Take the opportunity to save the global settings
    getConfig().saveGlobal();

    dirty();
    return true;
}


void
vwMain::clearViews(void)
{
    plData("Subaction", plMakeString("Clear views"));
    plScope("clearViews");

    _hlThreadId = cmConst::MAX_THREAD_QTY;

#define CLEAR_ARRAY_VIEW(array) for(auto& a : (array)) releaseId(a.uniqueId); (array).clear();
    CLEAR_ARRAY_VIEW(_timelines);
    CLEAR_ARRAY_VIEW(_memTimelines);
    CLEAR_ARRAY_VIEW(_memDetails);
    CLEAR_ARRAY_VIEW(_profiles);
    CLEAR_ARRAY_VIEW(_texts);
    CLEAR_ARRAY_VIEW(_logViews);
    CLEAR_ARRAY_VIEW(_plots);
    CLEAR_ARRAY_VIEW(_histograms);
    _profiledCmDataIdx = -1;
    _plotMenuItems.clear();
    _search.reset();

    dirty();
}


void
vwMain::clearRecord(void)
{
    plData("Subaction", plMakeString("Clear record"));
    plScope("clearRecord");

    // Save the configuration
    if(_record) {
        getConfig().saveApplication(_record->appName);
        getConfig().saveGlobal();
    }

    // Reset views
    clearViews();

    // Delete record
    delete _record; _record = 0;
    _streamQty = 0;
    _underDisplayAppIdx = -1;
    _underDisplayRecIdx = -1;

    // Select catalog view, if visible
    if(getConfig().getWindowCatalogVisibility()) _catalogWindow.isWindowSelected = true;

    // Update the window
    osSetWindowTitle("Palanteer");
    dirty();
}


// =====================================
// Record file management
// =====================================

void
vwMain::removeSomeRecords(const bsVec<bsString>& recordsToDelete)
{
    if(recordsToDelete.empty()) return;

    // Case we delete the current folder
    bsString currentDisplayedPath;
    if(_underDisplayAppIdx>=0 && _underDisplayRecIdx>=0) {
        currentDisplayedPath = _cmRecordInfos[_underDisplayAppIdx].records[_underDisplayRecIdx].path;
    }

    // Loop on folders to delete
    for(const auto& path : recordsToDelete) {
        if(!currentDisplayedPath.empty() && currentDisplayedPath==path) _doClearRecord = true;
        // Remove the record file
        logToConsole(LOG_INFO, "Removing record %s", path.toChar());
        osRemoveFile(path);
        // Remove the nickname and external string file, if they exist
        bsString baseName = path.subString(0, path.size()-4);
        osRemoveFile(baseName+"_nickname");
        osRemoveFile(baseName+"_externalStrings");
    }

    // Update the record list
    updateRecordList();
    if(!currentDisplayedPath.empty()) {
        findRecord(currentDisplayedPath, _underDisplayAppIdx, _underDisplayRecIdx); // @#TBC See if we can simplify this "find record" and indexes stuff
    }
    dirty();
}


void
vwMain::updateRecordList(void)
{
    _cmRecordInfos.clear();

    // List the folder in the record folder
    bsVec<bsDirEntry> dirEntries;
    bsDirStatusCode status;
    if((status=osGetDirContent(_storagePath, dirEntries))!=bsDirStatusCode::OK) {
        logToConsole(LOG_ERROR, "Update record list: Unable to read the directory content of %s. Reason is '%s'",
                     _storagePath.toChar(), osGetDirStatusCodeStr(status));
        return;
    }

    for(auto& appEntry : dirEntries) {
        if(!appEntry.isDir) continue; // We are looking for folder only
        // Create the application structure
        AppRecordInfos appElem;
        appElem.path = _storagePath+appEntry.name;
        appElem.size = 0;
        appElem.name = appEntry.name;

        // Collect its list of records
        bsVec<bsDirEntry> appEntries;
        if((status=osGetDirContent(appElem.path, appEntries))!=bsDirStatusCode::OK) {
            logToConsole(LOG_ERROR, "Update record list: Unable to read the directory content of %s. Reason is '%s'",
                         appElem.path.toChar(), osGetDirStatusCodeStr(status));
            continue;
        }
        for(auto& recEntry : appEntries) {
            if(recEntry.isDir || !recEntry.name.endsWith(".plt")) continue; // We are looking for .plt files only
            RecordInfos recElem;
            recElem.idx  = (int)appElem.records.size();
            recElem.path = (appElem.path+PL_DIR_SEP)+recEntry.name;
            recElem.date = osGetCreationDate(recElem.path);
            if(recElem.date.isEmpty()) continue; // Not a real record
            recElem.size = osGetSize(recElem.path);
            appElem.size += recElem.size;
            recElem.nickname[0] = 0;
            bsVec<u8> bufferName;
            if(osLoadFileContent(recElem.path.subString(0, recElem.path.size()-4)+"_nickname", bufferName, sizeof(recElem.nickname))) {
                memcpy(&recElem.nickname[0], &bufferName[0], bufferName.size());
                recElem.nickname[sizeof(recElem.nickname)-1] = 0;
            }
            appElem.records.push_back(recElem);
        }
        if(appElem.records.empty()) continue;

        // Store in anti-chronological order (more recent first)
        std::sort(appElem.records.begin(), appElem.records.end(),
                  [](const RecordInfos& a, const RecordInfos& b)->bool { return b.date.isOlderThan(a.date); });
        int idx = 0;
        for(auto& recElem : appElem.records) recElem.idx = idx++;
        _cmRecordInfos.push_back(appElem);
    }

    // Most recently used app first
    std::sort(_cmRecordInfos.begin(), _cmRecordInfos.end(),
              [](const AppRecordInfos& a, const AppRecordInfos& b)->bool { return b.records[0].date.isOlderThan(a.records[0].date); });
    int idx = 0;
    for(auto& appElem : _cmRecordInfos) appElem.idx = idx++;
}


bool
vwMain::findRecord(const bsString& recordPath, int& foundAppIdx, int& foundRecIdx)
{
    foundAppIdx = -1;
    foundRecIdx = -1;
    for(int appIdx=0; appIdx<_cmRecordInfos.size(); ++appIdx) {
        const AppRecordInfos& appElem = _cmRecordInfos[appIdx];
        for(int recIdx=0; recIdx<appElem.records.size(); ++recIdx) {
            const RecordInfos& recElem = appElem.records[recIdx];
            if(recElem.path!=recordPath) continue;
            foundAppIdx = appIdx;  // Record was found
            foundRecIdx = recIdx;
            return true;
        }
    }
    return false;
}


// ====================
// UI Layout management
// ====================

void
vwMain::setFullScreenView(int uniqueId)
{
    _nextUniqueIdFullScreen = uniqueId;
}


void
vwMain::selectBestDockLocation(bool bigWidth, bool bigHeight)
{
    // Get root node and prepare recursive parsing
    ImGuiID mainDockspaceId = ImGui::GetID("MainDockSpace");
    ImGuiDockNode* root = ImGui::DockBuilderGetNode(mainDockspaceId);
    plAssert(root);
    struct ClassDockId { ImGuiID id=0; float criterion=0.; };
    ClassDockId cds[4]; // 0 = biggest area, 1=highest, 2=widest, 3=smallest area

    // Parse the layout tree
    bsVec<ImGuiDockNode*> stack; stack.reserve(32);
    stack.push_back(root);
    while(!stack.empty()) {
        ImGuiDockNode* node = stack.back(); stack.pop_back();
        if(node->IsLeafNode()) {
            // Categorize the node
            ImVec2& s = node->SizeRef;
            float criterion = 0.f;
            for(int classKind=0; classKind<4; ++classKind) {
                switch(classKind) {
                case 0: criterion = s[0]*s[1]/sqrt(bsMax(s[0]/s[1], s[1]/s[0])); break; // Max area and favoring square shape
                case 1: criterion = s[1]/sqrt(s[0]); break; // Max height and favoring thin shape
                case 2: criterion = s[0]/sqrt(s[1]); break; // Max width and favoring thick shape
                case 3: criterion = 1.f/(s[0]*s[1]);  break; // Smallest area
                }
                criterion *= 1.f- ((node->TabBar)? 0.001f*node->TabBar->Tabs.size() : 0.f); // Slightly favorize node with least views inside
                if(cds[classKind].id==0 || cds[classKind].criterion<criterion) cds[classKind] = { node->ID, criterion };
            }
        } else {
            // Propagate
            stack.push_back(node->ChildNodes[0]);
            stack.push_back(node->ChildNodes[1]);
        }
    }

    // Assign the next ImGui window to this dockinbg location
    int classIdx = (bigWidth? 0:1) + (bigHeight? 0:2);
    ImGui::SetNextWindowDockID(cds[classIdx].id);
}


void
vwMain::createLayoutViews(const vwConfig::ScreenLayout& layout)
{
    plAssert(_record);
#define READ_VIEW(keywordName, kwLength, readQty, ...)                  \
    isFound = false;                                                    \
    if(!strncmp(view.descr.toChar(), #keywordName, kwLength)) {         \
        if(sscanf(view.descr.toChar()+kwLength+1, __VA_ARGS__)!=readQty) { printf("Unable to find the view '" #keywordName "'\n"); } \
        else isFound = true;                                            \
    }
#define SET_VIEW_ATTRIBUTES(array)                          \
    plAssert(view.id<1000000);                              \
    while(idArray.size()<=view.id) idArray.push_back(0);    \
    idArray[view.id]                = 1; /* in use */       \
    (array).back().syncMode         = syncMode;             \
    (array).back().isNew            = false;                \
    (array).back().isWindowSelected = false;

    // Init (clear views)
    u64 hash=0, hash2=0;
    int syncMode=0, tmp2=0, tmp3=0, tmp4=0;
    bool isFound = false;
    bsVec<u8> idArray; idArray.reserve(128); // For ID pool initial state
    clearViews();
    plData("Subaction", plMakeString("Create layout views"));

    // Loop on view specifications
    for(const vwConfig::LayoutView& view : layout.views) {
        // Timeline
        READ_VIEW(timeline, 8, 1, "%d", &syncMode);
        if(isFound) {
            if(addTimeline(view.id)) {
                SET_VIEW_ATTRIBUTES(_timelines);
            }
            continue;
        }
        // Memory timeline
        READ_VIEW(memtimeline, 11, 1, "%d", &syncMode);
        if(isFound) {
            if(addMemoryTimeline(view.id)) {
                SET_VIEW_ATTRIBUTES(_memTimelines);
            }
            continue;
        }
        // Log
        READ_VIEW(log, 3, 1, "%d", &syncMode);
        if(isFound) {
            if(addLog(view.id, 0)) {
                SET_VIEW_ATTRIBUTES(_logViews);
            }
            continue;
        }
        // Text
        READ_VIEW(text, 4, 2, "%d %" PRIX64, &syncMode, &hash);
        if(isFound) {
            if(addText(view.id, -1, hash, 0, 0)) {
                SET_VIEW_ATTRIBUTES(_texts);
            }
            continue;
        }
        // Profile
        READ_VIEW(profile, 7, 5, "%d %d %d %d %" PRIX64, &syncMode, &tmp2, &tmp3, &tmp4, &hash);
        if(isFound) {
            if(addProfileRange(view.id, (ProfileKind)tmp2, -1, hash, 0L, _record->durationNs)) {
                _profiles.back().isFlameGraph         = tmp3;
                _profiles.back().isFlameGraphDownward = tmp4;
                SET_VIEW_ATTRIBUTES(_profiles);
            }
            continue;
        }
        // Histogram
        READ_VIEW(histogram, 9, 4, "%d %" PRIX64 " %" PRIX64 " %d", &syncMode, &hash, &hash2, &tmp2);
        if(isFound) {
            if(addHistogram(view.id, hash, hash2, -1, 0, _record->durationNs, tmp2)) {
                SET_VIEW_ATTRIBUTES(_histograms);
            }
            continue;
        }
        // Plot
        READ_VIEW(plot, 4, 1, "%d", &syncMode);
        if(isFound) {
            // Collect all the curve to add inside
            bsVec<u64> elemHashPaths;
            bsVec<int> logParamIndices;
            const char* s = view.descr.toChar()+5;
            while(*s==' ') ++s;  // Skip the syncMode and its space separation
            while(*s && *s!=' ') ++s;
            while(*s==' ') ++s;
            while(1) {
                if(sscanf(s, "%" PRIX64 " %" PRIX64 " %d", &hash, &hash2, &tmp2)!=3) break;
                for(int i=0; i<3; ++i) {   // Skip the 3 numbers and their space separation
                    while(*s && *s!=' ') ++s;
                    while(*s==' ') ++s;
                }
                elemHashPaths.push_back(hash);   // Thread Unique Hash
                elemHashPaths.push_back(hash2);  // Element hashPath
                logParamIndices.push_back(tmp2); // Log parameter
            }
            // Create the plot window
            if(!elemHashPaths.empty()) {
                _plots.push_back( { } );
                PlotWindow& pw = _plots.back();
                pw.uniqueId    = view.id;
                pw.startTimeNs = 0;
                pw.timeRangeNs = _record->durationNs;
                SET_VIEW_ATTRIBUTES(_plots);
                for(int i=0; i<logParamIndices.size(); ++i) {
                    pw.curves.push_back( { elemHashPaths[2*i+0], elemHashPaths[2*i+1], -1, false, false, logParamIndices[i] } );
                }
            }
            continue;
        }

    } // End of loop on views

    // Force the state of the ID pool
    _idMax = idArray.size();
    _idPool.clear();
    for(int i=0; i<idArray.size(); ++i) if(idArray[i]==0) _idPool.push_back(i);
}


void
vwMain::copyCurrentLayout(vwConfig::ScreenLayout& layout, const bsString& windowLayout)
{
    layout.windows = windowLayout;
    layout.views.clear(); layout.views.reserve(32);

#define SAVE_VIEWS(array) for(int i=0; i<(array).size(); ++i) layout.views.push_back({ (array)[i].uniqueId, (array)[i].getDescr() });
    SAVE_VIEWS(_timelines);
    SAVE_VIEWS(_memTimelines);
    SAVE_VIEWS(_logViews);
    SAVE_VIEWS(_texts);
    SAVE_VIEWS(_profiles);
    SAVE_VIEWS(_histograms);
    SAVE_VIEWS(_plots);
}
