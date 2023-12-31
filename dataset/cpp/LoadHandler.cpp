////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LoadHandler.hpp"

#include <include/cef_app.h>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

void LoadHandler::WaitForFinishedLoading() const {
  while (!mInitialized) {
    CefDoMessageLoopWork();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void LoadHandler::OnLoadingStateChange(
    CefRefPtr<CefBrowser> /*browser*/, bool isLoading, bool /*canGoBack*/, bool /*canGoForward*/) {
  if (!isLoading) {
    mInitialized = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
