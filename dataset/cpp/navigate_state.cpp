// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/ui/editor/navigate_state.h"

#include "app/ui/editor/editor.h"
#include "app/ui/editor/scrolling_state.h"
#include "ui/message.h"

namespace app {

using namespace ui;

NavigateState::NavigateState()
{
}

bool NavigateState::onMouseDown(Editor* editor, MouseMessage* msg)
{
  if (editor->hasCapture())
    return true;

  // UIContext* context = UIContext::instance();
  // // When an editor is clicked the current view is changed.
  // context->setActiveView(editor->getDocumentView());

  // Start scroll loop
  EditorStatePtr newState(new ScrollingState());
  editor->setState(newState);
  newState->onMouseDown(editor, msg);
  return true;
}

bool NavigateState::onMouseUp(Editor* editor, MouseMessage* msg)
{
  editor->releaseMouse();
  return true;
}

bool NavigateState::onMouseMove(Editor* editor, MouseMessage* msg)
{
  editor->showBrushPreview(msg->position());
  editor->updateStatusBar();
  return true;
}

bool NavigateState::onKeyDown(Editor* editor, KeyMessage* msg)
{
  return false;
}

bool NavigateState::onKeyUp(Editor* editor, KeyMessage* msg)
{
  return false;
}

} // namespace app
