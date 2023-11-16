// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_frame_tag_anidir.h"
#include "app/cmd/set_frame_tag_color.h"
#include "app/cmd/set_frame_tag_name.h"
#include "app/cmd/set_frame_tag_range.h"
#include "app/color.h"
#include "app/commands/command.h"
#include "app/commands/params.h"
#include "app/context_access.h"
#include "app/loop_tag.h"
#include "app/transaction.h"
#include "app/ui/frame_tag_window.h"
#include "base/convert_to.h"
#include "doc/anidir.h"
#include "doc/frame_tag.h"
#include "doc/sprite.h"

namespace app {

using namespace ui;

class FrameTagPropertiesCommand : public Command {
public:
  FrameTagPropertiesCommand();
  Command* clone() const override { return new FrameTagPropertiesCommand(*this); }

protected:
  void onLoadParams(const Params& params) override;
  bool onEnabled(Context* context) override;
  void onExecute(Context* context) override;

private:
  std::string m_tagName;
  ObjectId m_tagId;
};

FrameTagPropertiesCommand::FrameTagPropertiesCommand()
  : Command("FrameTagProperties",
            "Frame Tag Properties",
            CmdUIOnlyFlag)
  , m_tagId(NullId)
{
}

void FrameTagPropertiesCommand::onLoadParams(const Params& params)
{
  m_tagName = params.get("name");

  std::string id = params.get("id");
  if (!id.empty())
    m_tagId = ObjectId(base::convert_to<ObjectId>(id));
  else
    m_tagId = NullId;
}

bool FrameTagPropertiesCommand::onEnabled(Context* context)
{
  return context->checkFlags(ContextFlags::ActiveDocumentIsWritable);
}

void FrameTagPropertiesCommand::onExecute(Context* context)
{
  const ContextReader reader(context);
  const Sprite* sprite = reader.sprite();
  frame_t frame = reader.frame();
  const FrameTag* foundTag = nullptr;

  if (!m_tagName.empty())
    foundTag = sprite->frameTags().getByName(m_tagName);
  else if (m_tagId != NullId)
    foundTag = sprite->frameTags().getById(m_tagId);
  else
    foundTag = sprite->frameTags().innerTag(frame);

  if (!foundTag)
    return;

  FrameTagWindow window(sprite, foundTag);
  if (!window.show())
    return;

  ContextWriter writer(reader);
  Transaction transaction(writer.context(), "Change Frame Tag Properties");
  FrameTag* tag = const_cast<FrameTag*>(foundTag);

  std::string name = window.nameValue();
  if (tag->name() != name)
    transaction.execute(new cmd::SetFrameTagName(tag, name));

  doc::frame_t from, to;
  window.rangeValue(from, to);
  if (tag->fromFrame() != from ||
      tag->toFrame() != to) {
    transaction.execute(new cmd::SetFrameTagRange(tag, from, to));
  }

  doc::color_t docColor = window.colorValue();
  if (tag->color() != docColor)
    transaction.execute(new cmd::SetFrameTagColor(tag, docColor));

  doc::AniDir anidir = window.aniDirValue();
  if (tag->aniDir() != anidir)
    transaction.execute(new cmd::SetFrameTagAniDir(tag, anidir));

  transaction.commit();
}

Command* CommandFactory::createFrameTagPropertiesCommand()
{
  return new FrameTagPropertiesCommand;
}

} // namespace app
