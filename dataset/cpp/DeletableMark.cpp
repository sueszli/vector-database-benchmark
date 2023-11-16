////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "DeletableMark.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-gui/GuiItem.hpp"
#include "../../../src/cs-gui/WorldSpaceGuiArea.hpp"
#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csl::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::DeletableMark(std::shared_ptr<cs::core::InputManager> pInputManager,
    std::shared_ptr<cs::core::SolarSystem>                           pSolarSystem,
    std::shared_ptr<cs::core::Settings> settings, std::string objectName)
    : Mark(std::move(pInputManager), std::move(pSolarSystem), std::move(settings),
          std::move(objectName))
    , mGuiArea(new cs::gui::WorldSpaceGuiArea(65, 75))
    , mGuiItem(new cs::gui::GuiItem("file://../share/resources/gui/deletable_mark.html")) {

  initData();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DeletableMark::~DeletableMark() {
  if (mGuiNode) {
    mGuiItem->unregisterCallback("deleteMe");
    mInputManager->unregisterSelectable(mGuiNode);
    mGuiArea->removeItem(mGuiItem.get());

    delete mGuiNode;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DeletableMark::initData() {
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  auto* pGuiTransform = pSG->NewTransformNode(mTransform.get());

  pGuiTransform->Translate(0.F, 0.75F, 0.F);

  float const scale = 0.0005F;
  pGuiTransform->Scale(scale * static_cast<float>(mGuiArea->getWidth()),
      scale * static_cast<float>(mGuiArea->getHeight()), 1.F);

  pGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));
  mGuiArea->addItem(mGuiItem.get());

  mGuiItem->setCursorChangeCallback([](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });
  mGuiItem->setCanScroll(false);

  mGuiNode = pSG->NewOpenGLNode(pGuiTransform, mGuiArea.get());
  mInputManager->registerSelectable(mGuiNode);

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      pGuiTransform, static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  mGuiItem->registerCallback("deleteMe", "Call this to remove the tool.",
      std::function([this]() { pShouldDelete = true; }));

  mSelfSelectedConnection =
      pSelected.connect([this](bool val) { mGuiItem->callJavascript("setMinimized", !val); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::tools
