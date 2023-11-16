////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SunFlare.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::trajectories {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* SunFlare::QUAD_VERT = R"(
#version 330

out vec2 vTexCoords;

uniform float uAspect;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec4 posVS  = uMatModelView * vec4(0, 0, 0, 1);
    vec4 posP   = uMatProjection * posVS;
    float scale = length(uMatModelView[0]) / length(posVS.xyz);

    if (posP.w < 0) {
        gl_Position = vec4(0);
        return;
    }

    posP /= posP.w;

    float h = scale * 10e10;
    float w = h / uAspect;

    posP.z = -0.9999;

    switch (gl_VertexID) {
        case 0:
            posP.xy += vec2(-w,  h);
            vTexCoords = vec2(-1, 1);
            break;
        case 1:
            posP.xy += vec2( w,  h);
            vTexCoords = vec2(1, 1);
            break;
        case 2:
            posP.xy += vec2(-w, -h);
            vTexCoords = vec2(-1, -1);
            break;
        default:
            posP.xy += vec2( w, -h);
            vTexCoords = vec2(1, -1);
            break;
    }

    gl_Position = posP;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* SunFlare::QUAD_FRAG = R"(
#version 330

uniform vec3 uColor;

in vec2 vTexCoords;

layout(location = 0) out vec3 oColor;

void main()
{
    // sun disc
    float dist = length(vTexCoords) * 100;
    float glow = exp(1.0 - dist);
    oColor = uColor * glow;
    
    // sun glow
    dist = min(1.0, length(vTexCoords));
    glow = 1.0 - pow(dist, 0.05);
    oColor += uColor * glow * 2;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

SunFlare::SunFlare(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<Plugin::Settings>                  pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>             solarSystem)
    : mSettings(std::move(settings))
    , mPluginSettings(std::move(pluginSettings))
    , mSolarSystem(std::move(solarSystem)) {

  mShader.InitVertexShaderFromString(QUAD_VERT);
  mShader.InitFragmentShaderFromString(QUAD_FRAG);
  mShader.Link();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.color            = mShader.GetUniformLocation("uColor");
  mUniforms.aspect           = mShader.GetUniformLocation("uAspect");

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eStars) + 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SunFlare::~SunFlare() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SunFlare::setObjectName(std::string objectName) {
  mObjectName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& SunFlare::getObjectName() const {
  return mObjectName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SunFlare::Do() {
  if (!mPluginSettings->mEnableSunFlares.get() || mSettings->mGraphics.pEnableHDR.get()) {
    return true;
  }

  auto object = mSolarSystem->getObject(mObjectName);
  if (!object || !object->getIsInExistence()) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer timer("SunFlare");
  // get viewport to draw dot with correct aspect ration
  std::array<GLint, 4> viewport{};
  glGetIntegerv(GL_VIEWPORT, viewport.data());
  float fAspect = 1.F * viewport.at(2) / viewport.at(3);

  // get modelview and projection matrices
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(object->getObserverRelativeTransform());

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE);
  glDepthMask(GL_FALSE);

  // draw simple dot
  mShader.Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.color, pColor.get()[0], pColor.get()[1], pColor.get()[2]);
  mShader.SetUniform(mUniforms.aspect, fAspect);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  mShader.Release();

  glDisable(GL_BLEND);
  glDepthMask(GL_TRUE);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SunFlare::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
