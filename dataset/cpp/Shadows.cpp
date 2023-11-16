////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Shadows.hpp"

#include "../cs-utils/FrameStats.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaFramebufferObj.h>
#include <VistaOGLExt/VistaTexture.h>
#include <array>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowCaster::setShadowMap(ShadowMap* pShadowMap) {
  mShadowMap = pShadowMap;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ShadowMap* ShadowCaster::getShadowMap() const {
  return mShadowMap;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ShadowMap::~ShadowMap() {
  cleanUp();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::registerCaster(ShadowCaster* caster) {
  caster->setShadowMap(this);
  mShadowCasters.insert(caster);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::deregisterCaster(ShadowCaster* caster) {
  caster->setShadowMap(nullptr);
  mShadowCasters.erase(caster);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setSunDirection(VistaVector3D const& direction) {
  if (mFreezeCascades) {
    return;
  }
  mSunDirection = direction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaVector3D const& ShadowMap::getSunDirection() const {
  return mSunDirection;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setResolution(unsigned resolution) {
  if (mResolution != resolution) {
    mFBODirty   = true;
    mResolution = resolution;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

unsigned ShadowMap::getResolution() const {
  return mResolution;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setCascadeSplits(std::vector<float> const& splits) {
  if (mFreezeCascades) {
    return;
  }

  if (mSplits.size() != splits.size()) {
    mFBODirty = true;
  }

  mSplits = splits;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setSunNearClipOffset(float offset) {
  if (mFreezeCascades) {
    return;
  }

  mSunNearClipOffset = offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ShadowMap::getSunNearClipOffset() const {
  return mSunNearClipOffset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setSunFarClipOffset(float offset) {
  if (mFreezeCascades) {
    return;
  }

  mSunFarClipOffset = offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ShadowMap::getSunFarClipOffset() const {
  return mSunFarClipOffset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setBias(float bias) {
  mBias = bias;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float ShadowMap::getBias() const {
  return mBias;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setFreezeCascades(bool freeze) {
  mFreezeCascades = freeze;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ShadowMap::getFreezeCascades() const {
  return mFreezeCascades;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::setEnabled(bool enable) {
  mEnabled = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ShadowMap::getEnabled() const {
  return mEnabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> const& ShadowMap::getCascadeSplits() const {
  return mSplits;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<VistaTexture*> const& ShadowMap::getMaps() const {
  return mShadowMaps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<VistaTransformMatrix> const& ShadowMap::getShadowMatrices() const {
  return mShadowMatrices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ShadowMap::Do() {
  if (!mEnabled) {
    return true;
  }

  utils::FrameStats::ScopedTimer timer("Update ShadowMap");

  if (mFBODirty) {
    cleanUp();

    if (mSplits.size() < 2) {
      logger().warn("Shadows will be ugly: No splits have been defined!");
      mSplits = {0.1F, 5.F, 20.F, 50.F, 100.F};
    }

    // create shadow maps for all cascades
    for (size_t i = 0; i < mSplits.size() - 1; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      mShadowMapFBOs.push_back(new VistaFramebufferObj());

      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      mShadowMaps.push_back(new VistaTexture(GL_TEXTURE_2D));
      mShadowMatrices.emplace_back(VistaTransformMatrix());

      mShadowMaps[i]->Bind();
      glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, mResolution, mResolution, 0,
          GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);

      mShadowMapFBOs[i]->Attach(mShadowMaps[i], GL_DEPTH_ATTACHMENT);
      mShadowMaps[i]->Unbind();
    }

    mFBODirty = false;
  }

  // save current viewport
  std::array<GLint, 4> iOrigViewport{};
  glGetIntegerv(GL_VIEWPORT, iOrigViewport.data());

  // get view matrix - as this shadowmap should be attached to
  // scenegraph root, we can just use the modelview matrix here
  std::array<GLfloat, 16> glViewMat{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glViewMat.data());
  VistaTransformMatrix currMatView = VistaTransformMatrix(glViewMat.data(), true);

  // update projection and view matrix only if update is not frozen
  if (!mFreezeCascades) {
    matView = currMatView;

    // get projection matrix
    std::array<GLfloat, 16> glProjectionMat{};
    glGetFloatv(GL_PROJECTION_MATRIX, glProjectionMat.data());
    matProjection = VistaTransformMatrix(glProjectionMat.data(), true);
  }

  // setup sun view matrix
  VistaVector3D viewDirection = matView.GetInverted().GetRotationAsQuaternion().GetViewDir();

  VistaVector3D lightBaseZ = mSunDirection;
  VistaVector3D lightBaseX = viewDirection.Cross(lightBaseZ);
  VistaVector3D lightBaseY = -lightBaseX.Cross(lightBaseZ);

  lightBaseX[3] = 0;
  lightBaseY[3] = 0;
  lightBaseZ[3] = 0;

  lightBaseX.Normalize();
  lightBaseY.Normalize();
  lightBaseZ.Normalize();

  VistaTransformMatrix lightMatrix(lightBaseX, lightBaseY, lightBaseZ);
  lightMatrix.Invert();

  // get user frustum corners and corners of frustum splits in light space
  std::vector<std::array<VistaVector3D, 4>> splitSlices;

  // clip space coordinates are transformed to world space with the inverted
  // projection-view matrix and then transformed into light space
  VistaTransformMatrix transform = (matProjection * matView).GetInverted();

  for (float split : mSplits) {
    float slicePosition = (matProjection * VistaVector3D(0, 0, -split)).GetHomogenized()[2];
    splitSlices.push_back(
        {lightMatrix * (transform * VistaVector3D(-1, -1, slicePosition, 1)).GetHomogenized(),
            lightMatrix * (transform * VistaVector3D(-1, 1, slicePosition, 1)).GetHomogenized(),
            lightMatrix * (transform * VistaVector3D(1, 1, slicePosition, 1)).GetHomogenized(),
            lightMatrix * (transform * VistaVector3D(1, -1, slicePosition, 1)).GetHomogenized()});
  }

  // ow we render all registered shadow casters into the shadow maps
  for (size_t i = 0; i < mSplits.size() - 1; ++i) {
    // bind the fbo
    mShadowMapFBOs[i]->Bind();

    // setup sun projection matrix
    float r = std::numeric_limits<float>::lowest();
    float t = std::numeric_limits<float>::lowest();
    float n = std::numeric_limits<float>::lowest();

    float l = std::numeric_limits<float>::max();
    float b = std::numeric_limits<float>::max();
    float f = std::numeric_limits<float>::max();

    // as slice corners are in light space alreay, we can just calculate the
    // bounding box of each frustum slice by min and max
    for (size_t s = i; s < i + 2; ++s) {
      for (auto const& p : splitSlices[s]) {
        r = std::max(r, p[0]);
        l = std::min(l, p[0]);
        t = std::max(t, p[1]);
        b = std::min(b, p[1]);
        f = std::min(f, p[2]);
        n = std::max(n, p[2]);
      }
    }

    // extend the sun frustum in sun direction
    f -= mSunFarClipOffset;
    n -= mSunNearClipOffset;

    // eliminate supixel movement
    float w = r - l;
    float x = (l + r) * 0.5F;
    x -= std::fmod(x, w / static_cast<float>(mResolution));
    l = x - w * 0.5F;
    r = x + w * 0.5F;

    float h = t - b;
    float y = (b + t) * 0.5F;
    y -= std::fmod(y, h / static_cast<float>(mResolution));
    b = y - h * 0.5F;
    t = y + h * 0.5F;

    // create the orthographic projection matrix
    VistaTransformMatrix projection(2.0F / (r - l), 0.0, 0.0, -(r + l) / (r - l), 0.0,
        2.0F / (t - b), 0.0, -(t + b) / (t - b), 0.0, 0.0, -2.0F / (n - f), -(n + f) / (n - f), 0.0,
        0.0, 0.0, 1.0);

    // these matrices are used by the shadow receivers to calculate the
    // lookup position in the shadow maps
    mShadowMatrices.at(i) = projection * lightMatrix;

    // save current projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

    // load sun projection
    glLoadMatrixf(projection.GetData());

    // save current modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    // setup viewport
    glViewport(0, 0, mResolution, mResolution);

    // clear fbo
    glClearDepth(1.0f);
    glClear(GL_DEPTH_BUFFER_BIT);

    // setup "normal" GL state (we are not using the reverse infinite projection for the shadows)
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);

    // draw all shadow casters
    for (auto* caster : mShadowCasters) {
      glLoadMatrixf(lightMatrix.GetData());
      caster->drawForShadowMap();
    }

    // restore previous projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    // restore previous modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // restore GL state for reverse infinite projection
    glDepthFunc(GL_GEQUAL);
    glClearDepth(0.0f);
    glEnable(GL_CULL_FACE);

    // unbind fbo again
    mShadowMapFBOs[i]->Release();
  }

  // restore previous viewport
  glViewport(iOrigViewport.at(0), iOrigViewport.at(1), iOrigViewport.at(2), iOrigViewport.at(3));

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ShadowMap::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float      min(std::numeric_limits<float>::min());
  float      max(std::numeric_limits<float>::max());
  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ShadowMap::cleanUp() {
  for (auto& map : mShadowMaps) {
    delete map; // NOLINT(cppcoreguidelines-owning-memory)
  }

  for (auto& fbo : mShadowMapFBOs) {
    delete fbo; // NOLINT(cppcoreguidelines-owning-memory)
  }

  mShadowMapFBOs.clear();
  mShadowMaps.clear();
  mShadowMatrices.clear();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
