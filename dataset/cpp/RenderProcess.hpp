#pragma once

#ifndef RAZ_RENDERPROCESS_HPP
#define RAZ_RENDERPROCESS_HPP

#include <memory>

namespace Raz {

class RenderGraph;
class RenderPass;
class Texture2D;
using Texture2DPtr = std::shared_ptr<Texture2D>;

/// RenderProcess class, representing a set of render passes with fixed actions; can be derived to implement post effects.
class RenderProcess {
public:
  explicit RenderProcess(RenderGraph& renderGraph) : m_renderGraph{ renderGraph } {}

  virtual bool isEnabled() const noexcept = 0;

  virtual void setState(bool enabled) = 0;
  void enable() noexcept { setState(true); }
  void disable() noexcept { setState(false); }
  virtual void addParent(RenderPass& parentPass) = 0;
  virtual void addParent(RenderProcess& parentProcess) = 0;
  virtual void addChild(RenderPass& childPass) = 0;
  virtual void addChild(RenderProcess& childProcess) = 0;
  virtual void resizeBuffers([[maybe_unused]] unsigned int width, [[maybe_unused]] unsigned int height) {}
  /// Recovers the elapsed time (in milliseconds) of the process' execution.
  /// \note This action is not available with OpenGL ES and will always return 0.
  /// \return Time taken to execute the process.
  virtual float recoverElapsedTime() const { return 0.f; }

  virtual ~RenderProcess() = default;

protected:
  RenderGraph& m_renderGraph;
};

} // namespace Raz

#endif // RAZ_RENDERPROCESS_HPP
