#pragma once

#ifndef RAZ_BOXBLURRENDERPROCESS_HPP
#define RAZ_BOXBLURRENDERPROCESS_HPP

#include "RaZ/Render/MonoPassRenderProcess.hpp"

namespace Raz {

class BoxBlurRenderProcess final : public MonoPassRenderProcess {
public:
  explicit BoxBlurRenderProcess(RenderGraph& renderGraph);

  void resizeBuffers(unsigned int width, unsigned int height) override;
  void setInputBuffer(Texture2DPtr colorBuffer);
  void setOutputBuffer(Texture2DPtr colorBuffer);
  void setStrength(unsigned int strength) const;
};

} // namespace Raz

#endif // RAZ_BOXBLURRENDERPROCESS_HPP
