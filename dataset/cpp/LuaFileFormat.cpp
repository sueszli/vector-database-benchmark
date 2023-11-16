#include "RaZ/Animation/Skeleton.hpp"
#include "RaZ/Audio/Sound.hpp"
#include "RaZ/Data/BvhFormat.hpp"
#if defined(RAZ_USE_FBX)
#include "RaZ/Data/FbxFormat.hpp"
#endif
#include "RaZ/Data/Image.hpp"
#include "RaZ/Data/ImageFormat.hpp"
#include "RaZ/Data/Mesh.hpp"
#include "RaZ/Data/MeshFormat.hpp"
#include "RaZ/Data/ObjFormat.hpp"
#include "RaZ/Data/OffFormat.hpp"
#include "RaZ/Data/PngFormat.hpp"
#include "RaZ/Data/TgaFormat.hpp"
#if defined(RAZ_USE_AUDIO)
#include "RaZ/Data/WavFormat.hpp"
#endif
#include "RaZ/Render/MeshRenderer.hpp"
#include "RaZ/Script/LuaWrapper.hpp"
#include "RaZ/Utils/FilePath.hpp"
#include "RaZ/Utils/TypeUtils.hpp"

#define SOL_ALL_SAFETIES_ON 1
#include "sol/sol.hpp"

namespace Raz {

using namespace TypeUtils;

void LuaWrapper::registerFileFormatTypes() {
  sol::state& state = getState();

  {
    sol::table bvhFormat = state["BvhFormat"].get_or_create<sol::table>();
    bvhFormat["load"]    = &BvhFormat::load;
  }

#if defined(RAZ_USE_FBX)
  {
    sol::table fbxFormat = state["FbxFormat"].get_or_create<sol::table>();
    fbxFormat["load"]    = &FbxFormat::load;
  }
#endif

  {
    sol::table imageFormat = state["ImageFormat"].get_or_create<sol::table>();
    imageFormat["load"]    = sol::overload([] (const FilePath& p) { return ImageFormat::load(p); },
                                           PickOverload<const FilePath&, bool>(&ImageFormat::load));
    imageFormat["save"]    = sol::overload([] (const FilePath& p, const Image& i) { ImageFormat::save(p, i); },
                                           PickOverload<const FilePath&, const Image&, bool>(&ImageFormat::save));
  }

  {
    sol::table meshFormat = state["MeshFormat"].get_or_create<sol::table>();
    meshFormat["load"]    = &MeshFormat::load;
    meshFormat["save"]    = sol::overload([] (const FilePath& p, const Mesh& m) { MeshFormat::save(p, m); },
                                          PickOverload<const FilePath&, const Mesh&, const MeshRenderer*>(&MeshFormat::save));
  }

  {
    sol::table objFormat = state["ObjFormat"].get_or_create<sol::table>();
    objFormat["load"]    = &ObjFormat::load;
    objFormat["save"]    = sol::overload([] (const FilePath& p, const Mesh& m) { ObjFormat::save(p, m); },
                                         PickOverload<const FilePath&, const Mesh&, const MeshRenderer*>(&ObjFormat::save));
  }

  {
    sol::table offFormat = state["OffFormat"].get_or_create<sol::table>();
    offFormat["load"]    = &OffFormat::load;
  }

  {
    sol::table pngFormat = state["PngFormat"].get_or_create<sol::table>();
    pngFormat["load"]    = sol::overload([] (const FilePath& p) { return PngFormat::load(p); },
                                         PickOverload<const FilePath&, bool>(&PngFormat::load));
    pngFormat["save"]    = sol::overload([] (const FilePath& p, const Image& i) { PngFormat::save(p, i); },
                                         PickOverload<const FilePath&, const Image&, bool>(&PngFormat::save));
  }

  {
    sol::table tgaFormat = state["TgaFormat"].get_or_create<sol::table>();
    tgaFormat["load"]    = sol::overload([] (const FilePath& p) { return TgaFormat::load(p); },
                                         PickOverload<const FilePath&, bool>(&TgaFormat::load));
  }

#if defined(RAZ_USE_AUDIO)
  {
    sol::table wavFormat = state["WavFormat"].get_or_create<sol::table>();
    wavFormat["load"]    = &WavFormat::load;
    wavFormat["save"]    = &WavFormat::save;
  }
#endif
}

} // namespace Raz
