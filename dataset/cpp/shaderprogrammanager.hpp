#ifndef EE_GRAPHICSSHADERPROGRAMANAGER_HPP
#define EE_GRAPHICSSHADERPROGRAMANAGER_HPP

#include <eepp/graphics/base.hpp>
#include <eepp/graphics/shaderprogram.hpp>

#include <eepp/system/resourcemanager.hpp>
#include <eepp/system/singleton.hpp>
using namespace EE::System;

namespace EE { namespace Graphics {

/** @brief The Shader Program Manager is a singleton class that manages all the instances of Shader
   Programs instanciated. Releases the Shader Program instances automatically. So the user doesn't
   need to release any Shader Program instance. */
class EE_API ShaderProgramManager : public ResourceManager<ShaderProgram> {
	SINGLETON_DECLARE_HEADERS( ShaderProgramManager )

  public:
	virtual ~ShaderProgramManager();

	void reload();

  protected:
	ShaderProgramManager();
};

}} // namespace EE::Graphics

#endif
