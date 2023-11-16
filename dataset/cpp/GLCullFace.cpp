#include "GLCullFace.h"

namespace nCine
{
	GLCullFace::State GLCullFace::state_;

	void GLCullFace::enable()
	{
		if (state_.enabled == false) {
			glEnable(GL_CULL_FACE);
			state_.enabled = true;
		}
	}

	void GLCullFace::disable()
	{
		if (state_.enabled == true) {
			glDisable(GL_CULL_FACE);
			state_.enabled = false;
		}
	}

	void GLCullFace::setMode(GLenum mode)
	{
		if (mode != state_.mode) {
			glCullFace(mode);
			state_.mode = mode;
		}
	}

	void GLCullFace::setState(State newState)
	{
		if (newState.enabled) {
			enable();
		} else {
			disable();
		}
		setMode(newState.mode);

		state_ = newState;
	}
}
