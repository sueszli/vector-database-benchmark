// Copyright 2015-2023 the openage authors. See copying.md for legal info.

#include <array>

#include <QTransform>

#include "../../../texture.h"
#include "gui_make_standalone_subtexture.h"
#include "gui_texture.h"

namespace openage::gui {

GuiTexture::GuiTexture(const SizedTextureHandle &texture_handle) :
	QSGTexture{},
	texture_handle(texture_handle) {
}

GuiTexture::~GuiTexture() = default;

void GuiTexture::bind() {
	glBindTexture(GL_TEXTURE_2D, this->textureId());
}

qint64 GuiTexture::comparisonKey() const {
	// TODO: Qt5 What does this do??????
	return 0;
}

bool GuiTexture::hasAlphaChannel() const {
	// assume 32bit textures
	return true;
}

bool GuiTexture::hasMipmaps() const {
	return false;
}

bool GuiTexture::isAtlasTexture() const {
	return openage::gui::isAtlasTexture(this->texture_handle);
}

namespace {
GLuint create_compatible_texture(GLuint texture_id, GLsizei w, GLsizei h) {
	glBindTexture(GL_TEXTURE_2D, texture_id);

	GLint min_filter;
	GLint mag_filter;
	GLint iformat;

	glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, &min_filter);
	glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, &mag_filter);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &iformat);

	GLuint new_texture_id;
	glGenTextures(1, &new_texture_id);
	glBindTexture(GL_TEXTURE_2D, new_texture_id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);

	glTexImage2D(GL_TEXTURE_2D, 0, iformat, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	glBindTexture(GL_TEXTURE_2D, 0);

	return new_texture_id;
}
} // namespace

QSGTexture *GuiTexture::removedFromAtlas(QRhiResourceUpdateBatch *resourceUpdates /* = nullptr */) const {
	if (this->isAtlasTexture()) {
		if (!this->standalone) {
			auto tex = this->texture_handle.texture;
			auto sub = tex->get_subtexture(this->texture_handle.subid);

			GLuint sub_texture_id = create_compatible_texture(tex->get_texture_id(), sub->w, sub->h);

			std::array<GLuint, 2> fbo;
			glGenFramebuffers(fbo.size(), &fbo.front());

			glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]);
			glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->textureId(), 0);
			glReadBuffer(GL_COLOR_ATTACHMENT0);

			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1]);
			glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sub_texture_id, 0);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);

			glBlitFramebuffer(sub->x, sub->y, sub->x + sub->w, sub->y + sub->h, 0, 0, sub->w, sub->h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			glDeleteFramebuffers(fbo.size(), &fbo.front());

			this->standalone = make_standalone_subtexture(sub_texture_id, QSize(sub->w, sub->h));
		}

		return this->standalone.get();
	}

	return nullptr;
}

QRectF GuiTexture::normalizedTextureSubRect() const {
	if (this->isAtlasTexture()) {
		auto tex = this->texture_handle.texture;
		auto sub = tex->get_subtexture(this->texture_handle.subid);
		return QTransform::fromScale(tex->w, tex->h).inverted().mapRect(QRectF(sub->x, sub->y, sub->w, sub->h));
	}
	else {
		return QSGTexture::normalizedTextureSubRect();
	}
}

int GuiTexture::textureId() const {
	return this->texture_handle.texture->get_texture_id();
}

QSize GuiTexture::textureSize() const {
	return openage::gui::textureSize(this->texture_handle);
}

} // namespace openage::gui
