/******************************************************************************
    Copyright (C) 2016-2019 by Streamlabs (General Workings Inc)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#include "gs-vertexbuffer.h"
#include <stdexcept>
#include "util-memory.h"
extern "C" {
#pragma warning(push)
#pragma warning(disable : 4201)
#include <obs.h>
#pragma warning(pop)
}

GS::VertexBuffer::~VertexBuffer()
{
	m_positions = nullptr;
	m_normals = nullptr;
	m_tangents = nullptr;
	m_colors = nullptr;

	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		if (m_uvs[n]) {
			m_uvs[n] = nullptr;
		}
	}
	m_layerdata = nullptr;

	if (m_vertexbuffer) {
		obs_enter_graphics();
		gs_vertexbuffer_destroy(m_vertexbuffer);
		obs_leave_graphics();
		m_vertexbuffer = nullptr;
	}
}

GS::VertexBuffer::VertexBuffer(uint32_t maximumVertices)
{
	SetupVertexBuffer(maximumVertices);

	// In case of device being removed, try again to create VertexBuffer
	// after manually rebuilding GPU device
	if (!m_vertexbuffer) {
		blog(LOG_ERROR, "GS::VertexBuffer: fail to create buffer, trying to rebuild device");

		obs_enter_graphics();
		gs_rebuild_device();
		obs_leave_graphics();

		// in case the exception is thrown during m_vertexbuffer creation,
		// it would delete the m_vertexbufferdata as well,
		// thus, need to recreate everything from scratch.
		SetupVertexBuffer(maximumVertices);

		if (!m_vertexbuffer) {
			throw std::runtime_error("Failed to create vertex buffer.");
		}
	}
}

GS::VertexBuffer::VertexBuffer(gs_vertbuffer_t *vb)
{
	gs_vb_data *vbd = gs_vertexbuffer_get_data(vb);
	this->SetUVLayers((uint32_t)vbd->num_tex);

	if (vbd->points != nullptr)
		memcpy(m_positions, vbd->points, vbd->num * sizeof(vec3));
	if (vbd->normals != nullptr)
		memcpy(m_normals, vbd->normals, vbd->num * sizeof(vec3));
	if (vbd->tangents != nullptr)
		memcpy(m_tangents, vbd->tangents, vbd->num * sizeof(vec3));
	if (vbd->colors != nullptr)
		memcpy(m_colors, vbd->colors, vbd->num * sizeof(uint32_t));
	if (vbd->tvarray != nullptr) {
		for (size_t n = 0; n < vbd->num_tex; n++) {
			if (vbd->tvarray[n].array != nullptr && vbd->tvarray[n].width <= 4 && vbd->tvarray[n].width > 0) {
				if (vbd->tvarray[n].width == 4) {
					memcpy(m_uvs[n], vbd->tvarray[n].array, vbd->num * sizeof(vec4));
				} else {
					for (size_t idx = 0; idx < m_capacity; idx++) {
						float *mem = reinterpret_cast<float *>(vbd->tvarray[n].array) + (idx * vbd->tvarray[n].width);
						memset(&m_uvs[n][idx], 0, sizeof(vec4));
						memcpy(&m_uvs[n][idx], mem, vbd->tvarray[n].width);
					}
				}
			}
		}
	}
}

GS::VertexBuffer::VertexBuffer(VertexBuffer const &other) : VertexBuffer(other.m_capacity)
{
	// Copy Constructor
	memcpy(m_positions, other.m_positions, m_capacity * sizeof(vec3));
	memcpy(m_normals, other.m_normals, m_capacity * sizeof(vec3));
	memcpy(m_tangents, other.m_tangents, m_capacity * sizeof(vec3));
	memcpy(m_colors, other.m_colors, m_capacity * sizeof(vec3));
	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		memcpy(m_uvs[n], other.m_uvs[n], m_capacity * sizeof(vec3));
	}
}

GS::VertexBuffer::VertexBuffer(VertexBuffer const &&other)
{
	// Move Constructor
	m_capacity = other.m_capacity;
	m_size = other.m_size;
	m_layers = other.m_layers;
	m_positions = other.m_positions;
	m_normals = other.m_normals;
	m_tangents = other.m_tangents;
	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		m_uvs[n] = other.m_uvs[n];
	}
	m_vertexbufferdata = other.m_vertexbufferdata;
	m_vertexbuffer = other.m_vertexbuffer;
	m_layerdata = other.m_layerdata;
	m_colors = other.m_colors;
}

void GS::VertexBuffer::operator=(VertexBuffer const &&other)
{
	// Move Assignment
	/// First self-destruct (semi-destruct itself).
	if (m_vertexbuffer) {
		obs_enter_graphics();
		gs_vertexbuffer_destroy(m_vertexbuffer);
		obs_leave_graphics();
		m_vertexbuffer = nullptr;
	}

	/// Then assign new values.
	m_capacity = other.m_capacity;
	m_size = other.m_size;
	m_layers = other.m_layers;
	m_positions = other.m_positions;
	m_normals = other.m_normals;
	m_tangents = other.m_tangents;
	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		m_uvs[n] = other.m_uvs[n];
	}
	m_vertexbufferdata = other.m_vertexbufferdata;
	m_vertexbuffer = other.m_vertexbuffer;
	m_layerdata = other.m_layerdata;
	m_colors = other.m_colors;
}

void GS::VertexBuffer::Resize(uint32_t new_size)
{
	if (new_size > m_capacity) {
		throw std::out_of_range("new_size out of range");
	}
	m_size = new_size;
}

uint32_t GS::VertexBuffer::Size()
{
	return m_size;
}

bool GS::VertexBuffer::Empty()
{
	return m_size == 0;
}

const GS::Vertex GS::VertexBuffer::At(uint32_t idx)
{
	if ((idx < 0) || (idx >= m_size)) {
		throw std::out_of_range("idx out of range");
	}

	GS::Vertex vtx(&m_positions[idx], &m_normals[idx], &m_tangents[idx], &m_colors[idx], nullptr);
	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		vtx.uv[n] = &m_uvs[n][idx];
	}
	return vtx;
}

const GS::Vertex GS::VertexBuffer::operator[](uint32_t const pos)
{
	return At(pos);
}

void GS::VertexBuffer::SetUVLayers(uint32_t layers)
{
	m_layers = layers;
}

uint32_t GS::VertexBuffer::GetUVLayers()
{
	return m_layers;
}

vec3 *GS::VertexBuffer::GetPositions()
{
	return m_positions;
}

vec3 *GS::VertexBuffer::GetNormals()
{
	return m_normals;
}

vec3 *GS::VertexBuffer::GetTangents()
{
	return m_tangents;
}

uint32_t *GS::VertexBuffer::GetColors()
{
	return m_colors;
}

vec4 *GS::VertexBuffer::GetUVLayer(size_t idx)
{
	if ((idx < 0) || (idx >= m_layers)) {
		throw std::out_of_range("idx out of range");
	}
	return m_uvs[idx];
}

gs_vertbuffer_t *GS::VertexBuffer::Update(bool refreshGPU)
{
	if (!refreshGPU)
		return m_vertexbuffer;

	if (m_size > m_capacity)
		throw std::out_of_range("size is larger than capacity");

	// Update VertexBuffer data.
	m_vertexbufferdata = gs_vertexbuffer_get_data(m_vertexbuffer);
	memset(m_vertexbufferdata, 0, sizeof(gs_vb_data));
	m_vertexbufferdata->num = m_capacity;
	m_vertexbufferdata->points = m_positions;
	m_vertexbufferdata->normals = m_normals;
	m_vertexbufferdata->tangents = m_tangents;
	m_vertexbufferdata->colors = m_colors;
	m_vertexbufferdata->num_tex = m_layers;
	m_vertexbufferdata->tvarray = m_layerdata;
	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		m_layerdata[n].array = m_uvs[n];
		m_layerdata[n].width = 4;
	}

	// Update GPU
	obs_enter_graphics();
	gs_vertexbuffer_flush(m_vertexbuffer);
	obs_leave_graphics();

	// WORKAROUND: OBS Studio 20.x and below incorrectly deletes data that it doesn't own.
	m_vertexbufferdata->num = m_capacity;
	m_vertexbufferdata->num_tex = m_layers;
	for (uint32_t n = 0; n < m_layers; n++) {
		m_layerdata[n].width = 4;
	}

	return m_vertexbuffer;
}

gs_vertbuffer_t *GS::VertexBuffer::Update()
{
	return Update(true);
}

void GS::VertexBuffer::SetupVertexBuffer(uint32_t maximumVertices)
{
	if (maximumVertices > MAXIMUM_VERTICES) {
		throw std::out_of_range("maximumVertices out of range");
	}

	m_size = 0;
	// Assign limits.
	m_capacity = maximumVertices;
	m_layers = MAXIMUM_UVW_LAYERS;

	// Allocate memory for data.
	m_vertexbufferdata = gs_vbdata_create();
	m_vertexbufferdata->num = m_capacity;
	m_vertexbufferdata->points = m_positions = (vec3 *)bmalloc(sizeof(vec3) * m_capacity);
	memset(m_positions, 0, sizeof(vec3) * m_capacity);
	m_vertexbufferdata->normals = m_normals = (vec3 *)bmalloc(sizeof(vec3) * m_capacity);
	memset(m_normals, 0, sizeof(vec3) * m_capacity);
	m_vertexbufferdata->tangents = m_tangents = (vec3 *)bmalloc(sizeof(vec3) * m_capacity);
	memset(m_tangents, 0, sizeof(vec3) * m_capacity);
	m_vertexbufferdata->colors = m_colors = (uint32_t *)bmalloc(sizeof(uint32_t) * m_capacity);
	memset(m_colors, 0, sizeof(uint32_t) * m_capacity);
	m_vertexbufferdata->num_tex = m_layers;
	m_vertexbufferdata->tvarray = m_layerdata = (gs_tvertarray *)bmalloc(sizeof(gs_tvertarray) * m_layers);
	for (size_t n = 0; n < MAXIMUM_UVW_LAYERS; n++) {
		m_layerdata[n].array = m_uvs[n] = (vec4 *)bmalloc(sizeof(vec4) * m_capacity);
		m_layerdata[n].width = 4;
		memset(m_uvs[n], 0, sizeof(vec4) * m_capacity);
	}

	// Allocate GPU
	obs_enter_graphics();
	m_vertexbuffer = gs_vertexbuffer_create(m_vertexbufferdata, GS_DYNAMIC);
	obs_leave_graphics();
}
