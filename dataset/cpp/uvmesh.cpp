/*
MIT License

Copyright (c) 2018-2020 Jonathan Young

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/*
example_uvmesh

This example uses the xatlas::AddUvMesh API to pack 10 copies of a model's existing texture coordinates into a single atlas.

Input: a .obj model file. It must have texture coordinates.

Output: the atlas texture coordinates rasterized to images, colored by chart (example_uvmesh_charts*.tga) and by triangle (example_uvmesh_tris*.tga).
*/
#include <mutex>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "split/3rd_stb_image_write.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rd_tinyobjloader.h"

#define XATLAS_IMPLEMENTATION
#include "3rd_xatlas.h"

#ifdef _MSC_VER
#define FOPEN(_file, _filename, _mode) { if (fopen_s(&_file, _filename, _mode) != 0) _file = NULL; }
#define STRICMP _stricmp
#else
#define FOPEN(_file, _filename, _mode) _file = fopen(_filename, _mode)
#include <strings.h>
#define STRICMP strcasecmp
#endif

static bool s_verbose = false;

class Stopwatch
{
public:
	Stopwatch() { reset(); }
	void reset() { m_start = clock(); }
	double elapsed() const { return (clock() - m_start) * 1000.0 / CLOCKS_PER_SEC; }
private:
	clock_t m_start;
};

static int Print(const char *format, ...)
{
	va_list arg;
	va_start(arg, format);
	printf("\r"); // Clear progress text.
	const int result = vprintf(format, arg);
	va_end(arg);
	return result;
}

// May be called from any thread.
static bool ProgressCallback(xatlas::ProgressCategory category, int progress, void *userData)
{
	// Don't interupt verbose printing.
	if (s_verbose)
		return true;
	Stopwatch *stopwatch = (Stopwatch *)userData;
	static std::mutex progressMutex;
	std::unique_lock<std::mutex> lock(progressMutex);
	if (progress == 0)
		stopwatch->reset();
	printf("\r   %s [", xatlas::StringForEnum(category));
	for (int i = 0; i < 10; i++)
		printf(progress / ((i + 1) * 10) ? "*" : " ");
	printf("] %d%%", progress);
	fflush(stdout);
	if (progress == 100)
		printf("\n      %.2f seconds (%g ms) elapsed\n", stopwatch->elapsed() / 1000.0, stopwatch->elapsed());
	return true;
}

static void RandomColor(uint8_t *color)
{
	for (int i = 0; i < 3; i++)
		color[i] = uint8_t((rand() % 255 + 192) * 0.5f);
}

static void SetPixel(uint8_t *dest, int destWidth, int x, int y, const uint8_t *color)
{
	uint8_t *pixel = &dest[x * 3 + y * (destWidth * 3)];
	pixel[0] = color[0];
	pixel[1] = color[1];
	pixel[2] = color[2];
}

// https://github.com/miloyip/line/blob/master/line_bresenham.c
// License: public domain.
static void RasterizeLine(uint8_t *dest, int destWidth, const int *p1, const int *p2, const uint8_t *color)
{
	const int dx = abs(p2[0] - p1[0]), sx = p1[0] < p2[0] ? 1 : -1;
	const int dy = abs(p2[1] - p1[1]), sy = p1[1] < p2[1] ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2;
	int current[2];
	current[0] = p1[0];
	current[1] = p1[1];
	while (SetPixel(dest, destWidth, current[0], current[1], color), current[0] != p2[0] || current[1] != p2[1])
	{
		const int e2 = err;
		if (e2 > -dx) { err -= dy; current[0] += sx; }
		if (e2 < dy) { err += dx; current[1] += sy; }
	}
}

/*
https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling
Copyright Dmitry V. Sokolov

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
static void RasterizeTriangle(uint8_t *dest, int destWidth, const int *t0, const int *t1, const int *t2, const uint8_t *color)
{
	if (t0[1] > t1[1]) std::swap(t0, t1);
	if (t0[1] > t2[1]) std::swap(t0, t2);
	if (t1[1] > t2[1]) std::swap(t1, t2);
	int total_height = t2[1] - t0[1];
	for (int i = 0; i < total_height; i++) {
		bool second_half = i > t1[1] - t0[1] || t1[1] == t0[1];
		int segment_height = second_half ? t2[1] - t1[1] : t1[1] - t0[1];
		float alpha = (float)i / total_height;
		float beta = (float)(i - (second_half ? t1[1] - t0[1] : 0)) / segment_height;
		int A[2], B[2];
		for (int j = 0; j < 2; j++) {
			A[j] = int(t0[j] + (t2[j] - t0[j]) * alpha);
			B[j] = int(second_half ? t1[j] + (t2[j] - t1[j]) * beta : t0[j] + (t1[j] - t0[j]) * beta);
		}
		if (A[0] > B[0]) std::swap(A, B);
		for (int j = A[0]; j <= B[0]; j++)
			SetPixel(dest, destWidth, j, t0[1] + i, color);
	}
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
	    printf("Usage: %s input_file.obj [options]\n", argv[0]);
		printf("  Options:\n");
		printf("    -verbose\n");  
	    return EXIT_FAILURE;
	}
	s_verbose = (argc >= 3 && STRICMP(argv[2], "-verbose") == 0);
	// Load object file.
	printf("Loading '%s'...\n", argv[1]);
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	if (!tinyobj::LoadObj(shapes, materials, err, argv[1], NULL, tinyobj::triangulation)) {
		printf("Error: %s\n", err.c_str());
		return EXIT_FAILURE;
	}
	if (shapes.size() == 0) {
		printf("Error: no shapes in obj file\n");
		return EXIT_FAILURE;
	}
	for (int i = 0; i < (int)shapes.size(); i++) {
		const tinyobj::mesh_t &objMesh = shapes[i].mesh;
		if (objMesh.texcoords.empty()) {
			printf("Error: obj file must have texture coordinates\n");
			return EXIT_FAILURE;
		}
	}
	printf("   %d shapes\n", (int)shapes.size());
	// Create atlas.
	xatlas::SetPrint(Print, s_verbose);
	xatlas::Atlas *atlas = xatlas::Create();
	// Set progress callback.
	Stopwatch globalStopwatch, stopwatch;
	xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);
	// Add meshes to atlas.
	// Add 10 copies of the same model.
	uint32_t totalVertices = 0, totalFaces = 0;
	const int n = 10;
	for (int i = 0; i < n; i++) {
		for (int s = 0; s < (int)shapes.size(); s++) {
			tinyobj::mesh_t &objMesh = shapes[s].mesh;
			xatlas::UvMeshDecl meshDecl;
			meshDecl.faceMaterialData = (const uint32_t *)objMesh.material_ids.data();
			meshDecl.vertexCount = (int)objMesh.texcoords.size() / 2;
			meshDecl.vertexUvData = objMesh.texcoords.data();
			meshDecl.vertexStride = sizeof(float) * 2;
			meshDecl.indexCount = (int)objMesh.indices.size();
			meshDecl.indexData = objMesh.indices.data();
			meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
			xatlas::AddMeshError error = xatlas::AddUvMesh(atlas, meshDecl);
			if (error != xatlas::AddMeshError::Success) {
				xatlas::Destroy(atlas);
				printf("\rError adding mesh %d '%s': %s\n", s, shapes[i].name.c_str(), xatlas::StringForEnum(error));
				return EXIT_FAILURE;
			}
			totalVertices += meshDecl.vertexCount;
			totalFaces += meshDecl.indexCount / 3;
		}
	}
	printf("   %u total vertices\n", totalVertices);
	printf("   %u total triangles\n", totalFaces);
	// Compute charts.
	printf("Computing charts\n");
	xatlas::ComputeCharts(atlas);
	// Pack charts.
	printf("Packing charts\n");
	xatlas::PackCharts(atlas);
	printf("   %d charts\n", atlas->chartCount);
	printf("   %d atlases\n", atlas->atlasCount);
	for (uint32_t i = 0; i < atlas->atlasCount; i++)
		printf("      %d: %0.2f%% utilization\n", i, atlas->utilization[i] * 100.0f);
	printf("   %ux%u resolution\n", atlas->width, atlas->height);
	printf("%.2f seconds (%g ms) elapsed total\n", globalStopwatch.elapsed() / 1000.0, globalStopwatch.elapsed());
	if (atlas->width > 0 && atlas->height > 0) {
		printf("Rasterizing result...\n");
		// Dump images.
		std::vector<uint8_t> outputTrisImage, outputChartsImage;
		const uint32_t imageDataSize = atlas->width * atlas->height * 3;
		outputTrisImage.resize(atlas->atlasCount * imageDataSize);
		outputChartsImage.resize(atlas->atlasCount * imageDataSize);
		for (uint32_t i = 0; i < atlas->meshCount; i++) {
			const xatlas::Mesh &mesh = atlas->meshes[i];
			// Rasterize mesh triangles.
			const uint8_t white[] = { 255, 255, 255 };
			for (uint32_t j = 0; j < mesh.indexCount; j += 3) {
				int32_t atlasIndex = -1;
				bool skip = false;
				int verts[3][2];
				for (int k = 0; k < 3; k++) {
					const xatlas::Vertex &v = mesh.vertexArray[mesh.indexArray[j + k]];
					if (v.atlasIndex == -1) {
						skip = true;
						break;
					}
					atlasIndex = v.atlasIndex;
					verts[k][0] = int(v.uv[0]);
					verts[k][1] = int(v.uv[1]);
				}
				if (skip)
					continue; // Skip triangles that weren't atlased.
				uint8_t color[3];
				RandomColor(color);
				uint8_t *imageData = &outputTrisImage[atlasIndex * imageDataSize];
				RasterizeTriangle(imageData, atlas->width, verts[0], verts[1], verts[2], color);
				RasterizeLine(imageData, atlas->width, verts[0], verts[1], white);
				RasterizeLine(imageData, atlas->width, verts[1], verts[2], white);
				RasterizeLine(imageData, atlas->width, verts[2], verts[0], white);
			}
			// Rasterize mesh charts.
			for (uint32_t j = 0; j < mesh.chartCount; j++) {
				const xatlas::Chart *chart = &mesh.chartArray[j];
				uint8_t color[3];
				RandomColor(color);
				for (uint32_t k = 0; k < chart->faceCount; k++) {
					int verts[3][2];
					for (int l = 0; l < 3; l++) {
						const xatlas::Vertex &v = mesh.vertexArray[mesh.indexArray[chart->faceArray[k] * 3 + l]];
						verts[l][0] = int(v.uv[0]);
						verts[l][1] = int(v.uv[1]);
					}
					uint8_t *imageData = &outputChartsImage[chart->atlasIndex * imageDataSize];
					RasterizeTriangle(imageData, atlas->width, verts[0], verts[1], verts[2], color);
					RasterizeLine(imageData, atlas->width, verts[0], verts[1], white);
					RasterizeLine(imageData, atlas->width, verts[1], verts[2], white);
					RasterizeLine(imageData, atlas->width, verts[2], verts[0], white);
				}
			}
		}
		for (uint32_t i = 0; i < atlas->atlasCount; i++) {
			char filename[256];
			snprintf(filename, sizeof(filename), "example_uvmesh_tris%02u.tga", i);
			printf("Writing '%s'...\n", filename);
			stbi_write_tga(filename, atlas->width, atlas->height, 3, &outputTrisImage[i * imageDataSize]);
			snprintf(filename, sizeof(filename), "example_uvmesh_charts%02u.tga", i);
			printf("Writing '%s'...\n", filename);
			stbi_write_tga(filename, atlas->width,atlas->height, 3, &outputChartsImage[i * imageDataSize]);
		}
	}
	// Cleanup.
	xatlas::Destroy(atlas);
	printf("Done\n");
	return EXIT_SUCCESS;
}
