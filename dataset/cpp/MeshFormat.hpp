#pragma once

#ifndef RAZ_MESHFORMAT_HPP
#define RAZ_MESHFORMAT_HPP

#include <utility>

namespace Raz {

class FilePath;
class Mesh;
class MeshRenderer;

namespace MeshFormat {

/// Loads a mesh from a file.
/// \param filePath File from which to load the mesh.
/// \return Pair containing respectively the mesh's data (vertices & indices) and rendering information (materials, textures, ...).
std::pair<Mesh, MeshRenderer> load(const FilePath& filePath);

/// Saves a mesh to a file.
/// \param filePath File to which to save the mesh.
/// \param mesh Mesh to export data from.
/// \param meshRenderer Optional mesh renderer to export materials & textures from.
void save(const FilePath& filePath, const Mesh& mesh, const MeshRenderer* meshRenderer = nullptr);

} // namespace MeshFormat

} // namespace Raz

#endif // RAZ_MESHFORMAT_HPP
