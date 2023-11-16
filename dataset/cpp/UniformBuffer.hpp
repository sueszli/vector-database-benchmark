#pragma once

#ifndef RAZ_UNIFORMBUFFER_HPP
#define RAZ_UNIFORMBUFFER_HPP

#include "RaZ/Data/OwnerValue.hpp"

namespace Raz {

class ShaderProgram;

template <typename T, std::size_t W, std::size_t H>
class Matrix;

template <typename T, std::size_t Size>
class Vector;

enum class UniformBufferUsage {
  STATIC,  ///< Data is assumed to never change.
  DYNAMIC, ///< Data is assumed to be changed often.
  STREAM   ///< Data is assumed to be given each frame.
};

class UniformBuffer {
public:
  explicit UniformBuffer(unsigned int size, UniformBufferUsage usage = UniformBufferUsage::DYNAMIC);
  UniformBuffer(const UniformBuffer&) = delete;
  UniformBuffer(UniformBuffer&&) noexcept = default;

  unsigned int getIndex() const { return m_index; }

  void bindUniformBlock(const ShaderProgram& program, unsigned int uboIndex, unsigned int shaderBindingIndex) const;
  void bindUniformBlock(const ShaderProgram& program, const std::string& uboName, unsigned int shaderBindingIndex) const;
  void bindBase(unsigned int bufferBindingIndex) const;
  void bindRange(unsigned int bufferBindingIndex, std::ptrdiff_t offset, std::ptrdiff_t size) const;
  void bind() const;
  void unbind() const;
  template <typename T>
  void sendData(T data, unsigned int offset) const noexcept { sendData(&data, sizeof(T), offset); }
  template <typename T, std::size_t Size>
  void sendData(const Vector<T, Size>& vec, unsigned int offset) const noexcept { sendData(vec.getDataPtr(), sizeof(vec), offset); }
  template <typename T, std::size_t W, std::size_t H>
  void sendData(const Matrix<T, W, H>& mat, unsigned int offset) const noexcept { sendData(mat.getDataPtr(), sizeof(mat), offset); }

  UniformBuffer& operator=(const UniformBuffer&) = delete;
  UniformBuffer& operator=(UniformBuffer&&) noexcept = default;

  ~UniformBuffer();

private:
  UniformBuffer();

  void sendData(const void* data, std::ptrdiff_t size, unsigned int offset) const noexcept;

  OwnerValue<unsigned int> m_index {};
};

} // namespace Raz

#endif // RAZ_UNIFORMBUFFER_HPP
