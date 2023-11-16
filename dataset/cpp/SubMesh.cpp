// Copyright(c) 2019 - 2020, #Momo
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and /or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "SubMesh.h"
#include "Core/Runtime/Reflection.h"

namespace MxEngine
{
    const AABB& SubMesh::GetAABB() const
    {
        return this->Data.GetAABB();
    }

    const BoundingSphere& SubMesh::GetBoundingSphere() const
    {
        return this->Data.GetBoundingSphere();
    }

    SubMesh::SubMesh(size_t materialId, std::reference_wrapper<Transform> transform, MeshData data)
        : materialId(materialId), transform(transform), Data(std::move(data)) { }

    void SubMesh::SetTransform(const Transform& transform)
    {
        this->transform.get() = transform;
    }

    size_t SubMesh::GetMaterialId() const
    {
        return this->materialId;
    }

    const Transform& SubMesh::GetTransform() const
    {
        return this->transform;
    }

    Transform& SubMesh::GetTransformReference()
    {
        return this->transform;
    }

    MXENGINE_REFLECT_TYPE
    {
        rttr::registration::class_<SubMesh>("SubMesh")
            (
                rttr::metadata(MetaInfo::COPY_FUNCTION, Copy<SubMesh>)
            )
            .property_readonly("name", &SubMesh::Name)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property_readonly("material id", &SubMesh::GetMaterialId)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property("transform", &SubMesh::GetTransform, &SubMesh::SetTransform)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property("name", &SubMesh::Name)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property("data", &SubMesh::Data)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property_readonly("aabb", &SubMesh::GetAABB)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            )
            .property_readonly("bounding box", &SubMesh::GetAABB)
            (
                rttr::metadata(MetaInfo::FLAGS, MetaInfo::EDITABLE)
            );
    }
}