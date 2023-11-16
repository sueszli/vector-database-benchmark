#include <cstddef>
#include <zeno/zeno.h>
#include <zeno/MeshObject.h>
#include <zeno/PrimitiveObject.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <zeno/VDBGrid.h>
#include <omp.h>
#include <zeno/ZenoInc.h>
#include <openvdb/tools/LevelSetUtil.h> 
//#include <tl/function_ref.hpp>
//openvdb::FloatGrid::Ptr grid = 
//openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>
//(*openvdb::math::Transform::createLinearTransform(h), 
//points, triangles, quads, 4, 4);

namespace zeno {

struct MeshToSDF : zeno::INode{
    virtual void apply() override {
    auto h = get_param<float>(("voxel_size"));
    if(has_input("Dx"))
    {
      h = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto mesh = get_input("mesh")->as<MeshObject>();
    auto result = zeno::IObject::make<VDBFloatGrid>();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(mesh->vertices.size());
    triangles.resize(mesh->vertices.size()/3);
    quads.resize(0);
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size();i++)
    {
        points[i] = openvdb::Vec3s(mesh->vertices[i].x, mesh->vertices[i].y, mesh->vertices[i].z);
    }
#pragma omp parallel for
    for(int i=0;i<mesh->vertices.size()/3;i++)
    {
        triangles[i] = openvdb::Vec3I(i*3, i*3+1, i*3+2);
    }
    auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
    if(get_param<std::string>(("type"))==std::string("vertex"))
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", result);
  }
};

static int defMeshToSDF = zeno::defNodeClass<MeshToSDF>("MeshToSDF",
    { /* inputs: */ {
        "mesh",{"float","Dx"},
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
    {"float", "voxel_size", "0.08 0"},
    {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "deprecated",
    }});



struct PrimitiveToSDF : zeno::INode{
    virtual void apply() override {
    //auto h = get_param<float>(("voxel_size"));
    //if(has_input("Dx"))
    //{
      //h = get_input<NumericObject>("Dx")->get<float>();
    //}
    auto h = get_input2<float>("Dx");
    //auto h = get_input("Dx")->as<NumericObject>()->get<float>();
    if (auto p = dynamic_cast<VDBFloatGrid *>(get_input("PrimitiveMesh").get())) {
        set_output("sdf", get_input("PrimitiveMesh"));
        return;
    }
    auto mesh = get_input("PrimitiveMesh")->as<PrimitiveObject>();
    auto result = zeno::IObject::make<VDBFloatGrid>();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    points.resize(mesh->verts.size());
    triangles.resize(mesh->tris.size());
    quads.resize(mesh->quads.size());
#pragma omp parallel for
    for(int i=0;i<points.size();i++)
    {
        points[i] = openvdb::Vec3s(mesh->verts[i][0], mesh->verts[i][1], mesh->verts[i][2]);
    }
#pragma omp parallel for
    for(int i=0;i<triangles.size();i++)
    {
        triangles[i] = openvdb::Vec3I(mesh->tris[i][0], mesh->tris[i][1], mesh->tris[i][2]);
    }
#pragma omp parallel for
    for(int i=0;i<quads.size();i++)
    {
        quads[i] = openvdb::Vec4I(mesh->quads[i][0], mesh->quads[i][1], mesh->quads[i][2], mesh->quads[i][3]);
    }
    auto vdbtransform = openvdb::math::Transform::createLinearTransform(h);
    if(get_param<std::string>(("type"))==std::string("vertex"))
    {
        vdbtransform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(h));
    }
    result->m_grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(*vdbtransform,points, triangles, quads, 4, 4);
    openvdb::tools::signedFloodFill(result->m_grid->tree());
    set_output("sdf", result);
  }
};

static int defPrimitiveToSDF = zeno::defNodeClass<PrimitiveToSDF>("PrimitiveToSDF",
    { /* inputs: */ {
        "PrimitiveMesh", {"float","Dx","0.08"},
    }, /* outputs: */ {
        "sdf",
    }, /* params: */ {
        //{"float", "voxel_size", "0.08 0"},
        {"enum vertex cell", "type", "vertex"},
    }, /* category: */ {
    "openvdb",
    }});

struct SDFToFog : INode 
{
    virtual void apply() override {
        auto sdf = get_input<VDBFloatGrid>("SDF");
        if (!has_input("inplace") || !get_input2<bool>("inplace")) {
            sdf = std::make_shared<VDBFloatGrid>(sdf->m_grid->deepCopy());
        }
        //auto dx = sdf->m_grid->voxelSize()[0];
        openvdb::tools::sdfToFogVolume(*(sdf->m_grid));
        set_output("oSDF", std::move(sdf));
    }
};
static int defSDFToFog = zeno::defNodeClass<SDFToFog>("SDFToFog",
    { /* inputs: */ {
        "SDF",
        {"bool", "inplace", "0"},
    }, /* outputs: */ {
        "oSDF",
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});
}
