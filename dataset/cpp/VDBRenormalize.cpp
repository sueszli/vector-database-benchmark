#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <zeno/StringObject.h>
#include <vector>
#include <zeno/VDBGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/ChangeBackground.h>

namespace zeno {
    struct  VDBRenormalizeSDF : zeno::INode {
  virtual void apply() override {

    auto inoutSDF = get_input("inoutSDF")->as<VDBFloatGrid>();
    int normIter = get_param<int>(("iterations"));
    int dilateIter = get_param<int>(("dilateIters"));
    auto lstracker = openvdb::tools::LevelSetTracker<openvdb::FloatGrid>(*(inoutSDF->m_grid));
    lstracker.setState({openvdb::math::FIRST_BIAS, openvdb::math::TVD_RK3, 1, 1});
    lstracker.setTrimming(openvdb::tools::lstrack::TrimMode::kNone);

    if (dilateIter > 0)
        lstracker.dilate(dilateIter);
    else if (dilateIter < 0)
        lstracker.erode(dilateIter);
    for(int i=0;i<normIter;i++)
        lstracker.normalize();
    //openvdb::tools::changeBackground(inoutSDF->m_grid->tree(), ((float)normIter)*(inoutSDF->m_grid->transformPtr()->voxelSize()[0]));
    //openvdb::tools::signedFloodFill(inoutSDF->m_grid->tree());

    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

static int defVDBRenormalizeSDF = zeno::defNodeClass<VDBRenormalizeSDF>("VDBRenormalizeSDF",
     { /* inputs: */ {
     "inoutSDF", 
     }, /* outputs: */ {
     "inoutSDF",
     }, /* params: */ {
         {"enum 1oUpwind", "method", "1oUpwind"},
         {"int", "iterations", "4"},
         {"int", "dilateIters", "0"},
     }, /* category: */ {
     "openvdb",
     }});

struct VDBSmooth : zeno::INode {
    virtual void apply() override {
        auto inoutVDBtype = get_input<VDBGrid>("inoutVDB")->getType();
        int width = get_input<NumericObject>("width")->get<int>();
        int iterations = get_input<NumericObject>("iterations")->get<int>();
        auto type = get_input<zeno::StringObject>("type")->value;

        openvdb::FloatGrid::Ptr mask = nullptr;
        if(has_input("MaskGrid")) {
            mask = get_input("MaskGrid")->as<VDBFloatGrid>()->m_grid;
        }

        if (inoutVDBtype == std::string("FloatGrid")) {
            auto inoutVDB = get_input("inoutVDB")->as<VDBFloatGrid>();
            auto lsf = openvdb::tools::Filter<openvdb::FloatGrid>(*(inoutVDB->m_grid));
            lsf.setGrainSize(1);
            if(type == "Gaussian")
              lsf.gaussian(width, iterations, mask.get());
            else if(type == "Mean")
              lsf.mean(width, iterations, mask.get());
            else if(type == "Median")
              lsf.median(width, iterations, mask.get());
            //openvdb::tools::ttls_internal::smoothLevelSet(*inoutSDF->m_grid, normIter, halfWidth);
            set_output("inoutVDB", get_input("inoutVDB"));
        }
        else if (inoutVDBtype == std::string("Vec3fGrid")) {
            auto inoutVDB = get_input("inoutVDB")->as<VDBFloat3Grid>();
            auto lsf = openvdb::tools::Filter<openvdb::Vec3fGrid>(*(inoutVDB->m_grid));
            lsf.setGrainSize(1);
            if(type == "Gaussian")
              lsf.gaussian(width, iterations, mask.get());
            else if(type == "Mean")
              lsf.mean(width, iterations, mask.get());
            else if(type == "Median")
              lsf.median(width, iterations, mask.get());
            set_output("inoutVDB", get_input("inoutVDB"));
        }
    }
};

ZENO_DEFNODE(VDBSmooth)(
    { /* inputs: */ {
    "inoutVDB",
    "MaskGrid",
    {"enum Mean Gaussian Median", "type", "Gaussian"},
    {"int", "width", "1"},
    {"int", "iterations", "1"},
    }, /* outputs: */ {
    "inoutVDB",
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
} });

struct  VDBSmoothSDF : zeno::INode { /* cihou old graph */
  virtual void apply() override {

    auto inoutSDF = get_input("inoutSDF")->as<VDBFloatGrid>();
    int width = get_param<int>(("width"));
    int iterations = get_param<int>(("iterations"));
    auto lsf = openvdb::tools::Filter<openvdb::FloatGrid>(*(inoutSDF->m_grid));
    lsf.setGrainSize(1);
    lsf.gaussian(width, iterations, nullptr);
    //openvdb::tools::ttls_internal::smoothLevelSet(*inoutSDF->m_grid, normIter, halfWidth);
    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

static int defVDBSmoothSDF = zeno::defNodeClass<VDBSmoothSDF>("VDBSmoothSDF",
     { /* inputs: */ {
     "inoutSDF", 
     }, /* outputs: */ {
     "inoutSDF",
     }, /* params: */ {
         {"int", "width", "1"},
         {"int", "iterations", "1"},
         {"string", "DEPRECATED", "Use VDBSmooth Instead"},
     }, /* category: */ {
     "deprecated",
     }});

struct  VDBDilateTopo : zeno::INode {
  virtual void apply() override {

    auto inoutSDF = get_input<zeno::VDBGrid>("inField");
    auto layers = get_input("layers")->as<zeno::NumericObject>()->get<int>();

    inoutSDF->dilateTopo(layers);
    set_output("oField", std::move(inoutSDF));
  }
};

static int defVDBDilateTopo = zeno::defNodeClass<VDBDilateTopo>("VDBDilateTopo",
     { /* inputs: */ {
     "inField", {"int", "layers",} 
     }, /* outputs: */ {
       "oField"
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

struct VDBErodeSDF : zeno::INode {
  virtual void apply() override {
    auto inoutSDF = get_input("inoutSDF")->as<VDBFloatGrid>();
    auto grid = inoutSDF->m_grid;
    auto depth = get_input("depth")->as<zeno::NumericObject>()->get<float>();
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
            iter.modifyValue([&](auto &v) {
                v += depth;
            });
        }
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);
    set_output("inoutSDF", get_input("inoutSDF"));
  }
};

static int defVDBErodeSDF = zeno::defNodeClass<VDBErodeSDF>("VDBErodeSDF",
     { /* inputs: */ {
     "inoutSDF", {"float", "depth"}, 
     }, /* outputs: */ {
       "inoutSDF",
     }, /* params: */ {
     }, /* category: */ {
     "openvdb",
     }});

}
