#include <zeno/zeno.h>
#include <zeno/VDBGrid.h>
#include <zeno/StringObject.h>
#include <zeno/ZenoInc.h>

//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zeno {

struct WriteVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_param<std::string>("path");
    auto data = get_input<VDBGrid>("data");
    data->output(path);
  }
};

static int defWriteVDBGrid = zeno::defNodeClass<WriteVDBGrid>("WriteVDBGrid",
    { /* inputs: */ {
    "data",
    }, /* outputs: */ {
    }, /* params: */ {
    {"writepath", "path", ""},
    }, /* category: */ {
    "deprecated",
    }});


struct ExportVDBGrid : zeno::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<zeno::StringObject>()->get();
    auto data = get_input("data")->as<VDBGrid>();
    data->output(path);
  }
};

static int defExportVDBGrid = zeno::defNodeClass<ExportVDBGrid>("ExportVDBGrid",
    { /* inputs: */ {
    "data",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "deprecated",
    }});
struct WriteVDB : ExportVDBGrid {
};

static int defWriteVDB = zeno::defNodeClass<WriteVDB>("WriteVDB",
    { /* inputs: */ {
    "data",
    {"writepath", "path"},
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "openvdb",
    }});

}
