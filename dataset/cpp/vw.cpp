#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/ChangeBackground.h>
#include <zeno/VDBGrid.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"
#include <zeno/StringObject.h>
#include <zeno/utils/zeno_p.h>

namespace zeno {
    std::string preApplyRefs(const std::string& code, Graph* pGraph);

namespace {

static zfx::Compiler compiler;
static zfx::x64::Assembler assembler;

template <class GridPtr>
void vdb_wrangle(zfx::x64::Executable *exec, GridPtr &grid, bool modifyActive, bool changeBackground, bool hasPos) {
    //ZENO_P(grid->background());
    auto wrangler = [&](auto &leaf, openvdb::Index leafpos) {
        std::visit([&] (auto hasPos) {
            for (auto iter = leaf.beginValueOn(); iter != leaf.endValueOn(); ++iter) {
                iter.modifyValue([&](auto &v) {
                    auto ctx = exec->make_context();
                    if constexpr (std::is_same_v<std::decay_t<decltype(v)>, openvdb::Vec3f>) {
                        ctx.channel(0)[0] = v[0];
                        ctx.channel(1)[0] = v[1];
                        ctx.channel(2)[0] = v[2];
                        if (hasPos) {
                            openvdb::Vec3f p = grid->transformPtr()->indexToWorld(iter.getCoord());
                            ctx.channel(3)[0] = p[0];
                            ctx.channel(4)[0] = p[1];
                            ctx.channel(5)[0] = p[2];
                        }
                        ctx.execute();
                        v[0] = ctx.channel(0)[0];
                        v[1] = ctx.channel(1)[0];
                        v[2] = ctx.channel(2)[0];
        
                    } else {
                        ctx.channel(0)[0] = v;
                        if (hasPos) {
                            openvdb::Vec3f p = grid->transformPtr()->indexToWorld(iter.getCoord());
                            ctx.channel(1)[0] = p[0];
                            ctx.channel(2)[0] = p[1];
                            ctx.channel(3)[0] = p[2];
                        }
                        ctx.execute();
                        v = ctx.channel(0)[0];

                    }
                    
                });

                if(modifyActive){
                    float testv;
                    auto v = iter.getValue();
                    if constexpr (std::is_same_v<std::decay_t<decltype(v)>, openvdb::Vec3f>)
                    {
                        testv = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
                    } else {
                        testv = std::abs(v);
                    }
                    if(testv<1e-5)
                    {
                        iter.setValueOn(false);
                    }
                    else{
                        iter.setValueOn(true);
                    }
                }
            }
        }, boolean_variant(hasPos));
    };
    auto velman = openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>>(grid->tree());
    velman.foreach(wrangler);
    if (changeBackground) {
        auto v = grid->background();
        {
            auto ctx = exec->make_context();
            openvdb::Vec3f p(0, 0, 0);
                    if constexpr (std::is_same_v<std::decay_t<decltype(v)>, openvdb::Vec3f>) {
                        ctx.channel(0)[0] = v[0];
                        ctx.channel(1)[0] = v[1];
                        ctx.channel(2)[0] = v[2];
                        if (hasPos) {
                            ctx.channel(3)[0] = p[0];
                            ctx.channel(4)[0] = p[1];
                            ctx.channel(5)[0] = p[2];
                        }
                        ctx.execute();
                        v[0] = ctx.channel(0)[0];
                        v[1] = ctx.channel(1)[0];
                        v[2] = ctx.channel(2)[0];
        
                    } else {
                        ctx.channel(0)[0] = v;
                        if (hasPos) {
                            ctx.channel(1)[0] = p[0];
                            ctx.channel(2)[0] = p[1];
                            ctx.channel(3)[0] = p[2];
                        }
                        ctx.execute();
                        v = ctx.channel(0)[0];

                    }
        }
        openvdb::tools::changeBackground(grid->tree(), v);
    }
    openvdb::tools::prune(grid->tree());
}

struct VDBWrangle : zeno::INode {
    virtual void apply() override {
        auto grid = get_input<zeno::VDBGrid>("grid");
        auto code = get_input<zeno::StringObject>("zfxCode")->get();

        auto hasPos = code.find("@pos") != code.npos;

        zfx::Options opts(zfx::Options::for_x64);
        if (std::dynamic_pointer_cast<zeno::VDBFloatGrid>(grid))
            opts.define_symbol("@val", 1);
        else if (std::dynamic_pointer_cast<zeno::VDBFloat3Grid>(grid))
            opts.define_symbol("@val", 3);
        else
            dbg_printf("unexpected vdb grid type");
        if (hasPos)
            opts.define_symbol("@pos", 3);
        opts.reassign_channels = false;

        auto params = has_input("params") ?
            get_input<zeno::DictObject>("params") :
            std::make_shared<zeno::DictObject>();
        {
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        auto const &gs = *this->getGlobalState();
        params->lut["PI"] = objectFromLiterial((float)(std::atan(1.f) * 4));
        params->lut["F"] = objectFromLiterial((float)gs.frameid);
        params->lut["DT"] = objectFromLiterial(gs.frame_time);
        params->lut["T"] = objectFromLiterial(gs.frame_time * gs.frameid + gs.frame_time_elapsed);
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动有$F$DT$T做参数
        // BEGIN心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        for (auto const &[key, ref]: getThisGraph()->portalIns) {
            if (auto i = code.find('$' + key); i != std::string::npos) {
                i = i + key.size() + 1;
                if (code.size() <= i || !std::isalnum(code[i])) {
                    if (params->lut.count(key)) continue;
                    dbg_printf("ref portal %s\n", key.c_str());
                    auto res = getThisGraph()->callTempNode("PortalOut",
                          {{"name:", objectFromLiterial(key)}}).at("port");
                    params->lut[key] = std::move(res);
                }
            }
        }
        // END心欣你也可以把这段代码加到其他wrangle节点去，这样这些wrangle也可以自动引用portal做参数
        // BEGIN伺候心欣伺候懒得extract出变量了
        std::vector<std::string> keys;
        for (auto const &[key, val]: params->lut) {
            keys.push_back(key);
        }
        for (auto const &key: keys) {
            if (!dynamic_cast<zeno::NumericObject*>(params->lut.at(key).get())) {
                dbg_printf("ignored non-numeric %s\n", key.c_str());
                params->lut.erase(key);
            }
        }
        // END伺候心欣伺候懒得extract出变量了
        }
        std::vector<float> parvals;
        std::vector<std::pair<std::string, int>> parnames;
        for (auto const &[key_, par]: params->getLiterial<zeno::NumericValue>()) {
            auto key = '$' + key_;
            auto dim = std::visit([&] (auto const &v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_convertible_v<T, zeno::vec3f>) {
                    parvals.push_back(v[0]);
                    parvals.push_back(v[1]);
                    parvals.push_back(v[2]);
                    parnames.emplace_back(key, 0);
                    parnames.emplace_back(key, 1);
                    parnames.emplace_back(key, 2);
                    return 3;
                } else if constexpr (std::is_convertible_v<T, zeno::vec2f>) {
                    parvals.push_back(v[0]);
                    parvals.push_back(v[1]);
                    parnames.emplace_back(key, 0);
                    parnames.emplace_back(key, 1);
                    return 2;
                } else if constexpr (std::is_convertible_v<T, float>) {
                    parvals.push_back(v);
                    parnames.emplace_back(key, 0);
                    return 1;
                } else return 0;
            }, par);
            opts.define_param(key, dim);
        }
        if (1)
        {
            // BEGIN 引用预解析：将其他节点参数引用到此处，可能涉及提前对该参数的计算
            // 方法是: 搜索code里所有ref(...)，然后对于每一个ref(...)，解析ref内部的引用，
            // 然后将计算结果替换对应ref(...)，相当于预处理操作。
            code = preApplyRefs(code, getThisGraph());
            // END 引用预解析
        }

        auto prog = compiler.compile(code, opts);
        auto exec = assembler.assemble(prog->assembly);

        std::vector<float> pars(prog->params.size());
        for (int i = 0; i < pars.size(); i++) {
            auto [name, dimid] = prog->params[i];
            assert(name[0] == '$');
            dbg_printf("parameter %d: %s.%d\n", i, name.c_str(), dimid);
            auto it = std::find(parnames.begin(),
                parnames.end(), std::pair{name, dimid});
            auto value = parvals.at(it - parnames.begin());
            dbg_printf("(valued %f)\n", value);
            exec->parameter(prog->param_id(name, dimid)) = value;
        }
        auto modifyActive = has_input("ModifyActive") ?
            (get_input<zeno::StringObject>("ModifyActive")->get())=="true" : false;
        auto changeBackground = has_input("ChangeBackground") ?
            (get_input<zeno::StringObject>("ChangeBackground")->get())=="true" : false;
        if (auto p = std::dynamic_pointer_cast<zeno::VDBFloatGrid>(grid); p)
            vdb_wrangle(exec, p->m_grid, modifyActive, changeBackground, hasPos);
        else if (auto p = std::dynamic_pointer_cast<zeno::VDBFloat3Grid>(grid); p)
            vdb_wrangle(exec, p->m_grid, modifyActive, changeBackground, hasPos);

        set_output("grid", std::move(grid));
    }
};

ZENDEFNODE(VDBWrangle, {
    {{"VDBGrid", "grid"}, {"string", "zfxCode"},
     {"enum true false","ModifyActive","false"},
     {"enum true false","ChangeBackground","false"},
     {"DictObject:NumericObject", "params"}},
    {{"VDBGrid", "grid"}},
    {},
    {"zenofx"},
});

}
}
