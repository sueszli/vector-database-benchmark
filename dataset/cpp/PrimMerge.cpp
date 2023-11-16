#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/types/UserData.h>

namespace zeno {

ZENO_API std::shared_ptr<zeno::PrimitiveObject> primMerge(std::vector<zeno::PrimitiveObject *> const &primList, std::string const &tagAttr) {
    //zeno::log_critical("asdfjhl {}", primList.size());
    //throw;

    auto outprim = std::make_shared<PrimitiveObject>();

    if (primList.size()) {
        std::vector<size_t> bases(primList.size() + 1);
        std::vector<size_t> pointbases(primList.size() + 1);
        std::vector<size_t> linebases(primList.size() + 1);
        std::vector<size_t> tribases(primList.size() + 1);
        std::vector<size_t> quadbases(primList.size() + 1);
        std::vector<size_t> loopbases(primList.size() + 1);
        std::vector<size_t> uvbases(primList.size() + 1);
        std::vector<size_t> polybases(primList.size() + 1);
        size_t total = 0;
        size_t pointtotal = 0;
        size_t linetotal = 0;
        size_t tritotal = 0;
        size_t quadtotal = 0;
        size_t looptotal = 0;
        size_t uvtotal = 0;
        size_t polytotal = 0;
        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto prim = primList[primIdx];
            /// @note promote pure vert prim to point-based prim
            if (!(prim->points.size() || prim->lines.size() || prim->tris.size() || prim->quads.size() || prim->polys.size())) {
                auto nverts = prim->verts.size();
                prim->points.resize(nverts);
                parallel_for(nverts, [&points = prim->points.values](size_t i) {
                    points[i] = i;
                });
            }
            /// 
            total += prim->verts.size();
            pointtotal += prim->points.size();
            linetotal += prim->lines.size();
            tritotal += prim->tris.size();
            quadtotal += prim->quads.size();
            looptotal += prim->loops.size();
            uvtotal += prim->uvs.size();
            polytotal += prim->polys.size();
            bases[primIdx + 1] = total;
            pointbases[primIdx + 1] = pointtotal;
            linebases[primIdx + 1] = linetotal;
            tribases[primIdx + 1] = tritotal;
            quadbases[primIdx + 1] = quadtotal;
            loopbases[primIdx + 1] = looptotal;
            uvbases[primIdx + 1] = uvtotal;
            polybases[primIdx + 1] = polytotal;
        }
        outprim->verts.resize(total);
        outprim->points.resize(pointtotal);
        outprim->lines.resize(linetotal);
        outprim->tris.resize(tritotal);
        outprim->quads.resize(quadtotal);
        outprim->loops.resize(looptotal);
        outprim->uvs.resize(uvtotal);
        outprim->polys.resize(polytotal);

        if (tagAttr.size()) {
            outprim->verts.add_attr<int>(tagAttr);
            outprim->points.add_attr<int>(tagAttr);
            outprim->lines.add_attr<int>(tagAttr);
            outprim->tris.add_attr<int>(tagAttr);
            outprim->quads.add_attr<int>(tagAttr);
            outprim->loops.add_attr<int>(tagAttr);
            outprim->uvs.add_attr<int>(tagAttr);
            outprim->polys.add_attr<int>(tagAttr);
        }
        for (size_t primIdx = 0; primIdx < primList.size(); primIdx++) {
            auto const &prim = primList[primIdx];
            prim->verts.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->verts.add_attr<T>(key);
            });
            prim->points.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->points.add_attr<T>(key);
            });
            prim->lines.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->lines.add_attr<T>(key);
            });
            prim->tris.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->tris.add_attr<T>(key);
            });
            prim->quads.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->quads.add_attr<T>(key);
            });
            prim->loops.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->loops.add_attr<T>(key);
            });
            prim->uvs.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->uvs.add_attr<T>(key);
            });
            prim->polys.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                outprim->polys.add_attr<T>(key);
            });
        }

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto base = bases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->verts.values;
                    } else {
                        return outprim->verts.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->verts.size());
                for (size_t i = 0; i < n; i++) {
                    outarr[base + i] = arr[i];
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->verts.values;
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                } else {
                    auto &outarr = outprim->verts.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->verts.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->verts.values);
            prim->verts.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->verts.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->verts.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = pointbases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->points.values;
                    } else {
                        return outprim->points.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->points.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->points.values;
                    size_t n = std::min(arr.size(), prim->points.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->points.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->points.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->points.values);
            prim->points.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->points.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->points.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = linebases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->lines.values;
                    } else {
                        return outprim->lines.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->lines.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->lines.values;
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->lines.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->lines.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->lines.values);
            prim->lines.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->lines.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->lines.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = tribases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->tris.values;
                    } else {
                        return outprim->tris.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->tris.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->tris.values;
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->tris.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->tris.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->tris.values);
            prim->tris.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->tris.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->tris.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = quadbases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->quads.values;
                    } else {
                        return outprim->quads.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->quads.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->quads.values;
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->quads.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->quads.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->quads.values);
            prim->quads.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->quads.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->quads.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto vbase = bases[primIdx];
            auto base = loopbases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->loops.values;
                    } else {
                        return outprim->loops.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->loops.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = vbase + arr[i];
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->loops.values;
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = vbase + arr[i];
                    }
                } else {
                    auto &outarr = outprim->loops.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->loops.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->loops.values);
            prim->loops.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->loops.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->loops.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto base = uvbases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->uvs.values;
                    size_t n = std::min(arr.size(), prim->uvs.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = base + arr[i];
                    }
                } else {
                    auto &outarr = outprim->uvs.attr<T>(key);
                    size_t n = std::min(arr.size(), prim->uvs.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
            };
            core(std::true_type{}, prim->uvs.values);
            prim->uvs.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->uvs.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->uvs.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });

        parallel_for(primList.size(), [&] (size_t primIdx) {
            auto prim = primList[primIdx];
            auto lbase = loopbases[primIdx];
            auto base = polybases[primIdx];
            auto core = [&] (auto key, auto const &arr) {
                using T = std::decay_t<decltype(arr[0])>;
#if 0
                auto &outarr = [&] () -> auto & {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        return outprim->polys.values;
                    } else {
                        return outprim->polys.attr<T>(key);
                    }
                }();
                size_t n = std::min(arr.size(), prim->polys.size());
                for (size_t i = 0; i < n; i++) {
                    if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                        outarr[base + i] = {arr[i].first + lbase, arr[i].second};
                    } else {
                        outarr[base + i] = arr[i];
                    }
                }
#else
                if constexpr (std::is_same_v<decltype(key), std::true_type>) {
                    auto &outarr = outprim->polys.values;
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = {arr[i][0] + (int)lbase, arr[i][1]};
                    }
                } else {
                    auto &outarr = outprim->polys.add_attr<T>(key);
                    size_t n = std::min(arr.size(), prim->polys.size());
                    for (size_t i = 0; i < n; i++) {
                        outarr[base + i] = arr[i];
                    }
                }
#endif
            };
            core(std::true_type{}, prim->polys.values);
            prim->polys.foreach_attr<AttrAcceptAll>(core);
            if (tagAttr.size()) {
                auto &outarr = outprim->polys.attr<int>(tagAttr);
                for (size_t i = 0; i < prim->polys.size(); i++) {
                    outarr[base + i] = primIdx;
                }
            }
        });
    }

    return outprim;
}

namespace {

struct PrimMerge : INode {
    virtual void apply() override {
        auto primList = get_input<ListObject>("listPrim")->getRaw<PrimitiveObject>();
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        //initialize

        std::vector<std::string> matNameList(0);
        for(auto &p:primList)
        {
            //if p has material
            int matNum = p->userData().get2<int>("matNum",0);
            if(matNum>0)
            {
                //for p's tris, quads...
                //    tris("matid")[i] += matNameList.size();
                for(int i=0; i<p->tris.size();i++)
                {
                    if(p->tris.attr<int>("matid")[i] != -1)
                    {
                        p->tris.attr<int>("matid")[i] += matNameList.size();
                    }
                }
                for(int i=0; i<p->quads.size();i++)
                {
                    if(p->quads.attr<int>("matid")[i] != -1)
                    {
                        p->quads.attr<int>("matid")[i] += matNameList.size();
                    }
                }
                for(int i=0; i<p->polys.size();i++)
                {
                    if(p->polys.attr<int>("matid")[i]!=-1)
                    {
                        p->polys.attr<int>("matid")[i] += matNameList.size();
                    }
                }
                //for p's materials
                //    add them to material list
                for(int i=0;i<matNum;i++)
                {
                    auto matIdx = "Material_" + to_string(i);
                    auto matName = p->userData().get2<std::string>(matIdx, "Default");
                    matNameList.emplace_back(matName);
                }
            }
            else
            {
                //for p's tris, quads...
                //    tris("matid")[] = -1;
                if(p->tris.size()>0) {
                    p->tris.add_attr<int>("matid");
                    p->tris.attr<int>("matid").assign(p->tris.size(), -1);
                }
                if(p->quads.size()>0) {
                    p->quads.add_attr<int>("matid");
                    p->quads.attr<int>("matid").assign(p->quads.size(), -1);
                }
                if(p->polys.size()>0) {
                    p->polys.add_attr<int>("matid");
                    p->polys.attr<int>("matid").assign(p->polys.size(), -1);
                }
            }
        }

        auto outprim = primMerge(primList, tagAttr);

        for(auto &p:primList){
            outprim->userData().merge(p->userData());
        }

        if(matNameList.size()>0)
        {
            //add matNames to userData
            int i=0;
            for(auto name:matNameList)
            {
                auto matIdx = "Material_" + to_string(i);
                outprim->userData().setLiterial(matIdx, name);
                i++;
            }
        }
        int oMatNum = matNameList.size();
        outprim->userData().set2("matNum", oMatNum);
        //auto outprim = std::make_shared<PrimitiveObject>(*primList[0]);
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(PrimMerge, {
    {
        {"list", "listPrim"},
        {"string", "tagAttr", ""},
    },
    {
        {"primitive", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
