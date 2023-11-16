#include <zeno/utils/nowarn.h>
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/vec.h>
#include <voro++/voro++.hh>
#include "EigenUtils.h"
#include "igl_sink.h"
#include <zeno/types/UserData.h>
#include <mutex>
#include <vector>
#include <tuple>

namespace {
using namespace zeno;

struct AABBVoronoi : INode {
    virtual void apply() override {
        auto pieces = std::make_shared<ListObject>();
        auto neighs = std::make_shared<ListObject>();

        auto triangulate = get_param<bool>("triangulate");

        auto bmin = has_input("bboxMin") ?
            get_input<NumericObject>("bboxMin")->get<vec3f>() : vec3f(-1);
        auto bmax = has_input("bboxMax") ?
            get_input<NumericObject>("bboxMax")->get<vec3f>() : vec3f(1);
        auto minx = bmin[0];
        auto miny = bmin[1];
        auto minz = bmin[2];
        auto maxx = bmax[0];
        auto maxy = bmax[1];
        auto maxz = bmax[2];
        auto periX = get_param<bool>("periodicX");
        auto periY = get_param<bool>("periodicY");
        auto periZ = get_param<bool>("periodicZ");

        {
            voro::pre_container pcon(minx,maxx,miny,maxy,minz,maxz,periX,periY,periZ);

            if (has_input("particlesPrim")) {
                auto particlesPrim = get_input<PrimitiveObject>("particlesPrim");
                auto &parspos = particlesPrim->attr<vec3f>("pos");
                for (int i = 0; i < parspos.size(); i++) {
                    auto p = parspos[i];
                    pcon.put(i + 1, p[0], p[1], p[2]);
                }
            } else {
                auto numParticles = get_param<int>("numRandPoints");
                wangsrng rng(numParticles);
                for (int i = 0; i < numParticles; i++) {
                    vec3f p(rng.next_float(),rng.next_float(),rng.next_float());
                    p = p * (bmax - bmin) + bmin;
                    pcon.put(i + 1, p[0], p[1], p[2]);
                }
            }

            int nx, ny, nz;
            pcon.guess_optimal(nx,ny,nz);
            voro::container con(minx,maxx,miny,maxy,minz,maxz,nx,ny,nz,periX,periY,periZ,8);

            /*if (has_input("meshPrim")) {
                auto mesh = get_input<PrimitiveObject>("meshPrim");
                auto meshpos = mesh->attr<zeno::vec3f>("pos");
                for (int i = 0; i < mesh->tris.size(); i++) {
                    auto p = mesh->tris[i];
                    auto n = cross(
                                meshpos[p[0]] - meshpos[p[1]],
                                meshpos[p[0]] - meshpos[p[2]]);
                    n *= 1 / (length(n) + 1e-6);
                    auto c = dot(meshpos[p[0]], n);
                    printf("%f %f %f %f\n", n[0], n[1], n[2], c);
                    voro::wall_plane wal(n[0], n[1], n[2], c);
                    con.add_wall(wal);
                }
            }*/
            pcon.setup(con);

            int cid = 0;
            voro::c_loop_all cl(con);
            voro::voronoicell_neighbor c;
            if(cl.start()) do if(con.compute_cell(c, cl)) {
                double x, y, z;
                cl.pos(x, y, z);

                std::vector<int> neigh, f_vert;
                c.neighbors(neigh);
                c.face_vertices(f_vert);
                std::vector<double> v;
                c.vertices(x, y, z, v);

                auto prim = std::make_shared<PrimitiveObject>();

                auto &pos = prim->add_attr<vec3f>("pos");
                for (int i = 0; i < (int)v.size(); i += 3) {
                    pos.emplace_back(v[i], v[i+1], v[i+2]);
                }
                prim->resize(pos.size());

                bool isBoundary = false;
                for (int i = 0, j = 0; i < (int)neigh.size(); i++) {
                    if (neigh[i] <= 0) {
                        isBoundary = true;
                    } else {
                        if (auto ncid = neigh[i] - 1; ncid > cid) {
                            neighs->arr.push_back(objectFromLiterial(vec2i(cid, ncid)));
                        }
                    }
                    int len = f_vert[j];
                    int start = (int)prim->loops.size();
                    for (int k = j + 1; k < j + 1 + len; k++) {
                        prim->loops.push_back(f_vert[k]);
                    }
                    prim->polys.emplace_back(start, len);
                    j = j + 1 + len;
                }

                prim->userData().set("isBoundary", std::make_shared<NumericObject>(isBoundary));
                pieces->arr.push_back(std::move(prim));

                cid++;
            } while (cl.inc());
        }

        log_info("AABBVoronoi got {} pieces, {} neighs", pieces->arr.size(), neighs->arr.size());

        if (triangulate) {
            for (auto const &prim: pieces->get<PrimitiveObject>()) {
                prim_triangulate(prim.get());
            }
        }

        set_output("primList", std::move(pieces));
        set_output("neighList", std::move(neighs));
    }
};

ZENO_DEFNODE(AABBVoronoi)({
        { // inputs:
        {"PrimitiveObject", "particlesPrim"},
        {"vec3f", "bboxMin", "-1,-1,-1"},
        {"vec3f", "bboxMax", "1,1,1"},
        },
        { // outputs:
        {"ListObject", "primList"},
        {"ListObject", "neighList"},
        },
        { // params:
        {"bool", "triangulate", "1"},
        {"int", "numRandPoints", "64"},
        {"bool", "periodicX", "0"},
        {"bool", "periodicY", "0"},
        {"bool", "periodicZ", "0"},
        },
        {"cgmesh"},
});


struct VoronoiFracture : AABBVoronoi {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("meshPrim");
        auto VFA = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primA.get()) : prim_to_eigen(primA.get());

        auto bmin = primA->verts.size() ? primA->verts[0] : vec3f(0);
        auto bmax = bmin;
        for (int i = 1; i < primA->verts.size(); i++) {
            bmin = zeno::min(primA->verts[i], bmin);
            bmax = zeno::max(primA->verts[i], bmax);
        }
        bmin -= 1e-6f;
        bmax += 1e-6f;
        inputs["bboxMin"] = objectFromLiterial(bmin);
        inputs["bboxMax"] = objectFromLiterial(bmax);
        inputs["triangulate:"] = objectFromLiterial(true);

        AABBVoronoi::apply();

        auto primListB = std::dynamic_pointer_cast<ListObject>(outputs.at("primList"));
        auto neighListB = std::dynamic_pointer_cast<ListObject>(outputs.at("neighList"));
        auto listB = primListB->get<PrimitiveObject>();
        std::map<int, std::shared_ptr<PrimitiveObject>> dictC;
        std::mutex mtx;

        #pragma omp parallel for
        for (int i = 0; i < listB.size(); i++) {
            log_debug("VoronoiFracture: processing fragment #{}...", i);
            auto const &primB = listB[i];
            auto [VB, FB] = get_param<bool>("doMeshFix2") ? prim_to_eigen_with_fix(primB.get()) : prim_to_eigen(primB.get());
            Eigen::MatrixXd VC;
            Eigen::MatrixXi FC;
            Eigen::VectorXi J;
            igl_mesh_boolean(VFA.first, VFA.second, VB, FB, "Intersect", VC, FC, J);
            if (VC.size() != 0) {
                bool anyFromA = false;
                for (int i = 0; i < J.size(); i++) {
                    if (J(i) < VFA.second.rows()) {
                        anyFromA = true;
                    }
                }
                auto primC = std::make_shared<PrimitiveObject>();
                eigen_to_prim(VC, FC, primC.get());
                primC->userData().set("isBoundary", objectFromLiterial(anyFromA));
                std::lock_guard _(mtx);
                dictC.emplace(i, std::move(primC));
            } else {
                log_debug("null piece encountered at #{}, removing...", i);
            }
        }

        auto neighB = neighListB->getLiterial<zeno::vec2i>();
        auto primListC = std::make_shared<ListObject>();
        std::map<int, int> dictD;
        for (auto const &[key, prim]: dictC) {
            dictD[key] = primListC->arr.size();
            primListC->arr.push_back(prim);
        }

        auto neighListC = std::make_shared<ListObject>();
        for (auto const &c: neighB) {
            if (auto xit = dictD.find(c[0]); xit != dictD.end()) {
                if (auto yit = dictD.find(c[1]); yit != dictD.end()) {
                    zeno::vec2i c2(xit->second, yit->second);
                    //log_trace("VoronoiFracture: neigh {} and {}", c2[0], c2[1]);
                    //auto ret = std::make_shared<NumericObject>(); ret->set(c2);
                    neighListC->arr.push_back(objectFromLiterial(c2));
                }
            }
        }

        log_info("VoronoiFracture got {} pieces, {} neighs", primListC->arr.size(), neighListC->arr.size());

        set_output("primList", std::move(primListC));
        set_output("neighList", std::move(neighListC));
    }
};

ZENO_DEFNODE(VoronoiFracture)({
        { // inputs:
        {"PrimitiveObject", "meshPrim"},
        {"PrimitiveObject", "particlesPrim"},
        },
        { // outputs:
        {"ListObject", "primList"},
        {"ListObject", "neighList"},
        },
        { // params:
        {"bool", "doMeshFix", "0"},
        {"bool", "doMeshFix2", "1"},
        {"int", "numRandPoints", "256"},
        {"bool", "periodicX", "0"},
        {"bool", "periodicY", "0"},
        {"bool", "periodicZ", "0"},
        },
        {"cgmesh"},
});


//struct VoronoiFracture : VoronoiFractureSingle {
    //virtual void apply() override {
        //if (!get_param<bool>("splitIsland")) {
            //VoronoiFractureSingle::apply();
            //return;
        //}
        //auto prim = get_input<PrimitiveObject>("meshPrim");
        //std::string islandAttr = "**" + std::to_string(wangsrng(prim->size()).next_uint32());
        //primMarkIsland(prim.get(), islandAttr);
        //auto prims = primUnmergeVerts(prim.get(), islandAttr);
        //zeno::log_info("VoronoiFracture split got {} islands", prims.size());
        //prim->verts.erase_attr(islandAttr);
        //auto primfinlst = std::make_shared<ListObject>();
        //auto neighfinlst = std::make_shared<ListObject>();
        //int redprimcount = 0, islandid = 0;
        //for (auto const &prim1: prims) {
            //zeno::log_debug("VoronoiFracture: fracturating island #{}", islandid);
            //inputs["meshPrim"] = prim1;
            //VoronoiFractureSingle::apply();
            //auto *prim1lst = static_cast<ListObject *>(outputs.at("primList").get());
            //auto *neigh1lst = static_cast<ListObject *>(outputs.at("neighList").get());
            //for (auto const &nei1li: neigh1lst->getLiterial<vec2i>()) {
                //neighfinlst->arr.push_back(std::make_shared<NumericObject>(nei1li + redprimcount));
            //}
            //for (auto const &prim1li: prim1lst->get<PrimitiveObject>()) {
                //primfinlst->arr.push_back(prim1li);
                //redprimcount++;
            //}
            //islandid++;
        //}
        //set_output("primList", std::move(primfinlst));
        //set_output("neighList", std::move(neighfinlst));
    //}
//};

//ZENO_DEFNODE(VoronoiFracture)({
        //{ // inputs:
        //{"PrimitiveObject", "meshPrim"},
        //{"PrimitiveObject", "particlesPrim"},
        //},
        //{ // outputs:
        //{"ListObject", "primList"},
        //{"ListObject", "neighList"},
        //},
        //{ // params:
        //{"bool", "splitIsland", "0"},
        //{"bool", "doMeshFix", "0"},
        //{"bool", "doMeshFix2", "1"},
        //{"int", "numRandPoints", "256"},
        //{"bool", "periodicX", "0"},
        //{"bool", "periodicY", "0"},
        //{"bool", "periodicZ", "0"},
        //},
        //{"cgmesh"},
//});


struct SimplifyVoroNeighborList : INode {
    virtual void apply() override {
        auto neighList = get_input<ListObject>("neighList");
        auto newNeighList = std::make_shared<ListObject>();

        std::map<int, std::vector<int>> lut;
        for (auto const &ind: neighList->getLiterial<vec2i>()) {
            auto x = ind[0], y = ind[1];
            lut[x].push_back(y);
            lut[y].push_back(x);
        }

        std::set<int> visited;
        std::vector<std::pair<int, int>> edges;
        auto touch = [&] (auto touch, int x) -> void {
            visited.insert(x);
            for (int y: lut.at(x)) {
                if (visited.find(y) == visited.end()) {
                    edges.emplace_back(x, y);
                    touch(touch, y);
                }
            }
        };
        for (auto const &[x, ys]: lut) {
            touch(touch, x);
        }

        for (auto const &[x, y]: edges) {
            newNeighList->arr.push_back(objectFromLiterial(vec2i(x, y)));
        }
        set_output("newNeighList", std::move(newNeighList));
    }
};

ZENO_DEFNODE(SimplifyVoroNeighborList)({
        { // inputs:
        {"ListObject", "neighList"},
        },
        { // outputs:
        {"ListObject", "newNeighList"},
        },
        { // params:
        },
        {"cgmesh"},
});

}
