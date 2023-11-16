#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <cstring>
#include <cstdlib>
#include <random>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {
namespace {

struct randtype_scalar01 {
    auto operator()(wangsrng &rng) const {
        float offs{rng.next_float()};
        return offs;
    }
};

struct randtype_scalar11 {
    auto operator()(wangsrng &rng) const {
        float offs{rng.next_float()};
        return offs * 2 - 1;
    }
};

struct randtype_cube01 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float(), rng.next_float(), rng.next_float()};
        return offs;
    }
};

struct randtype_cube11 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float(), rng.next_float(), rng.next_float()};
        return offs * 2 - 1;
    }
};

struct randtype_plane01 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float(), rng.next_float(), 0};
        return offs;
    }
};

struct randtype_plane11 {
    auto operator()(wangsrng &rng) const {
        vec3f offs{rng.next_float() * 2 - 1, rng.next_float() * 2 - 1, 0};
        return offs;
    }
};

struct randtype_disk {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r1 = std::sqrt(r1);
        r2 *= M_PI * 2;
        vec3f offs{r1 * std::sin(r2), r1 * std::cos(r2), 0};
        return offs;
    }
};

struct randtype_cylinder {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r1 = r1 * 2 - 1;
        r2 *= M_PI * 2;
        vec3f offs{std::sin(r2), std::cos(r2), r1};
        return offs;
    }
};

struct randtype_ball {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        float r3 = rng.next_float();
        r1 = r1 * 2 - 1;
        r2 *= M_PI * 2;
        r3 = std::cbrt(r3) * std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

struct randtype_semiball {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        float r3 = rng.next_float();
        r2 *= M_PI * 2;
        r3 = std::cbrt(r3) * std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

struct randtype_sphere {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r1 = r1 * 2 - 1;
        r2 *= M_PI * 2;
        float r3 = std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

struct randtype_semisphere {
    auto operator()(wangsrng &rng) const {
        float r1 = rng.next_float();
        float r2 = rng.next_float();
        r2 *= M_PI * 2;
        float r3 = std::sqrt(1 - r1 * r1);
        vec3f offs{r3 * std::sin(r2), r3 * std::cos(r2), r1};
        return offs;
    }
};

using RandTypes = std::variant
    < randtype_scalar01
    , randtype_scalar11
    , randtype_cube01
    , randtype_cube11
    , randtype_plane01
    , randtype_plane11
    , randtype_disk
    , randtype_cylinder
    , randtype_ball
    , randtype_semiball
    , randtype_sphere
    , randtype_semisphere
>;

static std::string_view lutRandTypes[] = {
    "scalar01",
    "scalar11",
    "cube01",
    "cube11",
    "plane01",
    "plane11",
    "disk",
    "cylinder",
    "ball",
    "semiball",
    "sphere",
    "semisphere",
};

}

ZENO_API void primRandomize(PrimitiveObject *prim, std::string attr, std::string dirAttr, std::string seedAttr, std::string randType, float base, float scale, int seed) {
    auto randty = enum_variant<RandTypes>(array_index_safe(lutRandTypes, randType, "randType"));
    auto seedSel = functor_variant(seedAttr.empty() ? 0 : 1, [&] {
        return [] (int i) {
            return i;
        };
    }, [&, &seedAttr = seedAttr] {
        auto &seedArr = prim->verts.attr<int>(seedAttr);
        return [&] (int i) {
            return seedArr[i];
        };
    });
    auto hasDirArr = boolean_variant(!dirAttr.empty());
    if (seed == -1) seed = std::random_device{}();
    std::visit([&] (auto const &randty, auto const &seedSel, auto hasDirArr) {
        using T = std::invoke_result_t<std::decay_t<decltype(randty)>, wangsrng &>;
        auto &arr = prim->verts.add_attr<T>(attr);
        auto const &dirArr = hasDirArr ? prim->attr<vec3f>(dirAttr) : std::vector<vec3f>();
        parallel_for((size_t)0, arr.size(), [&] (size_t i) {
            wangsrng rng(seed, seedSel(i));
            T offs = base + randty(rng) * scale;

            if constexpr (hasDirArr.value && std::is_same_v<T, vec3f>) {
                vec3f dir = dirArr[i], b1, b2;
                pixarONB(dir, b1, b2);
                offs = offs[0] * b1 + offs[1] * b2 + offs[2] * dir;
            }

            arr[i] = offs;
        });
    }, randty, seedSel, hasDirArr);
}

namespace {

struct PrimRandomize : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto base = get_input2<float>("base");
        auto scale = get_input2<float>("scale");
        auto seed = get_input2<int>("seed");
        auto attr = get_input2<std::string>("attr");
        auto dirAttr = get_input2<std::string>("dirAttr");
        auto seedAttr = get_input2<std::string>("seedAttr");
        auto randType = get_input2<std::string>("randType");
        primRandomize(prim.get(), attr, dirAttr, seedAttr, randType, base, scale, seed);
        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(PrimRandomize, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "attr", "tmp"},
    {"string", "dirAttr", ""},
    {"string", "seedAttr", ""},
    {"float", "base", "0"},
    {"float", "scale", "1"},
    {"int", "seed", "-1"},
    {"enum scalar01 scalar11 cube01 cube11 plane01 plane11 disk cylinder ball semiball sphere semisphere", "randType", "scalar01"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
