#include "test_mixin_allocator.hpp"

#include <dynamix/domain.hpp>
#include <dynamix/feature_info_data.hpp>
#include <dynamix/mixin_info_data.hpp>
#include <dynamix/object.hpp>
#include <dynamix/mutate.hpp>
#include <dynamix/exception.hpp>
#include <dynamix/dbg_dmp.hpp>

#include <doctest/doctest.h>

#include <itlib/qalgorithm.hpp>
#include <itlib/flat_set.hpp>

#include <thread>
#include <deque>
#include <cstdlib>
#include <random>
#include <array>
#include <iostream>

static constexpr int NUM_FEATURES = 50;
static constexpr int NUM_DEPS = 15;
static constexpr int NUM_MIXINS = 100;
static constexpr int SIZE = 1000;
static constexpr int MAX_FAILS = 850;

dynamix::error_return_t count_init_func(const dnmx_mixin_info* info, void* vptr) {
    auto ptr = static_cast<dynamix::byte_t*>(vptr);
    for (dynamix::byte_size_t i = 0; i < info->size; ++i) {
        *ptr++ = dynamix::byte_t(i);
    }
    return dynamix::result_success;
}

template <typename T>
void shuffle(std::vector<T>& vec, std::minstd_rand& rnd) {
    // don't use std::shuffle as it has different implementations in
    // different standard libraries and breaks the fuzz test determinism with a given seed
    for (int i = 0; i < 2; ++i) {
        for (auto& elem : vec) {
            std::swap(elem, vec[rnd() % vec.size()]);
        }
    }
}

struct object_producer {
    dynamix::domain& dom;
    const std::deque<dynamix::util::mixin_info_data>& mixins;
    uint32_t seed;
    std::minstd_rand rnd;
    std::vector<dynamix::object> objects;
    std::vector<dynamix::object> copies;

    object_producer(dynamix::domain& d, const std::deque<dynamix::util::mixin_info_data>& mix, uint32_t seed)
        : dom(d)
        , mixins(mix)
        , seed(seed)
        , rnd(seed)
    {}

    void produce() {
        // intentionally not reserving
        // we want the vector to move things around
        int fails = 0;
        while (objects.size() != SIZE) {
            auto num_mixins = rnd() % 10 + 1;

            if (objects.size() == 1) {
                // have the second object empty
                num_mixins = 0;
            }

            itlib::flat_set<const dynamix::mixin_info*> mix;
            while (mix.size() != num_mixins) mix.insert(&mixins[rnd() % mixins.size()].info); // generate unique infos
            auto mix_shuf = std::move(mix.modify_container());
            shuffle(mix_shuf, rnd); // but not sorted

            try {
                auto& t = dom.get_type(mix_shuf);
                {
                    dynamix::object o(dom);
                    o.reset_type(t);
                    objects.push_back(std::move(o));
                }
                const auto& o = objects.back();
                fails = 0;

                if (rnd() % 2) {
                    // try object::copy() 50% of the time
                    copies.emplace_back(o.copy());
                }
                else {
                    dynamix::object c(dom);
                    c.copy_from(o);
                    copies.push_back(std::move(c));
                }
            }
            catch (const dynamix::exception&) {
                ++fails;
                if (fails == MAX_FAILS) {
                    std::cout << "producer " << seed << " failed more than " << MAX_FAILS << " consecutive times\n";
                    return;
                }
                continue;
            }
        }
    }
};

class custom_rule {
    dynamix::mutation_rule_info m_info;
    const dynamix::mixin_info& m_primary;
    const dynamix::mixin_info& m_dep;
    bool m_use_primary_name;
    bool m_use_dep_name;
    std::string m_name;
public:
    custom_rule(const dynamix::mixin_info& primary, const dynamix::mixin_info& dep, bool pname, bool dname)
        : m_primary(primary)
        , m_dep(dep)
        , m_use_primary_name(pname)
        , m_use_dep_name(dname)
    {
        m_name = "r ";
        m_name += m_primary.name.to_std();
        if (m_use_primary_name) {
            m_name += "(name)";
        }
        else {
            m_name += "(info)";
        }

        m_name += " drags ";

        m_name += m_dep.name.to_std();
        if (m_use_dep_name) {
            m_name += "(name)";
        }
        else {
            m_name += "(info)";
        }

        m_info.apply = apply;
        m_info.name = dnmx_sv::from_std(m_name);
        m_info.order_priority = 0;
        m_info.user_data = reinterpret_cast<uintptr_t>(this);
    }

    static dynamix::error_return_t apply(dnmx_type_mutation_handle mutation, uintptr_t user_data) {
        auto mut = dynamix::type_mutation::from_c_handle(mutation);
        auto self = reinterpret_cast<custom_rule*>(user_data);
        bool has_prim;
        if (self->m_use_primary_name) {
            has_prim = mut->has(self->m_primary.name.to_std());
        }
        else {
            has_prim = mut->has(self->m_primary);
        }

        if (!has_prim) return dynamix::result_success;

        if (self->m_use_dep_name) {
            mut->add_if_lacking(self->m_dep.name.to_std());
        }
        else {
            mut->add_if_lacking(self->m_dep);
        }

        return dynamix::result_success;
    }

    custom_rule(const custom_rule&) = delete;
    custom_rule& operator=(const custom_rule&) = delete;

    const dynamix::mutation_rule_info& info() const noexcept { return m_info; };
};

TEST_CASE("fuzz objects and types") {
    const unsigned initial_seed = std::random_device{}();
    // const unsigned initial_seed = 1283054047;
    printf("initial seed: %u\n", initial_seed);
    std::minstd_rand seeder(initial_seed);

    std::minstd_rand rnd(seeder());

    std::deque<dynamix::util::feature_info_data> features;
    // generate features
    for (int i = 0; i < NUM_FEATURES; ++i) {
        auto& f = features.emplace_back();
        dynamix::util::feature_info_data_builder b(f, "");
        auto name = "feature_" + std::to_string(i);
        b.store_name(name);

        auto op = rnd() % 3;
        if (op == 0) {
            b.default_payload_by(name);
        }
        else if (op == 1) {
            b.default_payload_with(dynamix::compat::pmr::string("fff"));
        }

        f.info.feature_class = rnd() % 5;
        f.info.allow_clashes = rnd() % 20 != 0; // allow clashes 95% of the time
    }

    test_mixin_allocator alloc;
    auto create_mixin = [&](dynamix::util::mixin_info_data& m, int i, bool dep) {
        auto& info = m.info;
        dynamix::util::mixin_info_data_builder b(m, "");
        if (dep) {
            b.store_name("dep_" + std::to_string(i));
            b.dependency();
        }
        else {
            b.store_name("mixin_" + std::to_string(i));
        }

        // funcs
        {
            // override init funcs from builder
            info.init = count_init_func;
            info.move_init = dnmx_mixin_common_move_func;
            info.move_asgn = dnmx_mixin_common_move_func;
            info.copy_init = dnmx_mixin_common_copy_func;
            info.copy_asgn = dnmx_mixin_common_copy_func;
            info.compare = dnmx_mixin_common_cmp_func;
            info.destroy = dnmx_mixin_common_destroy_func;

            if (!dep && i % 2 == 1 && (rnd() % 3 == 0)) {
                // no default ctor 16% of the time
                // but leave at least half of the mixins with a default ctor
                // and only if not a dependency
                info.init = nullptr;
            }

            if (rnd() % 20 == 0) {
                // no move 5% of the time
                info.move_init = nullptr;
                info.move_asgn = nullptr;
            }

            if (rnd() % 5 == 0) {
                // no copy 20% of the time
                info.copy_init = nullptr;
                info.copy_asgn = nullptr;
            }

            if (rnd() % 2 == 0) {
                // no compare 50% of the time
                info.compare = nullptr;
                if (rnd() % 2) {
                    // ... but have an equals func 25% of the time
                    info.equals = dnmx_mixin_common_eq_func;
                }
            }

            if (rnd() % 2 == 0) {
                // no destroy 50% of the time
                info.destroy = nullptr;
            }
        }

        // size, aligment
        {
            static constexpr std::array<dynamix::byte_size_t, 7> alignments = {0, 1, 2, 4, 8, 16, 32};
            auto alignment = alignments[rnd() % alignments.size()];
            auto size = alignment * (rnd() % 10 + 1);
            info.set_size_alignment(size, alignment);
        }

        // features
        {
            auto max_features = dep ? 3 : 13;
            auto num_features = rnd() % max_features;
            itlib::flat_set<uint32_t> fids;
            while (fids.size() != num_features) fids.insert(rnd() % NUM_FEATURES); // generate unique indices
            auto fids_shuf = std::move(fids.modify_container());
            shuffle(fids_shuf, rnd); // but not sorted

            for (auto fid : fids_shuf) {
                auto& name = m.stored_name;
                auto& f = features[fid];

                dynamix::util::builder_perks perks;
                perks.bid = rnd() % 3 - 1;
                perks.priority = rnd() % 3 - 1;

                auto op = rnd() % 2;
                if (op == 0) {
                    dynamix::compat::pmr::string ffm = name + '-' + f.stored_name;
                    ffm += f.info.name.to_std();
                    b.implements_with(f.info, ffm, perks);
                }
                else {
                    b.implements_by(f.info, name, perks);
                }
            }
        }

        // other

        if (rnd() % 5 == 0) {
            // custom allocator 20% of the time
            if (rnd() % 2) {
                b.uses_allocator(alloc);
            }
            else {
                b.uses_allocator<test_mixin_allocator>();
            }
        }

        info.mixin_class = rnd() % 10;
        info.type_info_class = rnd() % 20;
        info.user_data = rnd() % 100;

        info.force_external = rnd() % 50 == 0; // force external 2% of the time
    };

    // generate deps
    std::deque<dynamix::util::mixin_info_data> deps;
    for (int i = 0; i < NUM_DEPS; ++i) {
        auto& m = deps.emplace_back();
        create_mixin(m, i, true);
    }

    // generate mixins
    std::deque<dynamix::util::mixin_info_data> mixins;
    for (int i = 0; i < NUM_MIXINS; ++i) {
        auto& m = mixins.emplace_back();
        create_mixin(m, i, false);
    }

    // add rules
    std::deque<custom_rule> rules;
    for (auto& d : deps) {
        auto num = rnd() % 2 + 1;
        for (uint32_t i = 0; i < num; ++i) {
            auto to = rnd() % NUM_MIXINS;
            auto& m = mixins[to];
            bool mname = rnd() % 2 == 0;
            bool dname = rnd() % 2 == 0;
            auto& r = rules.emplace_back(m.info, d.info, mname, dname);
            d.mutation_rule_infos.push_back(&r.info());
        }
    }

    dynamix::domain dom("fot");
    for (auto& m : mixins) {
        m.register_in(dom);
    }
    for (auto& d : deps) {
        d.register_in(dom);
    }

    std::deque<object_producer> producers;
    for (int i = 0; i < 3; ++i) {
        producers.emplace_back(dom, mixins, seeder());
    }

    // dynamix::util::dbg_dmp(std::cout, dom);

    // producers[0].produce();

    std::vector<std::thread> threads;
    for (auto& p : producers) {
        threads.emplace_back([&]() { p.produce(); });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (auto& p : producers) {
        CHECK(p.objects.size() == SIZE);
        REQUIRE_FALSE(p.copies.empty());
        auto ci = p.copies.begin();
        for (auto& o : p.objects) {
            if (!o.get_type().copyable()) continue;
            if (o.get_type().equality_comparable()) {
                CHECK(o.equals(*ci));
            }
            ++ci;
        }
    }
}
