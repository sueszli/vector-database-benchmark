#include <zeno/zeno.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DummyObject.h>
#include <zeno/extra/ContextManaged.h>
#include <zeno/extra/evaluate_condition.h>
#include <zeno/utils/safe_at.h>

namespace zeno {

struct IBeginFor : zeno::INode {
private:
    bool m_is_break = false;
    //bool m_updated = false;

protected:
    virtual bool isContinue() const = 0;
    virtual void update() = 0;
    virtual void execute() = 0;

public:
    void breakThisFor() {
        m_is_break = true;
    }

    bool checkIsContinue() const {
        return !m_is_break && isContinue();
    }

    void doUpdate() {
        //m_updated = true;
        update();
    }

    virtual void apply() override final {
        //if (!m_updated)
            //throw makeError("BeginFor and EndFor not enclosed! "
                            //"Don't VIEW any nodes between BeginFor and EndFor, "
                            //"VIEW the EndFor node instead");
        m_is_break = false;
        execute();
    }
};


struct BeginFor : IBeginFor {
    int m_index = 0;
    int m_count = 0;
    
    virtual bool isContinue() const override final {
        return m_index < m_count;
    }

    virtual void execute() override final {
        m_index = 0;
        m_count = get_input<zeno::NumericObject>("count")->get<int>();
        set_output("FOR", std::make_shared<zeno::DummyObject>());
        if (outputs.find("index") == outputs.end()) {
            set_output("index", std::make_shared<zeno::NumericObject>());
        }
    }

    virtual void update() override final {
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(m_index);
        set_output("index", std::move(ret));
        m_index++;
    }
};

ZENDEFNODE(BeginFor, {
    {{"int", "count"}},
    {{"int", "index"}, "FOR"},
    {},
    {"control"},
});


struct EndFor : zeno::ContextManagedNode {
    virtual void post_do_apply() {}

    virtual void preApply() override {
        auto [sn, ss] = safe_at(inputBounds, "FOR", "input socket of EndFor");
        auto fore = dynamic_cast<IBeginFor *>(graph->nodes.at(sn).get());
        if (!fore) {
            throw Exception("EndFor::FOR must be conn to BeginFor::FOR!\n");
        }
        graph->applyNode(sn);
        std::unique_ptr<zeno::Context> old_ctx = nullptr;
        while (fore->checkIsContinue()) {
            fore->doUpdate();
            push_context();
            INode::preApply();
            post_do_apply();
            old_ctx = pop_context();
        }
        if (old_ctx) {
            // auto-valid the nodes in last iteration when refered from outside
            graph->ctx->mergeVisited(*old_ctx);
            old_ctx = nullptr;
        }
    }

    virtual void apply() override {}
};

ZENDEFNODE(EndFor, {
    {"FOR"},
    {},
    {},
    {"control"},
});


struct BreakFor : zeno::INode {
    virtual void apply() override {
        auto [sn, ss] = safe_at(inputBounds, "FOR", "input socket of BreakFor");
        auto fore = dynamic_cast<IBeginFor *>(graph->nodes.at(sn).get());
        if (!fore) {
            throw Exception("BreakFor::FOR must be conn to BeginFor::FOR!\n");
        }
        if (!has_input("breaks") || get_input2<bool>("breaks")) {
            fore->breakThisFor();  // will still keep going the rest of loop body? yes
        }
    }

    //virtual void apply() override {}
};

ZENDEFNODE(BreakFor, {
    {"FOR", {"bool", "breaks", "1"}},
    {},
    {},
    {"control"},
});

struct BeginForEach : IBeginFor {
    int m_index = 0;
    std::shared_ptr<zeno::ListObject> m_list;
    zany m_accumate;

    virtual bool isContinue() const override final {
        return m_index < m_list->arr.size();
    }

    virtual void execute() override final {
        m_index = 0;
        m_list = get_input<zeno::ListObject>("list");
        if (has_input("accumate"))
            m_accumate = get_input("accumate");
        set_output("FOR", std::make_shared<zeno::DummyObject>());
    }

    virtual void update() override final {
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(m_index);
        set_output("index", std::move(ret));
        auto obj = m_list->arr[m_index];
        set_output("object", std::move(obj));
        m_index++;
        if (m_accumate)
            set_output("accumate", std::move(m_accumate));
    }
};

ZENDEFNODE(BeginForEach, {
    {"list", "accumate"},
    {"object", "accumate", {"int", "index"}, "FOR"},
    {},
    {"control"},
});

struct EndForEach : EndFor {
    std::vector<zany> result;
    std::vector<zany> dropped_result;

    virtual void post_do_apply() override {
        bool accept = true;
        if (requireInput("accept")) {
            accept = evaluate_condition(get_input("accept").get());
        }
        if (requireInput("object")) {
            auto obj = get_input("object");
            if (accept)
                result.push_back(std::move(obj));
            else
                dropped_result.push_back(std::move(obj));
        }
        if (requireInput("list")) {
            if (accept) {
                auto listObj = get_input<zeno::ListObject>("list");
                for(auto obj : listObj->arr)
                    result.push_back(std::move(obj));
            }
            else{
                auto listObj = get_input<zeno::ListObject>("list");
                for(auto obj : listObj->arr)
                    dropped_result.push_back(std::move(obj));
            }
        }
        if (requireInput("accumate")) {
            auto [sn, ss] = safe_at(inputBounds, "FOR", "input socket of EndForEach");
            auto fore = dynamic_cast<BeginForEach *>(graph->nodes.at(sn).get());
            if (!fore) {
                throw Exception("EndForEach::FOR must be conn to BeginForEach::FOR (when accumate used)!\n");
            }
            auto accumate = get_input("accumate");
            fore->m_accumate = std::move(accumate);
        }
    }

    virtual void preApply() override {
        EndFor::preApply();
        if (get_param<bool>("doConcat")) {
            decltype(result) newres;
            for (auto &xs: result) {
                for (auto &x: safe_dynamic_cast<ListObject>(xs, "do concat ")->arr)
                    newres.push_back(std::move(x));
            }
            result = std::move(newres);
            decltype(dropped_result) dropped_newres;
            for (auto &xs: dropped_result) {
                for (auto &x: safe_dynamic_cast<ListObject>(xs, "do concat ")->arr)
                    dropped_newres.push_back(std::move(x));
            }
            dropped_result = std::move(dropped_newres);
        }
        auto list = std::make_shared<ListObject>();
        list->arr = std::move(result);
        set_output("list", std::move(list));
        auto dropped_list = std::make_shared<ListObject>();
        dropped_list->arr = std::move(dropped_result);
        set_output("droppedList", std::move(dropped_list));

        auto [sn, ss] = safe_at(inputBounds, "FOR", "input socket of EndForEach");
        if (auto fore = dynamic_cast<BeginForEach *>(graph->nodes.at(sn).get()); fore) {
            if (fore->m_accumate)
                set_output("accumate", std::move(fore->m_accumate));
        }
    }
};

ZENDEFNODE(EndForEach, {
    {"object", "list", "accumate", {"bool", "accept", "1"}, "FOR"},
    {"list", "droppedList", "accumate"},
    {{"bool", "doConcat", "0"}},
    {"control"},
});


struct BeginSubstep : IBeginFor {
    float m_total = 0;
    float m_mindt = 0;
    float m_elapsed = 0;
    bool m_ever_called = false;

    virtual bool isContinue() const override final {
        return m_elapsed < m_total;
    }

    virtual void execute() override final {
        m_elapsed = 0;
        m_ever_called = false;
        m_total = get_input<zeno::NumericObject>("total_dt")->get<float>();
        auto min_scale = has_input("min_scale") ?
            get_input<zeno::NumericObject>("min_scale")->get<float>() : 0.05f;
        m_mindt = m_total * min_scale;
        set_output("FOR", std::make_shared<zeno::DummyObject>());
    }

    virtual void update() override final {
        auto ret = std::make_shared<zeno::NumericObject>();
        ret->set(m_elapsed);
        set_output("elapsed_time", std::move(ret));
    }
};

ZENDEFNODE(BeginSubstep, {
    {{"float", "total_dt"}, {"float", "min_scale", "0.05"}},
    {"FOR", {"float", "elapsed_time"}},
    {},
    {"control"},
});

struct SubstepDt : zeno::INode {
    void apply() override {
        auto [sn, ss] = safe_at(inputBounds, "FOR", "input socket of SubstepDt");
        auto fore = dynamic_cast<BeginSubstep *>(graph->nodes.at(sn).get());
        if (!fore) {
            throw Exception("SubstepDt::FOR must be conn to BeginSubstep::FOR!\n");
        }
        fore->m_ever_called = true;
        float dt = get_input<zeno::NumericObject>("desired_dt")->get<float>();
        if (fore->m_elapsed + dt >= fore->m_total) {
            dt = std::max(0.f, fore->m_total - fore->m_elapsed);
            fore->m_elapsed = fore->m_total;
        } else {
            if (dt < fore->m_mindt) {
                dt = fore->m_mindt;
            }
            fore->m_elapsed += dt;
        }
        float portion = fore->m_total ? fore->m_elapsed / fore->m_total : 0.0f;
        printf("** actual_dt: %f\n", dt);
        printf("** portion: %f\n", portion);
        auto ret_dt = std::make_shared<zeno::NumericObject>();
        ret_dt->set(dt);
        set_output("actual_dt", std::move(ret_dt));
        auto ret_portion = std::make_shared<zeno::NumericObject>();
        ret_portion->set(portion);
        set_output("portion", std::move(ret_portion));
    }
};

ZENDEFNODE(SubstepDt, {
    {"FOR", {"float", "desired_dt", "0.04"}},
    {{"float", "actual_dt"}, {"float", "portion"}},
    {},
    {"control"},
});




struct IfElse : zeno::INode {
    virtual void preApply() override {
        requireInput("cond");
        auto cond = get_input("cond");
        /*if (has_option("MUTE")) {
            requireInput("true");
        } else*/ if (evaluate_condition(cond.get())) {
            if (requireInput("true")) {
                set_output("result", get_input("true"));
            } else {
                set_output("result", std::make_shared<DummyObject>());
            }
        } else {
            if (requireInput("false")) {
                set_output("result", get_input("false"));
            } else {
                set_output("result", std::make_shared<DummyObject>());
            }
        }

        apply();
    }

    virtual void apply() override {}
};

ZENDEFNODE(IfElse, {
    {"true", "false", {"bool", "cond"}},
    {"result"},
    {},
    {"control"},
});


/*** Start Of - ZHXX Control Flow ***

struct IF : zeno::INode {

    bool m_condition;
    virtual void apply() override {
        m_condition = (get_input<zeno::NumericObject>("condition")->get<int>());
        set_output("Then", std::make_shared<zeno::DummyObject>());
    }
    bool getCondition() const
    {
        return m_condition==1;
    }

};

ZENDEFNODE(IF, {
    {"condition"},
    {"Then"},
    {},
    {"control"},
});

struct EndIF : zeno::ContextManagedNode {
    virtual void preApply() override {
        auto [sn, ss] = inputBounds.at("IF");
        auto true_exp = dynamic_cast<IF *>(graph->nodes.at(sn).get());
        if (!true_exp) {
            throw Exception("please connect true and false execution tree\n");
        }
        graph->applyNode(sn);
        if(true_exp->getCondition()){
            push_context();
            zeno::INode::doApply();
            pop_context();
        }
    }

    virtual void apply() override {}
};

ZENDEFNODE(EndIF, {
    {"IF"},
    {},
    {},
    {"control"},
});

struct IBranch : zeno::INode {
    virtual bool getCondition() const = 0;
};

struct TrueBranch : IBranch {
    bool m_execute;
    virtual void apply() override {
        auto innum = (get_input<zeno::NumericObject>("condition")->get<int>());
        m_execute = (innum == 1);
        set_output("Branch", std::make_shared<zeno::DummyObject>());
    }
    virtual bool getCondition() const override
    {
        return m_execute;
    }

};

ZENDEFNODE(TrueBranch, {
    {"condition"},
    {"Branch"},
    {},
    {"control"},
});
struct FalseBranch : IBranch {
    bool m_execute;
    virtual void apply() override {
        auto innum = (get_input<zeno::NumericObject>("condition")->get<int>());
        m_execute = (innum == 1);
        set_output("Branch", std::make_shared<zeno::DummyObject>());
    }
    virtual bool getCondition() const override
    {
        return !m_execute;
    }

};

ZENDEFNODE(FalseBranch, {
    {"condition"},
    {"Branch"},
    {},
    {"control"},
});

struct EndBranch : zeno::ContextManagedNode {
    virtual void preApply() override {
        auto [sn, ss] = inputBounds.at("BranchIn");
        auto exec = dynamic_cast<IBranch *>(graph->nodes.at(sn).get());
        graph->applyNode(sn);
        if(exec->getCondition()){
            push_context();
            zeno::INode::doApply();
            pop_context();
        }
    }

    virtual void apply() override {
        set_output("Branch", std::make_shared<zeno::DummyObject>());
    }
};

ZENDEFNODE(EndBranch, {
    {"BranchIn"},
    {"Branch"},
    {},
    {"control"},
});


struct ConditionedDo : zeno::INode {
    bool m_which;
    virtual void preApply() override {
        
        auto [sn, ss] = inputBounds.at("True");
        auto [sn1, ss1] = inputBounds.at("False");
        
        auto exec = dynamic_cast<EndBranch *>(graph->nodes.at(sn).get());
        exec->doApply();

        auto exec1 = dynamic_cast<EndBranch *>(graph->nodes.at(sn1).get());
        exec1->doApply();

    }

    virtual void apply() override {
    }
};

ZENDEFNODE(ConditionedDo, {
    {"True", "False"},
    {},
    {},
    {"control"},
});

*** End Of - ZHXX Control Flow ***/




}
