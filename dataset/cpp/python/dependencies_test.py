import os

extra_plugin_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "dependent_plugins"
)


def test_if_all_loaded_by_default(testbot):
    plug_names = testbot.bot.plugin_manager.get_all_active_plugin_names()
    assert "Single" in plug_names
    assert "Parent1" in plug_names
    assert "Parent2" in plug_names


def test_single_dependency(testbot):
    pm = testbot.bot.plugin_manager
    for p in ("Single", "Parent1", "Parent2"):
        pm.deactivate_plugin(p)

    # everything should be gone
    plug_names = pm.get_all_active_plugin_names()
    assert "Single" not in plug_names
    assert "Parent1" not in plug_names
    assert "Parent2" not in plug_names

    pm.activate_plugin("Single")

    # it should have activated the dependent plugin 'Parent1' only
    plug_names = pm.get_all_active_plugin_names()
    assert "Single" in plug_names
    assert "Parent1" in plug_names
    assert "Parent2" not in plug_names


def test_double_dependency(testbot):
    pm = testbot.bot.plugin_manager
    all = ("Double", "Parent1", "Parent2")
    for p in all:
        pm.deactivate_plugin(p)

    pm.activate_plugin("Double")
    plug_names = pm.get_all_active_plugin_names()
    for p in all:
        assert p in plug_names


def test_dependency_retrieval(testbot):
    assert "youpi" in testbot.exec_command("!depfunc")


def test_direct_circular_dependency(testbot):
    plug_names = testbot.bot.plugin_manager.get_all_active_plugin_names()
    assert "Circular1" not in plug_names


def test_indirect_circular_dependency(testbot):
    plug_names = testbot.bot.plugin_manager.get_all_active_plugin_names()
    assert "Circular2" not in plug_names
    assert "Circular3" not in plug_names
    assert "Circular4" not in plug_names


def test_chained_dependency(testbot):
    plug_names = testbot.bot.plugin_manager.get_all_active_plugin_names()
    assert "Chained1" in plug_names
    assert "Chained2" in plug_names
    assert "Chained3" in plug_names
