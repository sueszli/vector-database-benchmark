from feeluown.app.gui_app import GuiApp, CollectionManager

def test_gui_app_initialize(qtbot, mocker, args, config, noharm):
    if False:
        while True:
            i = 10
    mocker.patch('feeluown.app.app.TaskManager')
    mocker.patch.object(CollectionManager, 'scan')
    app = GuiApp(args, config)
    qtbot.addWidget(app)
    mocker.patch.object(app.live_lyric.sentence_changed, 'connect')
    mocker.patch.object(app, 'about_to_exit')
    app.initialize()