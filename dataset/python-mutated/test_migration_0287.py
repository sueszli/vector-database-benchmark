from posthog.test.base import NonAtomicTestMigrations

class CreatingSessionRecordingModelMigrationTestCase(NonAtomicTestMigrations):
    migrate_from = '0286_index_insightcachingstate_lookup'
    migrate_to = '0287_add_session_recording_model'
    CLASS_DATA_LEVEL_SETUP = False

    def setUpBeforeMigration(self, apps):
        if False:
            return 10
        Organization = apps.get_model('posthog', 'Organization')
        Team = apps.get_model('posthog', 'Team')
        SessionRecordingPlaylist = apps.get_model('posthog', 'SessionRecordingPlaylist')
        SessionRecordingPlaylistItem = apps.get_model('posthog', 'SessionRecordingPlaylistItem')
        org = Organization.objects.create(name='o1')
        team = Team.objects.create(name='t1', organization=org)
        playlist1 = SessionRecordingPlaylist.objects.create(name='p1', team=team)
        SessionRecordingPlaylistItem.objects.create(session_id='s_0', playlist=playlist1, deleted=True)
        playlist2 = SessionRecordingPlaylist.objects.create(name='p2', team=team)
        playlist3 = SessionRecordingPlaylist.objects.create(name='p3', team=team)
        SessionRecordingPlaylistItem.objects.create(session_id='s_1', playlist=playlist2)
        SessionRecordingPlaylistItem.objects.create(session_id='s_1', playlist=playlist3)
        playlist4 = SessionRecordingPlaylist.objects.create(name='p4', team=team)
        playlist5 = SessionRecordingPlaylist.objects.create(name='p5', team=team)
        SessionRecordingPlaylistItem.objects.create(session_id='s_2_1', playlist=playlist4)
        SessionRecordingPlaylistItem.objects.create(session_id='s_2_2', playlist=playlist5)
        playlist6 = SessionRecordingPlaylist.objects.create(name='p6', team=team)
        SessionRecordingPlaylistItem.objects.create(session_id='s_3_1', playlist=playlist6)
        SessionRecordingPlaylistItem.objects.create(session_id='s_3_2', playlist=playlist6)

    def test_migrate_to_create_session_recordings(self):
        if False:
            for i in range(10):
                print('nop')
        SessionRecording = self.apps.get_model('posthog', 'SessionRecording')
        SessionRecordingPlaylist = self.apps.get_model('posthog', 'SessionRecordingPlaylist')
        SessionRecordingPlaylistItem = self.apps.get_model('posthog', 'SessionRecordingPlaylistItem')
        playlist = SessionRecordingPlaylist.objects.get(name='p1')
        assert SessionRecordingPlaylistItem.objects.filter(playlist=playlist, deleted=False).count() == 0
        assert SessionRecording.objects.filter(session_id='s_0').count() == 0
        assert SessionRecording.objects.filter(session_id='s_1').count() == 1
        playlist_items = SessionRecordingPlaylistItem.objects.filter(session_id='s_1').select_related('playlist').all()
        assert len(playlist_items) == 2
        assert playlist_items[0].playlist.name == 'p2'
        assert playlist_items[1].playlist.name == 'p3'
        assert SessionRecording.objects.filter(session_id='s_2_1').count() == 1
        assert SessionRecording.objects.filter(session_id='s_2_2').count() == 1
        playlist_items = SessionRecordingPlaylistItem.objects.filter(session_id='s_2_1').select_related('playlist').all()
        assert len(playlist_items) == 1
        assert playlist_items[0].playlist.name == 'p4'
        playlist_items = SessionRecordingPlaylistItem.objects.filter(session_id='s_2_2').select_related('playlist').all()
        assert len(playlist_items) == 1
        assert playlist_items[0].playlist.name == 'p5'
        assert SessionRecording.objects.filter(session_id='s_3_1').count() == 1
        assert SessionRecording.objects.filter(session_id='s_3_2').count() == 1
        playlist_items = SessionRecordingPlaylistItem.objects.filter(playlist__name='p6').all()
        assert len(playlist_items) == 2
        assert playlist_items[0].recording_id == 's_3_1'
        assert playlist_items[1].recording_id == 's_3_2'

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        Organization = self.apps.get_model('posthog', 'Organization')
        Team = self.apps.get_model('posthog', 'Team')
        SessionRecording = self.apps.get_model('posthog', 'SessionRecording')
        SessionRecordingPlaylistItem = self.apps.get_model('posthog', 'SessionRecordingPlaylistItem')
        SessionRecordingPlaylist = self.apps.get_model('posthog', 'SessionRecordingPlaylist')
        SessionRecording.objects.all().delete()
        SessionRecordingPlaylistItem.objects.all().delete()
        SessionRecordingPlaylist.objects.all().delete()
        Team.objects.all().delete()
        Organization.objects.all().delete()