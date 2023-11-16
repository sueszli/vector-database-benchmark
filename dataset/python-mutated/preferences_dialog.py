from hscommon.trans import trget
from hscommon.plat import ISLINUX
from core.scanner import ScanType
from core.app import AppMode
from qt.preferences_dialog import PreferencesDialogBase
tr = trget('ui')

class PreferencesDialog(PreferencesDialogBase):

    def _setupPreferenceWidgets(self):
        if False:
            while True:
                i = 10
        self._setupFilterHardnessBox()
        self.widgetsVLayout.addLayout(self.filterHardnessHLayout)
        self._setupAddCheckbox('matchScaledBox', tr('Match pictures of different dimensions'))
        self.widgetsVLayout.addWidget(self.matchScaledBox)
        self._setupAddCheckbox('mixFileKindBox', tr('Can mix file kind'))
        self.widgetsVLayout.addWidget(self.mixFileKindBox)
        self._setupAddCheckbox('useRegexpBox', tr('Use regular expressions when filtering'))
        self.widgetsVLayout.addWidget(self.useRegexpBox)
        self._setupAddCheckbox('removeEmptyFoldersBox', tr('Remove empty folders on delete or move'))
        self.widgetsVLayout.addWidget(self.removeEmptyFoldersBox)
        self._setupAddCheckbox('ignoreHardlinkMatches', tr('Ignore duplicates hardlinking to the same file'))
        self.widgetsVLayout.addWidget(self.ignoreHardlinkMatches)
        self._setupBottomPart()

    def _setupDisplayPage(self):
        if False:
            while True:
                i = 10
        super()._setupDisplayPage()
        self._setupAddCheckbox('details_dialog_override_theme_icons', tr('Override theme icons in viewer toolbar'))
        self.details_dialog_override_theme_icons.setToolTip(tr('Use our own internal icons instead of those provided by the theme engine'))
        self.details_dialog_override_theme_icons.setEnabled(False if not ISLINUX else True)
        index = self.details_groupbox_layout.indexOf(self.details_dialog_vertical_titlebar)
        self.details_groupbox_layout.insertWidget(index + 1, self.details_dialog_override_theme_icons)
        self._setupAddCheckbox('details_dialog_viewers_show_scrollbars', tr('Show scrollbars in image viewers'))
        self.details_dialog_viewers_show_scrollbars.setToolTip(tr("When the image displayed doesn't fit the viewport, show scrollbars to span the view around"))
        self.details_groupbox_layout.insertWidget(index + 2, self.details_dialog_viewers_show_scrollbars)

    def _load(self, prefs, setchecked, section):
        if False:
            i = 10
            return i + 15
        setchecked(self.matchScaledBox, prefs.match_scaled)
        scan_type = prefs.get_scan_type(AppMode.PICTURE)
        fuzzy_scan = scan_type == ScanType.FUZZYBLOCK
        self.filterHardnessSlider.setEnabled(fuzzy_scan)
        setchecked(self.details_dialog_override_theme_icons, prefs.details_dialog_override_theme_icons)
        setchecked(self.details_dialog_viewers_show_scrollbars, prefs.details_dialog_viewers_show_scrollbars)

    def _save(self, prefs, ischecked):
        if False:
            for i in range(10):
                print('nop')
        prefs.match_scaled = ischecked(self.matchScaledBox)
        prefs.details_dialog_override_theme_icons = ischecked(self.details_dialog_override_theme_icons)
        prefs.details_dialog_viewers_show_scrollbars = ischecked(self.details_dialog_viewers_show_scrollbars)