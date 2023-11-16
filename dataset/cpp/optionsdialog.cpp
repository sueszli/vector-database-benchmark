/*
 * Copyright (C) 2009, 2010, 2011, 2013 Nicolas Bonnefon and other contributors
 *
 * This file is part of glogg.
 *
 * glogg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * glogg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with glogg.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Copyright (C) 2016 -- 2019 Anton Filimonov and other contributors
 *
 * This file is part of klogg.
 *
 * klogg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * klogg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with klogg.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <QColorDialog>
#include <QKeySequenceEdit>
#include <QMessageBox>
#include <QToolButton>
#include <QtGui>

#include "encodings.h"
#include "fontutils.h"
#include "highlighteredit.h"
#include "log.h"
#include "mainwindow.h"
#include "recentfiles.h"
#include "savedsearches.h"
#include "shortcuts.h"
#include "styles.h"

#include "optionsdialog.h"

static constexpr int PollIntervalMin = 10;
static constexpr int PollIntervalMax = 3600000;

// Constructor
OptionsDialog::OptionsDialog( QWidget* parent )
    : QDialog( parent )
{
    setupUi( this );

    setupTabs();
    setupFontList();
    setupRegexp();
    setupStyles();
    setupEncodings();
    setupLanguageList();

    // Validators
    QValidator* pollingIntervalValidator = new QIntValidator( PollIntervalMin, PollIntervalMax );
    pollIntervalLineEdit->setValidator( pollingIntervalValidator );

    connect( buttonBox, &QDialogButtonBox::clicked, this, &OptionsDialog::onButtonBoxClicked );
    connect( fontFamilyBox, &QComboBox::currentTextChanged, this, &OptionsDialog::updateFontSize );
    connect( pollingCheckBox, &QCheckBox::toggled, [ this ]( auto ) { this->setupPolling(); } );
    connect( searchResultsCacheCheckBox, &QCheckBox::toggled,
             [ this ]( auto ) { this->setupSearchResultsCache(); } );
    connect( loggingCheckBox, &QCheckBox::toggled, [ this ]( auto ) { this->setupLogging(); } );

    connect( extractArchivesCheckBox, &QCheckBox::toggled,
             [ this ]( auto ) { this->setupArchives(); } );

    connect( mainSearchColorButton, &QPushButton::clicked, this, &OptionsDialog::changeMainColor );
    connect( quickFindColorButton, &QPushButton::clicked, this, &OptionsDialog::changeQfColor );

    connect( restoreShortcutsDefaults, &QPushButton::clicked, this, [ this ]() {
        auto ret = QMessageBox::question(
            this, "Restore Default Shortcuts", "Do you want to restore default shortcuts?",
            QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel );
        if ( ret == QMessageBox::Yes )
            buildShortcutsTable( true );
    } );

    updateDialogFromConfig();

    setupPolling();
    setupSearchResultsCache();
    setupLogging();
    setupArchives();
}

//
// Private functions
//

// Setups the tabs depending on the configuration
void OptionsDialog::setupTabs()
{
#ifndef Q_OS_WIN
    keepFileClosedCheckBox->setVisible( false );
#endif

#ifdef Q_OS_MAC
    minimizeToTrayCheckBox->setVisible( false );
#endif

#ifndef KLOGG_HAS_HS
    regexpEngineLabel->setVisible( false );
    regexpEngineComboBox->setVisible( false );
#endif
}

// Populates the 'family' ComboBox
void OptionsDialog::setupFontList()
{
    const auto families = FontUtils::availableFonts();
    for ( const QString& str : qAsConst( families ) ) {
        fontFamilyBox->addItem( str );
    }
}

// Populate the regexp ComboBoxes
void OptionsDialog::setupRegexp()
{
    QStringList regexpTypes;
    regexpTypes << tr( "Extended Regexp" ) << tr( "Fixed Strings" );

    mainSearchBox->addItems( regexpTypes );
    quickFindSearchBox->addItems( regexpTypes );

    QStringList regexpEngines;
    regexpEngines << tr( "Hyperscan" ) << tr( "Qt" );

    regexpEngineComboBox->addItems( regexpEngines );
}

void OptionsDialog::setupStyles()
{
    styleComboBox->addItems( StyleManager::availableStyles() );
}

void OptionsDialog::setupEncodings()
{
    const auto availableEncodings = EncodingMenu::supportedEncodings();
    encodingComboBox->addItem( "Auto", -1 );

    std::map<QString, int> allMibs;

    for ( const auto& group : availableEncodings ) {
        for ( const auto& mib : group.second ) {
            auto codec = QTextCodec::codecForMib( mib );
            if ( codec ) {
                allMibs.emplace( codec->name(), mib );
            }
        }
    }

    for ( const auto& codec : allMibs ) {
        encodingComboBox->addItem( codec.first, codec.second );
    }
}

void OptionsDialog::setupLanguageList()
{
    QResource resource( ":/i18n/Languages.xml" );
    QByteArray bytes( reinterpret_cast<const char*>( resource.data() ), (int)resource.size() );
    QXmlStreamReader xml( bytes );

    while ( !xml.atEnd() ) {
        QXmlStreamReader::TokenType token = xml.readNext();
        if ( xml.hasError() ) {
            LOG_ERROR << "load language error";
            return;
        }

        if ( xml.name() == QString( "language" ) && token == QXmlStreamReader::StartElement ) {
            QXmlStreamAttributes attributes = xml.attributes();
            languageComboBox->addItem( attributes.value( "name" ).toString(),
                                       attributes.value( "ietfCode" ).toString() );
        }
    }
}

void OptionsDialog::setupPolling()
{
    pollIntervalLineEdit->setEnabled( pollingCheckBox->isChecked() );
}

void OptionsDialog::setupSearchResultsCache()
{
    searchCacheSpinBox->setEnabled( searchResultsCacheCheckBox->isChecked() );
}

void OptionsDialog::setupLogging()
{
    verbositySpinBox->setEnabled( loggingCheckBox->isChecked() );
}

void OptionsDialog::setupArchives()
{
    extractArchivesAlwaysCheckBox->setEnabled( extractArchivesCheckBox->isChecked() );
}

// Convert a regexp type to its index in the list
int OptionsDialog::getRegexpTypeIndex( SearchRegexpType syntax ) const
{
    int index;

    switch ( syntax ) {
    case SearchRegexpType::FixedString:
        index = 1;
        break;
    default:
        index = 0;
        break;
    }

    return index;
}

// Convert the index of a regexp type to its type
SearchRegexpType OptionsDialog::getRegexpTypeFromIndex( int index ) const
{
    SearchRegexpType type;

    switch ( index ) {
    case 1:
        type = SearchRegexpType::FixedString;
        break;
    default:
        type = SearchRegexpType::ExtendedRegexp;
        break;
    }

    return type;
}

int OptionsDialog::getRegexpEngineIndex( RegexpEngine engine ) const
{
    int index;

    switch ( engine ) {
    case RegexpEngine::QRegularExpression:
        index = 1;
        break;
    default:
        index = 0;
        break;
    }

    return index;
}

RegexpEngine OptionsDialog::getRegexpEngineFromIndex( int index ) const
{
    RegexpEngine type;

    switch ( index ) {
    case 1:
        type = RegexpEngine::QRegularExpression;
        break;
    default:
        type = RegexpEngine::Hyperscan;
        break;
    }

    return type;
}

// Updates the dialog box using values in global Config()
void OptionsDialog::updateDialogFromConfig()
{
    const auto& config = Configuration::get();

    // Main font
    QFontInfo fontInfo = QFontInfo( config.mainFont() );

    int familyIndex = fontFamilyBox->findText( fontInfo.family() );
    if ( familyIndex != -1 )
        fontFamilyBox->setCurrentIndex( familyIndex );

    updateFontSize( fontInfo.family() );

    int sizeIndex = fontSizeBox->findText( QString::number( fontInfo.pointSize() ) );
    if ( sizeIndex != -1 )
        fontSizeBox->setCurrentIndex( sizeIndex );

    fontSmoothCheckBox->setChecked( config.forceFontAntialiasing() );
    wrapTextCheckBox->setChecked( config.useTextWrap() );
    enableQtHiDpiCheckBox->setChecked( config.enableQtHighDpi() );
    scaleRoundingComboBox->setCurrentIndex( config.scaleFactorRounding() - 1 );

    // Language
    auto langIdx = languageComboBox->findData( { config.language() } );
    if ( langIdx == -1 ) {
        langIdx = 0;
    }
    languageComboBox->setCurrentIndex( langIdx );

    const auto style = config.style();
    if ( !styleComboBox->findText( style, Qt::MatchExactly ) ) {
        styleComboBox->setCurrentIndex( 0 );
    }
    else {
        styleComboBox->setCurrentText( style );
    }

    hideAnsiColorsCheckBox->setChecked( config.hideAnsiColorSequences() );

    // Regexp types
    mainSearchBox->setCurrentIndex( getRegexpTypeIndex( config.mainRegexpType() ) );
    mainSearchColor_ = config.mainSearchBackColor();
    HighlighterEdit::updateIcon( mainSearchColorButton, mainSearchColor_ );
    quickFindSearchBox->setCurrentIndex( getRegexpTypeIndex( config.quickfindRegexpType() ) );
    qfSearchColor_ = config.qfBackColor();
    HighlighterEdit::updateIcon( quickFindColorButton, qfSearchColor_ );
    regexpEngineComboBox->setCurrentIndex( getRegexpEngineIndex( config.regexpEngine() ) );
    autoRunSearchOnAddCheckBox->setChecked( config.autoRunSearchOnPatternChange() );

    highlightMainSearchCheckBox->setChecked( config.mainSearchHighlight() );
    variateHighlightCheckBox->setChecked( config.variateMainSearchHighlight() );
    incrementalCheckBox->setChecked( config.isQuickfindIncremental() );

    // Polling
    nativeFileWatchCheckBox->setChecked( config.nativeFileWatchEnabled() );
    fastModificationDetectionCheckBox->setChecked( config.fastModificationDetection() );
    pollingCheckBox->setChecked( config.pollingEnabled() );
    pollIntervalLineEdit->setText( QString::number( config.pollIntervalMs() ) );
    allowFollowOnScrollCheckBox->setChecked( config.allowFollowOnScroll() );

    // Last session
    loadLastSessionCheckBox->setChecked( config.loadLastSession() );
    followFileOnLoadCheckBox->setChecked( config.followFileOnLoad() );
    minimizeToTrayCheckBox->setChecked( config.minimizeToTray() );
    multipleWindowsCheckBox->setChecked( config.allowMultipleWindows() );

    loggingCheckBox->setChecked( config.enableLogging() );
    verbositySpinBox->setValue( config.loggingLevel() );

    extractArchivesCheckBox->setChecked( config.extractArchives() );
    extractArchivesAlwaysCheckBox->setChecked( config.extractArchivesAlways() );

    // Perf
    parallelSearchCheckBox->setChecked( config.useParallelSearch() );
    searchResultsCacheCheckBox->setChecked( config.useSearchResultsCache() );
    searchCacheSpinBox->setValue( static_cast<int>( config.searchResultsCacheLines() ) );
    indexReadBufferSpinBox->setValue( config.indexReadBufferSizeMb() );
    searchReadBufferSpinBox->setValue( config.searchReadBufferSizeLines() );
    keepFileClosedCheckBox->setChecked( config.keepFileClosed() );
    optimizeForNotLatinEncodingsCheckBox->setChecked( config.optimizeForNotLatinEncodings() );

    // version checking
    checkForNewVersionCheckBox->setChecked( config.versionCheckingEnabled() );

    // downloads
    verifySslCheckBox->setChecked( config.verifySslPeers() );

    const auto encodingIndex = encodingComboBox->findData( config.defaultEncodingMib() );
    encodingComboBox->setCurrentIndex( encodingIndex < 0 ? 0 : encodingIndex );

    buildShortcutsTable( false );

    const auto& savedSearches = SavedSearches::get();
    searchHistorySpinBox->setValue( savedSearches.historySize() );

    const auto& recentFiles = RecentFiles::get();
    filesHistoryMaxItemsSpinBox->setMinimum( 1 );
    filesHistoryMaxItemsSpinBox->setMaximum( MAX_RECENT_FILES );
    filesHistoryMaxItemsSpinBox->setValue( recentFiles.filesHistoryMaxItems() );
}

//
// Q_SLOTS:
//

void OptionsDialog::updateFontSize( const QString& fontFamily )
{
    QString oldFontSize = fontSizeBox->currentText();
    const auto sizes = FontUtils::availableFontSizes( fontFamily );

    fontSizeBox->clear();
    for ( int size : sizes ) {
        fontSizeBox->addItem( QString::number( size ) );
    }
    // Now restore the size we had before
    int i = fontSizeBox->findText( oldFontSize );
    if ( i != -1 )
        fontSizeBox->setCurrentIndex( i );
}

void OptionsDialog::changeMainColor()
{
    QColor newColor;
    if ( HighlighterEdit::showColorPicker( mainSearchColor_, newColor ) ) {
        mainSearchColor_ = newColor;
        HighlighterEdit::updateIcon( mainSearchColorButton, mainSearchColor_ );
    }
}

void OptionsDialog::changeQfColor()
{
    QColor newColor;
    if ( HighlighterEdit::showColorPicker( qfSearchColor_, newColor ) ) {
        qfSearchColor_ = newColor;
        HighlighterEdit::updateIcon( quickFindColorButton, qfSearchColor_ );
    }
}

void OptionsDialog::checkShortcutsOnDuplicate() const
{
    static constexpr int PRIMARY_COL = 1;
    static constexpr int SECONDARY_COL = 2;

    if ( !shortcutsTable->rowCount() ) {
        return;
    }

    const auto DEFAULT_BACKGROUND = shortcutsTable->item( 0, PRIMARY_COL )->background();

    for ( auto shortcutRow = 0; shortcutRow < shortcutsTable->rowCount(); ++shortcutRow ) {
        shortcutsTable->item( shortcutRow, PRIMARY_COL )->setBackground( DEFAULT_BACKGROUND );
        shortcutsTable->item( shortcutRow, SECONDARY_COL )->setBackground( DEFAULT_BACKGROUND );
    }

    std::unordered_map<std::string, std::pair<int, int>> uniqueShortcuts;
    bool hasDuplicateShortcuts = false;
    for ( auto shortcutRow = 0; shortcutRow < shortcutsTable->rowCount(); ++shortcutRow ) {

        auto hasDuplicates = [ &uniqueShortcuts, shortcutRow, this ]( int ncol ) {
            auto keySequence = static_cast<KeySequencePresenter*>(
                                   shortcutsTable->cellWidget( shortcutRow, ncol ) )
                                   ->keySequence();

            if ( !keySequence.isEmpty() ) {
                if ( auto it = uniqueShortcuts.find( keySequence.toStdString() );
                     it != uniqueShortcuts.end() ) {

                    shortcutsTable->item( it->second.first, it->second.second )
                        ->setBackground( Qt::red );
                    shortcutsTable->item( shortcutRow, ncol )->setBackground( Qt::red );

                    return true;
                }

                uniqueShortcuts.try_emplace( keySequence.toStdString(),
                                             std::make_pair( shortcutRow, ncol ) );
            }

            return false;
        };

        if ( hasDuplicates( PRIMARY_COL ) || hasDuplicates( SECONDARY_COL ) ) {
            hasDuplicateShortcuts = true;
        }
    }

    buttonBox->button( QDialogButtonBox::Ok )->setEnabled( !hasDuplicateShortcuts );
    buttonBox->button( QDialogButtonBox::Apply )->setEnabled( !hasDuplicateShortcuts );
}

int OptionsDialog::updateTranslate()
{
    auto mw = dynamic_cast<MainWindow*>( parent() );
    return mw->installLanguage( languageComboBox->currentData().toString() );
}

void OptionsDialog::updateConfigFromDialog()
{
    bool restartAppMessage = false;
    auto& config = Configuration::get();

    QFont font = QFont( fontFamilyBox->currentText(), ( fontSizeBox->currentText() ).toInt() );
    config.setMainFont( font );
    config.setForceFontAntialiasing( fontSmoothCheckBox->isChecked() );
    config.setUseTextWrap( wrapTextCheckBox->isChecked() );
    config.setEnableQtHighDpi( enableQtHiDpiCheckBox->isChecked() );
    config.setScaleFactorRounding( scaleRoundingComboBox->currentIndex() + 1 );

    config.setMainRegexpType( getRegexpTypeFromIndex( mainSearchBox->currentIndex() ) );
    config.setMainSearchBackColor( mainSearchColor_ );
    config.setEnableMainSearchHighlight( highlightMainSearchCheckBox->isChecked() );
    config.setVariateMainSearchHighlight( variateHighlightCheckBox->isChecked() );
    config.setQuickfindRegexpType( getRegexpTypeFromIndex( quickFindSearchBox->currentIndex() ) );
    config.setQfBackColor( qfSearchColor_ );
    config.setQuickfindIncremental( incrementalCheckBox->isChecked() );
    config.setRegexpEnging( getRegexpEngineFromIndex( regexpEngineComboBox->currentIndex() ) );
    config.setAutoRunSearchOnPatternChange( autoRunSearchOnAddCheckBox->isChecked() );

    config.setNativeFileWatchEnabled( nativeFileWatchCheckBox->isChecked() );
    config.setPollingEnabled( pollingCheckBox->isChecked() );
    auto pollInterval = pollIntervalLineEdit->text().toInt();
    if ( pollInterval < PollIntervalMin )
        pollInterval = PollIntervalMin;
    else if ( pollInterval > PollIntervalMax )
        pollInterval = PollIntervalMax;

    config.setPollIntervalMs( pollInterval );
    config.setFastModificationDetection( fastModificationDetectionCheckBox->isChecked() );
    config.setAllowFollowOnScroll( allowFollowOnScrollCheckBox->isChecked() );

    config.setLoadLastSession( loadLastSessionCheckBox->isChecked() );
    config.setFollowFileOnLoad( followFileOnLoadCheckBox->isChecked() );
    config.setAllowMultipleWindows( multipleWindowsCheckBox->isChecked() );
    config.setMinimizeToTray( minimizeToTrayCheckBox->isChecked() );
    config.setEnableLogging( loggingCheckBox->isChecked() );
    config.setLoggingLevel( verbositySpinBox->value() );

    config.setExtractArchives( extractArchivesCheckBox->isChecked() );
    config.setExtractArchivesAlways( extractArchivesAlwaysCheckBox->isChecked() );

    config.setUseParallelSearch( parallelSearchCheckBox->isChecked() );
    config.setUseSearchResultsCache( searchResultsCacheCheckBox->isChecked() );
    config.setSearchResultsCacheLines( static_cast<unsigned>( searchCacheSpinBox->value() ) );
    config.setIndexReadBufferSizeMb( indexReadBufferSpinBox->value() );
    config.setSearchReadBufferSizeLines( searchReadBufferSpinBox->value() );
    config.setKeepFileClosed( keepFileClosedCheckBox->isChecked() );
    config.setOptimizeForNotLatinEncodings( optimizeForNotLatinEncodingsCheckBox->isChecked() );

    // version checking
    config.setVersionCheckingEnabled( checkForNewVersionCheckBox->isChecked() );

    config.setVerifySslPeers( verifySslCheckBox->isChecked() );

    restartAppMessage = config.style() != styleComboBox->currentText();

    config.setStyle( styleComboBox->currentText() );
    config.setHideAnsiColorSequences( hideAnsiColorsCheckBox->isChecked() );

    config.setDefaultEncodingMib( encodingComboBox->currentData().toInt() );

    auto shortcuts = config.shortcuts();
    for ( auto shortcutRow = 0; shortcutRow < shortcutsTable->rowCount(); ++shortcutRow ) {
        QStringList actionKeys;

        auto primaryKeySequence
            = static_cast<KeySequencePresenter*>( shortcutsTable->cellWidget( shortcutRow, 1 ) )
                  ->keySequence();
        auto secondaryKeySequence
            = static_cast<KeySequencePresenter*>( shortcutsTable->cellWidget( shortcutRow, 2 ) )
                  ->keySequence();
        actionKeys << primaryKeySequence << secondaryKeySequence;

        auto action
            = shortcutsTable->item( shortcutRow, 0 )->data( Qt::UserRole ).toString().toStdString();
        shortcuts[ action ] = actionKeys;
    }
    config.setShortcuts( shortcuts );

    // update translate when accept or apply clicked
    restartAppMessage |= config.language() != languageComboBox->currentData().toString();
    updateTranslate();
    config.setLanguage( languageComboBox->currentData().toString() );
    retranslateUi( this );

    config.save();

    auto& savedSearches = SavedSearches::get();
    savedSearches.setHistorySize( searchHistorySpinBox->value() );
    savedSearches.save();

    auto& recentFiles = RecentFiles::get();
    recentFiles.setFilesHistoryMaxItems( filesHistoryMaxItemsSpinBox->value() );
    recentFiles.save();

    if ( restartAppMessage ) {
        QMessageBox::warning(
            this, "klogg",
            QApplication::translate( "OptionsDialog",
                                     "Klogg needs to be restarted to apply some changes. " ) );
    }

    Q_EMIT optionsChanged();
}

void OptionsDialog::onButtonBoxClicked( QAbstractButton* button )
{
    QDialogButtonBox::ButtonRole role = buttonBox->buttonRole( button );
    if ( ( role == QDialogButtonBox::AcceptRole ) || ( role == QDialogButtonBox::ApplyRole ) ) {
        updateConfigFromDialog();
    }

    if ( role == QDialogButtonBox::AcceptRole )
        accept();
    else if ( role == QDialogButtonBox::RejectRole )
        reject();
}

KeySequencePresenter::KeySequencePresenter( const QString& keySequence )
{
    keySequenceLabel_
        = new QLabel( QKeySequence( keySequence ).toString( QKeySequence::NativeText ) );

    auto editButton = new QPushButton();
    editButton->setText( "..." );
    editButton->setFixedWidth( 50 );

    auto layout = new QHBoxLayout();

    connect( editButton, &QPushButton::clicked, this, &KeySequencePresenter::showEditor );
    layout->addWidget( keySequenceLabel_ );
    layout->addStretch();
    layout->addWidget( editButton );
    layout->setContentsMargins( 4, 4, 4, 4 );

    this->setLayout( layout );
}

QString KeySequencePresenter::keySequence() const
{
    return keySequenceLabel_->text();
}

void KeySequencePresenter::showEditor()
{
    QDialog keyEditDialog;

    auto label = new QLabel( "Press new key combination" );
    auto editor = new QKeySequenceEdit( QKeySequence( keySequenceLabel_->text() ) );
    auto clearButton = new QToolButton();
    clearButton->setText( "Clear" );
    auto dialogButtons = new QDialogButtonBox( QDialogButtonBox::Ok | QDialogButtonBox::Cancel );

    auto layout = new QVBoxLayout();
    layout->addWidget( label );
    auto editorLayout = new QHBoxLayout();
    editorLayout->addWidget( editor );
    editorLayout->addWidget( clearButton );
    layout->addLayout( editorLayout );
    layout->addWidget( dialogButtons );
    keyEditDialog.setLayout( layout );

    connect( clearButton, &QToolButton::clicked, editor, &QKeySequenceEdit::clear );
    connect( dialogButtons, &QDialogButtonBox::accepted, &keyEditDialog, &QDialog::accept );
    connect( dialogButtons, &QDialogButtonBox::rejected, &keyEditDialog, &QDialog::reject );

    if ( keyEditDialog.exec() == QDialog::Accepted ) {
        keySequenceLabel_->setText( editor->keySequence().toString() );
        Q_EMIT edited(); // NOTE: it's important to emit this signal only after changing
                         // \keySequenceLabel_'s text
    }
}

void OptionsDialog::buildShortcutsTable( bool useDefaultsOnly )
{
    shortcutsTable->setRowCount( 0 );

    const auto& config = Configuration::get();
    auto shortcuts = ShortcutAction::defaultShortcuts();
    if ( !useDefaultsOnly ) {
        for ( const auto& [ action, keys ] : config.shortcuts() ) {
            shortcuts[ action ] = keys;
        }
    }

    for ( const auto& [ action, keys ] : shortcuts ) {
        auto currentRow = shortcutsTable->rowCount();
        shortcutsTable->insertRow( currentRow );

        auto keyItem = new QTableWidgetItem( ShortcutAction::actionName( action ) );
        keyItem->setFlags( Qt::ItemIsEnabled | Qt::ItemIsSelectable );
        keyItem->setData( Qt::UserRole, QString::fromStdString( action ) );
        shortcutsTable->setItem( currentRow, 0, keyItem );

        auto primaryKeySequence = new KeySequencePresenter( keys.size() > 0 ? keys[ 0 ] : "" );
        shortcutsTable->setItem( currentRow, 1, new QTableWidgetItem );
        shortcutsTable->setCellWidget( currentRow, 1, primaryKeySequence );
        connect( primaryKeySequence, &KeySequencePresenter::edited, this,
                 &OptionsDialog::checkShortcutsOnDuplicate );

        auto secondaryKeySequence = new KeySequencePresenter( keys.size() > 1 ? keys[ 1 ] : "" );
        shortcutsTable->setItem( currentRow, 2, new QTableWidgetItem );
        shortcutsTable->setCellWidget( currentRow, 2, secondaryKeySequence );
        connect( secondaryKeySequence, &KeySequencePresenter::edited, this,
                 &OptionsDialog::checkShortcutsOnDuplicate );
    }

    shortcutsTable->horizontalHeader()->setSectionResizeMode( QHeaderView::Stretch );
    shortcutsTable->horizontalHeader()->setSectionResizeMode( 0, QHeaderView::Interactive );
    shortcutsTable->horizontalHeader()->setMinimumSectionSize( 150 );
    shortcutsTable->resizeColumnToContents( 0 );
    shortcutsTable->setHorizontalHeaderItem( 0, new QTableWidgetItem( tr( "Action" ) ) );
    shortcutsTable->setHorizontalHeaderItem( 1, new QTableWidgetItem( tr( "Primary shortcut" ) ) );
    shortcutsTable->setHorizontalHeaderItem( 2,
                                             new QTableWidgetItem( tr( "Secondary shortcut" ) ) );

    // in case if user set duplicate keys and after restores defaults
    // it is need to enable back standard buttons
    checkShortcutsOnDuplicate();

    shortcutsTable->sortItems( 0 );
}
