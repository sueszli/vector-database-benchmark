#include "ConfigWidget.h"
#include "ui_ConfigWidget.h"

#include <CheckBox.h>
#include <CredentialsDlg.h>
#include <FileEditor.h>
#include <GitBase.h>
#include <GitConfig.h>
#include <GitCredentials.h>
#include <GitQlientSettings.h>
#include <NewVersionInfoDlg.h>
#include <PluginsDownloader.h>
#include <QLogger.h>

#include <QDir>
#include <QDirIterator>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QStandardPaths>
#include <QTimer>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#   include <qtermwidget_interface.h>
#endif

using namespace QLogger;

namespace
{
qint64 dirSize(QString dirPath)
{
   qint64 size = 0;
   QDir dir(dirPath);

   auto entryList = dir.entryList(QDir::Files | QDir::System | QDir::Hidden);

   for (const auto &filePath : std::as_const(entryList))
   {
      QFileInfo fi(dir, filePath);
      size += fi.size();
   }

   entryList = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::System | QDir::Hidden);

   for (const auto &childDirPath : std::as_const(entryList))
      size += dirSize(dirPath + QDir::separator() + childDirPath);

   return size;
}
}

ConfigWidget::ConfigWidget(const QSharedPointer<GitBase> &git, QWidget *parent)
   : QWidget(parent)
   , ui(new Ui::ConfigWidget)
   , mGit(git)
   , mFeedbackTimer(new QTimer())
   , mSave(new QPushButton(this))
   , mPluginsDownloader(new PluginsDownloader(this))
   , mDownloadButtons(new QButtonGroup(this))
{
   ui->setupUi(this);

   ui->lePluginsDestination->setText(QSettings().value("PluginsFolder", QString()).toString());
   ui->lPluginsWarning->setVisible(ui->lePluginsDestination->text().isEmpty());

   ui->lTerminalColorScheme->setVisible(false);
   ui->cbTerminalColorScheme->setVisible(false);

   mFeedbackTimer->setInterval(3000);

   mSave->setIcon(QIcon(":/icons/save"));
   mSave->setToolTip(tr("Save"));
   connect(mSave, &QPushButton::clicked, this, &ConfigWidget::saveFile);
   ui->tabWidget->setCornerWidget(mSave);

   ui->mainLayout->setColumnStretch(0, 1);
   ui->mainLayout->setColumnStretch(1, 3);

   const auto localGitLayout = new QVBoxLayout(ui->localGit);
   localGitLayout->setContentsMargins(QMargins());

   mLocalGit = new FileEditor(false, this);
   mLocalGit->editFile(mGit->getGitDir().append("/config"));
   localGitLayout->addWidget(mLocalGit);

   const auto globalGitLayout = new QVBoxLayout(ui->globalGit);
   globalGitLayout->setContentsMargins(QMargins());

   mGlobalGit = new FileEditor(false, this);
   mGlobalGit->editFile(QString("%1/%2").arg(QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
                                             QString::fromUtf8(".gitconfig")));
   globalGitLayout->addWidget(mGlobalGit);

   GitQlientSettings settings(mGit->getGitDir());

   const auto logsFolder = settings.globalValue("logsFolder").toString();
   if (logsFolder.isEmpty())
      settings.setGlobalValue("logsFolder", QString(QDir::currentPath()).append("/logs/"));

   ui->leLogsLocation->setText(settings.globalValue("logsFolder").toString());

   ui->chDevMode->setChecked(settings.localValue("DevMode", false).toBool());
   enableWidgets();

   // GitQlient configuration
   ui->chDisableLogs->setChecked(settings.globalValue("logsDisabled", true).toBool());
   ui->cbLogLevel->setCurrentIndex(settings.globalValue("logsLevel", static_cast<int>(LogLevel::Warning)).toInt());
   ui->spCommitTitleLength->setValue(settings.globalValue("commitTitleMaxLength", 50).toInt());
   ui->sbEditorFontSize->setValue(settings.globalValue("FileDiffView/FontSize", 8).toInt());
   ui->chSingleClickDiffView->setChecked(settings.globalValue("singleClickDiffView", false).toBool());

#ifdef Q_OS_LINUX
   ui->leEditor->setText(settings.globalValue("ExternalEditor", QString()).toString());
   ui->leExtFileExplorer->setText(settings.globalValue("FileExplorer", "xdg-open").toString());
#else
   ui->leExtFileExplorer->setHidden(true);
   ui->labelExtFileExplorer->setHidden(true);
#endif

   const auto originalStyles = settings.globalValue("colorSchema", 0).toInt();

   ui->cbStyle->setCurrentIndex(originalStyles);
   connect(ui->cbStyle, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
           [this, originalStyles](int newIndex) {
              mShowResetMsg = newIndex != originalStyles;
              saveConfig();
           });

   // Repository configuration
   mOriginalRepoOrder = settings.localValue("GraphSortingOrder", 0).toInt();
   ui->cbLogOrder->setCurrentIndex(mOriginalRepoOrder);
   ui->autoFetch->setValue(settings.localValue("AutoFetch", 5).toInt());
   ui->pruneOnFetch->setChecked(settings.localValue("PruneOnFetch", true).toBool());
   ui->clangFormat->setChecked(settings.localValue("ClangFormatOnCommit", false).toBool());
   ui->updateOnPull->setChecked(settings.localValue("UpdateOnPull", false).toBool());
   ui->sbMaxCommits->setValue(settings.localValue("MaxCommits", 0).toInt());

   ui->tabWidget->setCurrentIndex(0);
   connect(ui->pbClearLogs, &ButtonLink::clicked, this, &ConfigWidget::clearLogs);
   connect(ui->pbClearCache, &ButtonLink::clicked, this, &ConfigWidget::clearCache);

   ui->cbPomodoroEnabled->setChecked(settings.localValue("Pomodoro/Enabled", true).toBool());

   ui->cbLocal->setChecked(settings.localValue("LocalHeader", true).toBool());
   ui->cbRemote->setChecked(settings.localValue("RemoteHeader", true).toBool());
   ui->cbTags->setChecked(settings.localValue("TagsHeader", true).toBool());
   ui->cbStash->setChecked(settings.localValue("StashesHeader", true).toBool());
   ui->cbSubmodule->setChecked(settings.localValue("SubmodulesHeader", true).toBool());
   ui->cbSubtree->setChecked(settings.localValue("SubtreeHeader", true).toBool());
   ui->cbDeleteFolder->setChecked(settings.localValue("DeleteRemoteFolder", false).toBool());

   QScopedPointer<GitConfig> gitConfig(new GitConfig(mGit));

   const auto url = gitConfig->getServerUrl();
   ui->credentialsFrames->setVisible(url.startsWith("https"));

   const auto mergeStrategyFF = gitConfig->getGitValue("pull.ff").output;
   const auto mergeStrategyRebase = gitConfig->getGitValue("pull.rebase").output;

   if (mergeStrategyFF.isEmpty())
   {
      if (mergeStrategyRebase.isEmpty() || mergeStrategyRebase.contains("false", Qt::CaseInsensitive))
         ui->cbPullStrategy->setCurrentIndex(0);
      else if (mergeStrategyRebase.contains("true", Qt::CaseInsensitive))
         ui->cbPullStrategy->setCurrentIndex(1);
   }
   else if (mergeStrategyFF.contains("true", Qt::CaseInsensitive))
      ui->cbPullStrategy->setCurrentIndex(2);

   fillLanguageBox();

   connect(ui->cbPullStrategy, SIGNAL(currentIndexChanged(int)), this, SLOT(onPullStrategyChanged(int)));

   connect(ui->buttonGroup, qOverload<QAbstractButton *>(&QButtonGroup::buttonClicked), this,
           &ConfigWidget::onCredentialsOptionChanged);
   connect(ui->pbAddCredentials, &QPushButton::clicked, this, &ConfigWidget::showCredentialsDlg);

   // TODO: Download the plugins info
   connect(ui->pbPluginsFolder, &QPushButton::clicked, this, &ConfigWidget::selectPluginsFolder);

   // Connects for automatic save
   connect(ui->chDevMode, &CheckBox::stateChanged, this, &ConfigWidget::enableWidgets);
   connect(ui->chDisableLogs, &CheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbLogLevel, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));
   connect(ui->leGitPath, &QLineEdit::editingFinished, this, &ConfigWidget::saveConfig);
   connect(ui->spCommitTitleLength, SIGNAL(valueChanged(int)), this, SLOT(saveConfig()));
   connect(ui->sbEditorFontSize, SIGNAL(valueChanged(int)), this, SLOT(saveConfig()));
   connect(ui->cbTranslations, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));
   connect(ui->sbMaxCommits, SIGNAL(valueChanged(int)), this, SLOT(saveConfig()));
   connect(ui->cbLogOrder, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));
   connect(ui->autoFetch, SIGNAL(valueChanged(int)), this, SLOT(saveConfig()));
   connect(ui->pruneOnFetch, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->updateOnPull, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->clangFormat, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbPomodoroEnabled, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbLocal, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbRemote, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbTags, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbStash, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbSubmodule, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbSubtree, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbDeleteFolder, &QCheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->pbSelectFolder, &QPushButton::clicked, this, &ConfigWidget::selectFolder);
   connect(ui->pbDefault, &QPushButton::clicked, this, &ConfigWidget::useDefaultLogsFolder);
   connect(ui->leEditor, &QLineEdit::editingFinished, this, &ConfigWidget::saveConfig);
   connect(ui->pbSelectEditor, &QPushButton::clicked, this, &ConfigWidget::selectEditor);
   connect(ui->leExtFileExplorer, &QLineEdit::editingFinished, this, &ConfigWidget::saveConfig);
   connect(mPluginsDownloader, &PluginsDownloader::availablePlugins, this, &ConfigWidget::onPluginsInfoReceived);
   connect(mPluginsDownloader, &PluginsDownloader::pluginStored, this, &ConfigWidget::onPluginStored);
   connect(ui->pbFeaturesTour, &QPushButton::clicked, this, &ConfigWidget::showFeaturesTour);
   connect(ui->chSingleClickDiffView, &CheckBox::stateChanged, this, &ConfigWidget::saveConfig);
   connect(ui->cbDiffView, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));
   connect(ui->cbBranchSeparator, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));
   connect(ui->cbLanguage, SIGNAL(currentIndexChanged(int)), this, SLOT(saveConfig()));

   ui->cbDiffView->setCurrentIndex(settings.globalValue("DefaultDiffView").toInt());
   ui->cbBranchSeparator->setCurrentText(settings.globalValue("BranchSeparator", "-").toString());

   auto size = calculateDirSize(ui->leLogsLocation->text());
   ui->lLogsSize->setText(QString("%1 KB").arg(size));

   size = calculateDirSize(QStandardPaths::writableLocation(QStandardPaths::CacheLocation));
   ui->lCacheSize->setText(QString("%1 KB").arg(size));

   mPluginsDownloader->checkAvailablePlugins();
}

ConfigWidget::~ConfigWidget()
{
   delete ui;
}

void ConfigWidget::onPanelsVisibilityChanged()
{
   GitQlientSettings settings(mGit->getGitDir());

   ui->cbLocal->setChecked(settings.localValue("LocalHeader", true).toBool());
   ui->cbRemote->setChecked(settings.localValue("RemoteHeader", true).toBool());
   ui->cbTags->setChecked(settings.localValue("TagsHeader", true).toBool());
   ui->cbStash->setChecked(settings.localValue("StashesHeader", true).toBool());
   ui->cbSubmodule->setChecked(settings.localValue("SubmodulesHeader", true).toBool());
   ui->cbSubtree->setChecked(settings.localValue("SubtreeHeader", true).toBool());
}

void ConfigWidget::onCredentialsOptionChanged(QAbstractButton *button)
{
   ui->sbTimeout->setEnabled(button == ui->rbCache);
}

void ConfigWidget::onPullStrategyChanged(int index)
{
   QScopedPointer<GitConfig> gitConfig(new GitConfig(mGit));

   switch (index)
   {
      case 0:
         gitConfig->unset("pull.ff");
         gitConfig->setLocalData("pull.rebase", "false");
         break;
      case 1:
         gitConfig->unset("pull.ff");
         gitConfig->setLocalData("pull.rebase", "true");
         break;
      case 2:
         gitConfig->unset("pull.rebase");
         gitConfig->setLocalData("pull.ff", "only");
         break;
   }
}

void ConfigWidget::onPluginsInfoReceived(const QVector<PluginInfo> &pluginsInfo)
{
   mPluginsInfo = pluginsInfo;

   auto row = 0;
   for (const auto &plugin : std::as_const(mPluginsInfo))
   {
      ui->availablePluginsLayout->addWidget(new QLabel(plugin.name), ++row, 0);
      ui->availablePluginsLayout->addWidget(new QLabel(plugin.version), row, 1);

      const auto pbDownload = new QPushButton("Download");
      connect(pbDownload, &QPushButton::clicked, this, [url = plugin.url, dependencies = plugin.dependencies, this]() {
         if (!dependencies.empty())
         {
            QMessageBox::information(
                this, tr("Dependencies needed!"),
                tr("This plugin needs some dependencies to work. Please make sure you have them installed:<br><br>%1")
                    .arg(dependencies.join("<br>")));
         }

         mPluginsDownloader->downloadPlugin(url);
      });
      mDownloadButtons->addButton(pbDownload);

      pbDownload->setDisabled(mPluginNames.contains(plugin.name) || ui->lePluginsDestination->text().isEmpty());

      ui->availablePluginsLayout->addWidget(pbDownload, row, 2);
   }
}

void ConfigWidget::onPluginStored()
{
   QMessageBox::information(this, tr("Reset needed"), tr("You need to restart GitQlient to load the plugins."));
}

void ConfigWidget::clearCache()
{
   const auto path = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
   QProcess p;
   p.setWorkingDirectory(path);
   p.start("rm", { "-rf", path });

   if (p.waitForFinished())
   {
      const auto size = calculateDirSize(path);
      ui->lCacheSize->setText(QString("%1 KB").arg(size));
   }
}

void ConfigWidget::clearLogs()
{
   const auto path = ui->leLogsLocation->text();
   QDir dir(path, { "GitQlient_*.log" });
   const auto entryList = dir.entryList();
   for (const auto &filename : entryList)
      dir.remove(filename);

   const auto size = calculateDirSize(path);
   ui->lLogsSize->setText(QString("%1 KB").arg(size));
}

void ConfigWidget::clearFolder(const QString &folder, QLabel *label)
{
   const auto path = folder;
   QProcess p;
   p.setWorkingDirectory(path);
   p.start("rm", { "-rf", path });

   if (p.waitForFinished())
   {
      const auto size = calculateDirSize(path);
      label->setText(QString("%1 KB").arg(size));
   }
}

uint64_t ConfigWidget::calculateDirSize(const QString &dirPath)
{
   auto size = 0U;
   QDir dir(dirPath);
   QDir::Filters dirFilters = QDir::Dirs | QDir::NoDotAndDotDot | QDir::System | QDir::Hidden | QDir::Files;
   const auto &list = dir.entryInfoList(dirFilters);

   for (const QFileInfo &file : list)
   {
      size += file.size();
      size += dirSize(dirPath + "/" + file.fileName());
   }

   return size / 1024.0;
}

void ConfigWidget::saveConfig()
{
   mFeedbackTimer->stop();

   ui->lFeedback->setText(tr("Changes saved"));

   GitQlientSettings settings(mGit->getGitDir());

   settings.setGlobalValue("logsDisabled", ui->chDisableLogs->isChecked());
   settings.setGlobalValue("logsLevel", ui->cbLogLevel->currentIndex());
   settings.setGlobalValue("logsFolder", ui->leLogsLocation->text());
   settings.setGlobalValue("commitTitleMaxLength", ui->spCommitTitleLength->value());
   settings.setGlobalValue("FileDiffView/FontSize", ui->sbEditorFontSize->value());
   settings.setGlobalValue("colorSchema", ui->cbStyle->currentIndex());
   settings.setGlobalValue("gitLocation", ui->leGitPath->text());
   settings.setGlobalValue("singleClickDiffView", ui->chSingleClickDiffView->isChecked());
   settings.setGlobalValue("DefaultDiffView", ui->cbDiffView->currentIndex());
   settings.setGlobalValue("BranchSeparator", ui->cbBranchSeparator->currentText());
   settings.setGlobalValue("UILanguage", ui->cbLanguage->currentData().toString());

   if (!ui->leEditor->text().isEmpty())
      settings.setGlobalValue("ExternalEditor", ui->leEditor->text());

#ifdef Q_OS_LINUX
   settings.setGlobalValue("FileExplorer", ui->leExtFileExplorer->text());
#endif

   mLocalGit->changeFontSize();
   mGlobalGit->changeFontSize();

   emit reloadDiffFont();
   emit commitTitleMaxLenghtChanged();

   if (mShowResetMsg || qobject_cast<QComboBox*>(sender()) == ui->cbLanguage)
   {
      QMessageBox::information(this, tr("Reset needed!"),
                               tr("You need to restart GitQlient to see the changes in the styles applied."));
   }

   const auto logger = QLoggerManager::getInstance();
   logger->overwriteLogLevel(static_cast<LogLevel>(ui->cbLogLevel->currentIndex()));

   if (ui->chDisableLogs->isChecked())
      logger->pause();
   else
      logger->resume();

   if (mOriginalRepoOrder != ui->cbLogOrder->currentIndex())
   {
      settings.setLocalValue("GraphSortingOrder", ui->cbLogOrder->currentIndex());
      emit reloadView();
   }

   settings.setLocalValue("AutoFetch", ui->autoFetch->value());

   emit autoFetchChanged(ui->autoFetch->value());

   settings.setLocalValue("PruneOnFetch", ui->pruneOnFetch->isChecked());
   settings.setLocalValue("ClangFormatOnCommit", ui->clangFormat->isChecked());
   settings.setLocalValue("UpdateOnPull", ui->updateOnPull->isChecked());
   settings.setLocalValue("MaxCommits", ui->sbMaxCommits->value());

   settings.setLocalValue("LocalHeader", ui->cbLocal->isChecked());
   settings.setLocalValue("RemoteHeader", ui->cbRemote->isChecked());
   settings.setLocalValue("TagsHeader", ui->cbTags->isChecked());
   settings.setLocalValue("StashesHeader", ui->cbStash->isChecked());
   settings.setLocalValue("SubmodulesHeader", ui->cbSubmodule->isChecked());
   settings.setLocalValue("SubtreeHeader", ui->cbSubtree->isChecked());

   settings.setLocalValue("DeleteRemoteFolder", ui->cbDeleteFolder->isChecked());

   emit panelsVisibilityChanged();

   /* POMODORO CONFIG */
   settings.setLocalValue("Pomodoro/Enabled", ui->cbPomodoroEnabled->isChecked());

   emit pomodoroVisibilityChanged();

   mFeedbackTimer->singleShot(3000, ui->lFeedback, &QLabel::clear);
}

void ConfigWidget::enableWidgets()
{
   const auto enable = ui->chDevMode->isChecked();

   GitQlientSettings settings(mGit->getGitDir());
   settings.setLocalValue("DevMode", enable);

   ui->tabWidget->setEnabled(enable);
}

void ConfigWidget::saveFile()
{
   const auto id = ui->tabWidget->currentIndex();

   if (id == 0)
      mLocalGit->saveFile();
   else
      mGlobalGit->saveFile();
}

void ConfigWidget::showCredentialsDlg()
{
   // Store credentials if allowed and the user checked the box
   if (ui->credentialsFrames->isVisible() && ui->chbCredentials->isChecked())
   {
      if (ui->rbCache->isChecked())
         GitCredentials::configureCache(ui->sbTimeout->value(), mGit);
      else
      {
         CredentialsDlg dlg(mGit, this);
         dlg.exec();
      }
   }
}

void ConfigWidget::selectFolder()
{
   const QString dirName(
       QFileDialog::getExistingDirectory(this, "Choose the directory for the GitQlient logs", QDir::currentPath()));

   if (!dirName.isEmpty() && dirName != QDir::currentPath().append("logs"))
   {
      QDir d(dirName);

      const auto ret = QMessageBox::information(
          this, tr("Restart needed!"),
          tr("The folder chosen to store GitQlient logs is: <br> <strong>%1</strong>. If you "
             "confirm the change, GitQlient will move all the logs to that folder. Once done, "
             "GitQlient will close. You need to restart it.")
              .arg(d.absolutePath()),
          QMessageBox::StandardButton::Ok, QMessageBox::StandardButton::Cancel);

      if (ret == QMessageBox::Ok)
      {
         ui->leLogsLocation->setText(d.absolutePath());

         saveConfig();

         emit moveLogsAndClose();
      }
   }
}

void ConfigWidget::selectPluginsFolder()
{
   const QString dirName(
       QFileDialog::getExistingDirectory(this, "Choose the directory for the GitQlient plugins", QDir::currentPath()));

   if (!dirName.isEmpty() && dirName != QDir::currentPath().append("logs"))
   {
      QDir d(dirName);

      ui->lePluginsDestination->setText(d.absolutePath());
      ui->availablePluginsWidget->setEnabled(true);

      ui->lPluginsWarning->setVisible(false);

      const auto buttons = mDownloadButtons->buttons();
      for (const auto &button : buttons)
         button->setEnabled(true);

      QSettings().setValue("PluginsFolder", d.absolutePath());
   }
}

void ConfigWidget::selectEditor()
{
   const QString dirName(
       QFileDialog::getOpenFileName(this, "Choose the directory of the external editor", QDir::currentPath()));

   if (!dirName.isEmpty())
   {
      QDir d(dirName);

      ui->leEditor->setText(d.absolutePath());

      saveConfig();
   }
}

void ConfigWidget::useDefaultLogsFolder()
{
   const auto dir = QDir::currentPath().append("/logs");

   if (dir != ui->leLogsLocation->text())
   {
      const auto ret = QMessageBox::information(
          this, tr("Restart needed!"),
          tr("The folder chosen to store GitQlient logs is: <br> <strong>%1</strong>. If you "
             "confirm the change, GitQlient will move all the logs to that folder. Once done, "
             "GitQlient will close. You need to restart it.")
              .arg(dir),
          QMessageBox::StandardButton::Ok, QMessageBox::StandardButton::Cancel);

      if (ret == QMessageBox::Ok)
      {
         ui->leLogsLocation->setText(dir);

         saveConfig();

         emit moveLogsAndClose();
      }
   }
}

void ConfigWidget::readRemotePluginsInfo() { }

void ConfigWidget::showFeaturesTour()
{
   NewVersionInfoDlg dlg(this);
   dlg.setFixedSize(600, 400);
   dlg.exec();
}

void ConfigWidget::loadPlugins(QMap<QString, QObject *> plugins)
{
   mPluginNames.clear();

   if (plugins.isEmpty())
   {
      const auto buttons = mDownloadButtons->buttons();
      for (const auto &button : buttons)
         button->setEnabled(true);

      return;
   }

   auto row = 1U;
   for (auto iter = plugins.cbegin(); iter != plugins.cend(); ++iter)
   {
      auto metadata = iter.key().split("-");
      const auto labelName = new QLabel(metadata.takeFirst());
      const auto labelVersion = new QLabel(metadata.takeFirst());

      mPluginNames.append(labelName->text());

      ui->pluginsLayout->addWidget(labelName, row, 0);
      ui->pluginsLayout->addWidget(labelVersion, row, 1);

      auto chEnabled = new CheckBox("");
      const auto auxLayout = new QHBoxLayout();
      auxLayout->addStretch();
      auxLayout->addWidget(chEnabled);
      auxLayout->addStretch();

      ui->pluginsLayout->addLayout(auxLayout, row, 2);
      ++row;

      mPluginWidgets.append(labelName);
      mPluginWidgets.append(labelVersion);

      if (labelName->text().contains("qtermwidget", Qt::CaseInsensitive))
      {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
         GitQlientSettings settings(mGit->getGitDir());
         const auto isTerminalEnabled = settings.localValue("TerminalEnabled", false).toBool();
         const auto terminal = qobject_cast<QTermWidgetInterface *>(iter.value());
         const auto availableSchemes = terminal->getAvailableColorSchemes();

         ui->lTerminalColorScheme->setVisible(isTerminalEnabled);
         ui->cbTerminalColorScheme->setVisible(isTerminalEnabled);
         ui->cbTerminalColorScheme->addItems(availableSchemes);

         if (const auto lastScheme = QSettings().value("TerminalScheme", QString()).toString();
             !lastScheme.isEmpty() && availableSchemes.contains(lastScheme))
         {
            ui->cbTerminalColorScheme->setCurrentText(lastScheme);
            terminal->setColorScheme(lastScheme);
         }

         connect(ui->cbTerminalColorScheme, &QComboBox::currentTextChanged, this, [terminal](const QString &newScheme) {
            dynamic_cast<QTermWidgetInterface *>(terminal)->setColorScheme(newScheme);

            QSettings().setValue("TerminalScheme", newScheme);
         });

         chEnabled->setChecked(GitQlientSettings(mGit->getGitDir()).localValue("TerminalEnabled", false).toBool());
         connect(chEnabled, &QCheckBox::stateChanged, this, [this, chEnabled](int) {
            const auto checked = chEnabled->isChecked();
            GitQlientSettings(mGit->getGitDir()).setLocalValue("TerminalEnabled", checked);

            ui->lTerminalColorScheme->setVisible(checked);
            ui->cbTerminalColorScheme->setVisible(checked);

            emit terminalEnabled(checked);
         });
#endif
      }
      else if (labelName->text().contains("jenkinsplugin", Qt::CaseInsensitive))
      {
         chEnabled->setChecked(GitQlientSettings(mGit->getGitDir()).localValue("BuildSystemEnabled", false).toBool());

         connect(chEnabled, &QCheckBox::stateChanged, this, [this, chEnabled]() {
            const auto checked = chEnabled->isChecked();
            GitQlientSettings(mGit->getGitDir()).setLocalValue("BuildSystemEnabled", checked);
            emit buildSystemEnabled(checked);
         });
      }
      else if (labelName->text().contains("gitserverplugin", Qt::CaseInsensitive))
      {
         chEnabled->setChecked(GitQlientSettings(mGit->getGitDir()).localValue("GitServerEnabled", false).toBool());

         connect(chEnabled, &QCheckBox::stateChanged, this, [this, chEnabled]() {
            const auto checked = chEnabled->isChecked();
            GitQlientSettings(mGit->getGitDir()).setLocalValue("GitServerEnabled", checked);
            emit gitServerEnabled(checked);
         });
      }
   }

   for (auto iter = mPluginDataMap.cbegin(); iter != mPluginDataMap.cend(); ++iter)
   {
      iter.key()->setEnabled(!mPluginNames.contains(iter.value().name));
   }
}

void ConfigWidget::fillLanguageBox() const
{
   const auto currentLanguage = GitQlientSettings().globalValue("UILanguage", "gitqlient_en").toString();

   const auto list = QDir(":translations", "gitqlient_*.qm").entryList();
   QDirIterator trIter(":translations", QStringList() << "gitqlient_*.qm");

   while (trIter.hasNext())
   {
      trIter.next();

      auto name = trIter.fileName();
      name.remove(".qm");

      const auto lang = name.mid(name.indexOf('_') + 1);
      QLocale tmpLocale(lang);
      const auto languageItem = QString::fromUtf8("%1 (%2)").arg(QLocale::languageToString(tmpLocale.language()),
                                                                   QLocale::countryToString(tmpLocale.country()));

      ui->cbLanguage->addItem(languageItem, name);

      if (name == currentLanguage)
         ui->cbLanguage->setCurrentIndex(ui->cbLanguage->count() - 1);
   }
}
