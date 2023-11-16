#include "GitQlientRepo.h"

#include <BlameWidget.h>
#include <BranchesWidget.h>
#include <CommitHistoryColumns.h>
#include <CommitInfo.h>
#include <ConfigData.h>
#include <ConfigWidget.h>
#include <Controls.h>
#include <DiffWidget.h>
#include <GitBase.h>
#include <GitCache.h>
#include <GitConfig.h>
#include <GitConfigDlg.h>
#include <GitHistory.h>
#include <GitLocal.h>
#include <GitMerge.h>
#include <GitQlientSettings.h>
#include <GitQlientStyles.h>
#include <GitRepoLoader.h>
#include <GitServerTypes.h>
#include <GitSubmodules.h>
#include <GitTags.h>
#include <GitWip.h>
#include <HistoryWidget.h>
#include <IGitServerCache.h>
#include <IGitServerWidget.h>
#include <IJenkinsWidget.h>
#include <MergeWidget.h>
#include <QLogger.h>
#include <WaitingDlg.h>
#include <WipHelper.h>

#include <QApplication>
#include <QDirIterator>
#include <QFileDialog>
#include <QGridLayout>
#include <QMessageBox>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QTimer>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#   include <qtermwidget_interface.h>
#endif

using namespace QLogger;
using namespace GitServerPlugin;

GitQlientRepo::GitQlientRepo(const QSharedPointer<GitBase> &git, const QSharedPointer<GitQlientSettings> &settings,
                             QWidget *parent)
   : QFrame(parent)
   , mGitQlientCache(new GitCache())
   , mGitBase(git)
   , mSettings(settings)
   , mGitLoader(new GitRepoLoader(mGitBase, mGitQlientCache, mSettings))
   , mAutoFetch(new QTimer())
   , mAutoFilesUpdate(new QTimer())
{
   setAttribute(Qt::WA_DeleteOnClose);

   QLog_Info("UI", QString("Initializing GitQlient for repo %1").arg(git->getGitDir()));

   setObjectName("mainWindow");
   setWindowTitle("GitQlient");
   setAttribute(Qt::WA_DeleteOnClose);

   mStackedLayout = new QStackedLayout();

   mHistoryWidget = new HistoryWidget(mGitQlientCache, mGitBase, mSettings, this);
   mHistoryWidget->setContentsMargins(QMargins(5, 5, 5, 5));

   mDiffWidget = new DiffWidget(mGitBase, mGitQlientCache, this);
   mDiffWidget->setContentsMargins(QMargins(5, 5, 5, 5));

   mBlameWidget = new BlameWidget(mGitQlientCache, mGitBase, mSettings, this);
   mBlameWidget->setContentsMargins(QMargins(5, 5, 5, 5));

   mMergeWidget = new MergeWidget(mGitQlientCache, mGitBase, this);
   mMergeWidget->setContentsMargins(QMargins(5, 5, 5, 5));

   mConfigWidget = new ConfigWidget(mGitBase, this);
   mConfigWidget->setContentsMargins(QMargins(5, 5, 5, 5));

   mIndexMap[ControlsMainViews::History] = mStackedLayout->addWidget(mHistoryWidget);
   mIndexMap[ControlsMainViews::Diff] = mStackedLayout->addWidget(mDiffWidget);
   mIndexMap[ControlsMainViews::Blame] = mStackedLayout->addWidget(mBlameWidget);
   mIndexMap[ControlsMainViews::Merge] = mStackedLayout->addWidget(mMergeWidget);
   mIndexMap[ControlsMainViews::Config] = mStackedLayout->addWidget(mConfigWidget);

   mControls = new Controls(mGitQlientCache, mGitBase, this);

   const auto mainLayout = new QVBoxLayout();
   mainLayout->setSpacing(0);
   mainLayout->setContentsMargins(QMargins());
   mainLayout->addWidget(mControls);
   mainLayout->addLayout(mStackedLayout);

   setLayout(mainLayout);

   showHistoryView();

   if (const auto fetchInterval = mSettings->localValue("AutoFetch", 5).toInt(); fetchInterval > 0)
      mAutoFetch->setInterval(fetchInterval * 60 * 1000);

   mAutoFilesUpdate->setInterval(15000);

   connect(mAutoFetch, &QTimer::timeout, mControls, &Controls::fetchAll);
   connect(mAutoFilesUpdate, &QTimer::timeout, this, &GitQlientRepo::updateUiFromWatcher);

   connect(mControls, &Controls::requestFullReload, this, &GitQlientRepo::fullReload);
   connect(mControls, &Controls::requestFullReload, this, &GitQlientRepo::updateUiFromWatcher);
   connect(mControls, &Controls::requestReferencesReload, this, &GitQlientRepo::referencesReload);

   connect(mControls, &Controls::signalGoRepo, this, &GitQlientRepo::showHistoryView);
   connect(mControls, &Controls::signalGoBlame, this, &GitQlientRepo::showBlameView);
   connect(mControls, &Controls::signalGoDiff, this, &GitQlientRepo::showDiffView);
   connect(mControls, &Controls::signalGoMerge, this, &GitQlientRepo::showMergeView);
   connect(mControls, &Controls::signalGoServer, this, &GitQlientRepo::showGitServerView);
   connect(mControls, &Controls::signalGoBuildSystem, this, &GitQlientRepo::showBuildSystemView);
   connect(mControls, &Controls::goConfig, this, &GitQlientRepo::showConfig);
   connect(mControls, &Controls::goTerminal, this, &GitQlientRepo::showTerminal);
   connect(mControls, &Controls::signalPullConflict, mControls, &Controls::activateMergeWarning);
   connect(mControls, &Controls::signalPullConflict, this, &GitQlientRepo::showWarningMerge);

   connect(mHistoryWidget, &HistoryWidget::signalAllBranchesActive, mGitLoader.data(), &GitRepoLoader::setShowAll);
   connect(mHistoryWidget, &HistoryWidget::fullReload, this, &GitQlientRepo::fullReload);
   connect(mHistoryWidget, &HistoryWidget::referencesReload, this, &GitQlientRepo::referencesReload);
   connect(mHistoryWidget, &HistoryWidget::logReload, this, &GitQlientRepo::logReload);

   connect(mHistoryWidget, &HistoryWidget::panelsVisibilityChanged, mConfigWidget,
           &ConfigWidget::onPanelsVisibilityChanged);
   connect(mHistoryWidget, &HistoryWidget::signalOpenSubmodule, this, &GitQlientRepo::signalOpenSubmodule);
   connect(mHistoryWidget, &HistoryWidget::signalShowDiff, this, &GitQlientRepo::loadFileDiff);
   connect(mHistoryWidget, &HistoryWidget::signalOpenDiff, this, &GitQlientRepo::openCommitDiff);
   connect(mHistoryWidget, &HistoryWidget::changesCommitted, this, &GitQlientRepo::onChangesCommitted);
   connect(mHistoryWidget, &HistoryWidget::signalShowFileHistory, this, &GitQlientRepo::showFileHistory);
   connect(mHistoryWidget, &HistoryWidget::signalRebaseConflict, mControls, &Controls::activateMergeWarning);
   connect(mHistoryWidget, &HistoryWidget::signalRebaseConflict, this, &GitQlientRepo::showRebaseConflict);
   connect(mHistoryWidget, &HistoryWidget::signalMergeConflicts, mControls, &Controls::activateMergeWarning);
   connect(mHistoryWidget, &HistoryWidget::signalMergeConflicts, this, &GitQlientRepo::showWarningMerge);
   connect(mHistoryWidget, &HistoryWidget::signalCherryPickConflict, mControls, &Controls::activateMergeWarning);
   connect(mHistoryWidget, &HistoryWidget::signalCherryPickConflict, this, &GitQlientRepo::showCherryPickConflict);
   connect(mHistoryWidget, &HistoryWidget::signalPullConflict, mControls, &Controls::activateMergeWarning);
   connect(mHistoryWidget, &HistoryWidget::signalPullConflict, this, &GitQlientRepo::showWarningMerge);
   connect(mHistoryWidget, &HistoryWidget::signalUpdateWip, this, &GitQlientRepo::updateWip);
   connect(mHistoryWidget, &HistoryWidget::showPrDetailedView, this, &GitQlientRepo::showGitServerPrView);

   connect(mDiffWidget, &DiffWidget::signalShowFileHistory, this, &GitQlientRepo::showFileHistory);
   connect(mDiffWidget, &DiffWidget::signalDiffEmpty, mControls, &Controls::disableDiff);
   connect(mDiffWidget, &DiffWidget::signalDiffEmpty, this, &GitQlientRepo::showPreviousView);

   connect(mBlameWidget, &BlameWidget::showFileDiff, this, &GitQlientRepo::loadFileDiff);

   connect(mMergeWidget, &MergeWidget::signalMergeFinished, this, &GitQlientRepo::showHistoryView);
   connect(mMergeWidget, &MergeWidget::signalMergeFinished, mGitLoader.data(), &GitRepoLoader::loadAll);
   connect(mMergeWidget, &MergeWidget::signalMergeFinished, mControls, &Controls::disableMergeWarning);

   connect(mConfigWidget, &ConfigWidget::commitTitleMaxLenghtChanged, mHistoryWidget,
           &HistoryWidget::onCommitTitleMaxLenghtChanged);
   connect(mConfigWidget, &ConfigWidget::panelsVisibilityChanged, mHistoryWidget,
           &HistoryWidget::onPanelsVisibilityChanged);
   connect(mConfigWidget, &ConfigWidget::reloadDiffFont, mHistoryWidget, &HistoryWidget::onDiffFontSizeChanged);
   connect(mConfigWidget, &ConfigWidget::pomodoroVisibilityChanged, mControls, &Controls::changePomodoroVisibility);
   connect(mConfigWidget, &ConfigWidget::moveLogsAndClose, this, &GitQlientRepo::moveLogsAndClose);
   connect(mConfigWidget, &ConfigWidget::autoFetchChanged, this, &GitQlientRepo::reconfigureAutoFetch);
   connect(mConfigWidget, &ConfigWidget::buildSystemEnabled, this, &GitQlientRepo::buildSystemActivationToggled);
   connect(mConfigWidget, &ConfigWidget::gitServerEnabled, this, &GitQlientRepo::gitServerActivationToggled);
   connect(mConfigWidget, &ConfigWidget::terminalEnabled, this, &GitQlientRepo::terminalActivationToggled);

   connect(mGitLoader.data(), &GitRepoLoader::signalLoadingStarted, this, &GitQlientRepo::createProgressDialog);
   connect(mGitLoader.data(), &GitRepoLoader::signalLoadingFinished, this, &GitQlientRepo::onRepoLoadFinished);

   m_loaderThread = new QThread();
   mGitLoader->moveToThread(m_loaderThread);
   mGitQlientCache->moveToThread(m_loaderThread);
   connect(this, &GitQlientRepo::loadRepo, mGitLoader.data(), &GitRepoLoader::loadAll);
   connect(this, &GitQlientRepo::fullReload, mGitLoader.data(), &GitRepoLoader::loadAll);
   connect(this, &GitQlientRepo::referencesReload, mGitLoader.data(), &GitRepoLoader::loadReferences);
   connect(this, &GitQlientRepo::logReload, mGitLoader.data(), &GitRepoLoader::loadLogHistory);
   m_loaderThread->start();

   mGitLoader->setShowAll(mSettings->localValue("ShowAllBranches", true).toBool());
}

GitQlientRepo::~GitQlientRepo()
{
   mStackedLayout->widget(mIndexMap[ControlsMainViews::Terminal])->deleteLater();

   delete mAutoFetch;
   delete mAutoFilesUpdate;

   m_loaderThread->exit();
   m_loaderThread->wait();

   delete m_loaderThread;
}

QString GitQlientRepo::currentBranch() const
{
   return mGitBase->getCurrentBranch();
}

void GitQlientRepo::updateUiFromWatcher()
{
   QLog_Info("UI", QString("Updating the GitQlient UI from watcher"));

   WipHelper::update(mGitBase, mGitQlientCache);

   mHistoryWidget->updateUiFromWatcher();

   mDiffWidget->reload();
}

void GitQlientRepo::openCommitDiff(const QString currentSha)
{
   const auto rev = mGitQlientCache->commitInfo(currentSha);
   const auto loaded = mDiffWidget->loadCommitDiff(currentSha, rev.firstParent());

   if (loaded)
   {
      mControls->enableDiff();

      showDiffView();
   }
}

void GitQlientRepo::setPlugins(QMap<QString, QObject *> plugins)
{
   mConfigWidget->loadPlugins(plugins);

   if (plugins.isEmpty())
      return;

   for (auto iter = plugins.constBegin(); iter != plugins.constEnd(); ++iter)
   {
      if (iter.key().split("-").constFirst().contains("jenkins", Qt::CaseInsensitive)
          && qobject_cast<QWidget *>(iter.value()))
      {
         mJenkins = qobject_cast<IJenkinsWidget *>(iter.value());
         mJenkins->setContentsMargins(QMargins(5, 5, 5, 5));
         connect(mJenkins, &IJenkinsWidget::gotoBranch, this, &GitQlientRepo::focusHistoryOnBranch);
         connect(mJenkins, &IJenkinsWidget::gotoPullRequest, this, &GitQlientRepo::focusHistoryOnPr);

         if (mJenkins->isConfigured())
            mJenkins->update();

         mControls->showJenkinsButton(true);
         mIndexMap[ControlsMainViews::BuildSystem] = mStackedLayout->addWidget(mJenkins);
      }
      else if (iter.key().split("-").constFirst().contains("gitserver", Qt::CaseInsensitive)
               && qobject_cast<QWidget *>(iter.value()))
      {

         mGitServerWidget = qobject_cast<IGitServerWidget *>(iter.value());
         mGitServerWidget->setContentsMargins(QMargins(5, 5, 5, 5));

         mControls->showGitServerButton(true);
         mIndexMap[ControlsMainViews::GitServer] = mStackedLayout->addWidget(mGitServerWidget);
      }
      else if (iter.key().split("-").constFirst().contains("qtermwidget", Qt::CaseInsensitive)
               && qobject_cast<QWidget *>(iter.value()))
      {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
         QFont font = QApplication::font();
#   ifdef Q_OS_MACOS
         font.setFamily(QStringLiteral("Monaco"));
#   elif defined(Q_WS_QWS)
         font.setFamily(QStringLiteral("fixed"));
#   else
         font.setFamily(QStringLiteral("Monospace"));
#   endif
         font.setPointSize(12);

         const auto terminalWidget = qobject_cast<QTermWidgetInterface *>(iter.value());
         terminalWidget->setTerminalFont(font);
         terminalWidget->setScrollBarPosition(QTermWidgetInterface::ScrollBarRight);
         terminalWidget->setBlinkingCursor(true);
         terminalWidget->setWorkingDirectory(mGitBase->getWorkingDir());
         terminalWidget->startShellProgram();

         QTimer::singleShot(250, this, [terminalWidget]() {
            const auto historySize = terminalWidget->historySize();
            terminalWidget->setHistorySize(0);
            terminalWidget->sendText(
                QString::fromUtf8(" export TERM=xterm-color\n source  ~/.bashrc\n alias "
                                  "exit=\"echo \\\"The exit command has been disabled\\\"\"\n clear\n"));
            terminalWidget->setHistorySize(historySize);
         });

         auto widget = dynamic_cast<QWidget *>(iter.value());
         widget->setContentsMargins(QMargins(5, 5, 5, 5));
         mIndexMap[ControlsMainViews::Terminal] = mStackedLayout->addWidget(widget);

         mControls->showTerminalButton(true);
#endif
      }
      else
         mPlugins[iter.key()] = iter.value();
   }
}

void GitQlientRepo::clearWindow()
{
   blockSignals(true);

   mHistoryWidget->clear();
   mDiffWidget->clear();

   blockSignals(false);
}

void GitQlientRepo::setWidgetsEnabled(bool enabled)
{
   mControls->enableButtons(enabled);
   mHistoryWidget->setEnabled(enabled);
   mDiffWidget->setEnabled(enabled);
}

void GitQlientRepo::showFileHistory(const QString &fileName)
{
   mBlameWidget->showFileHistory(fileName);

   showBlameView();
}

void GitQlientRepo::createProgressDialog()
{
   if (!mWaitDlg)
   {
      mWaitDlg = new WaitingDlg(tr("Loading repository..."));
      mWaitDlg->setWindowFlag(Qt::Tool);
      mWaitDlg->open();

      QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
   }
}

void GitQlientRepo::onRepoLoadFinished()
{
   if (!mIsInit)
   {
      mIsInit = true;

      mCurrentDir = mGitBase->getWorkingDir();

      emit repoOpened(mCurrentDir);

      setWidgetsEnabled(true);

      mBlameWidget->init(mCurrentDir);

      mControls->enableButtons(true);

      mAutoFilesUpdate->start();

      if (const auto fetchInterval = mSettings->localValue("AutoFetch", 5).toInt(); fetchInterval > 0)
         mAutoFetch->start();

      QScopedPointer<GitConfig> git(new GitConfig(mGitBase));

      if (!git->getGlobalUserInfo().isValid() && !git->getLocalUserInfo().isValid())
      {
         QLog_Info("UI", QString("Configuring Git..."));

         GitConfigDlg configDlg(mGitBase);

         configDlg.exec();

         QLog_Info("UI", QString("... Git configured!"));
      }

      QLog_Info("UI", "... repository loaded successfully");
   }

   const auto totalCommits = mGitQlientCache->commitCount();

   if (totalCommits == 0)
   {
      if (mWaitDlg)
         mWaitDlg->close();

      return;
   }

   mHistoryWidget->updateGraphView(totalCommits);

   mBlameWidget->onNewRevisions(totalCommits);

   mDiffWidget->reload();

   if (mWaitDlg)
      mWaitDlg->close();

   if (QScopedPointer<GitMerge> gitMerge(new GitMerge(mGitBase)); gitMerge->isInMerge())
   {
      mControls->activateMergeWarning();
      showWarningMerge();

      QMessageBox::warning(this, tr("Merge in progress"),
                           tr("There is a merge conflict in progress. Solve the merge before moving on."));
   }
   else if (QScopedPointer<GitLocal> gitMerge(new GitLocal(mGitBase)); gitMerge->isInCherryPickMerge())
   {
      mControls->activateMergeWarning();
      showCherryPickConflict();

      QMessageBox::warning(
          this, tr("Cherry-pick in progress"),
          tr("There is a cherry-pick in progress that contains with conflicts. Solve them before moving on."));
   }

   emit currentBranchChanged();
}

void GitQlientRepo::loadFileDiff(const QString &currentSha, const QString &previousSha, const QString &file)
{
   const auto loaded = mDiffWidget->loadFileDiff(currentSha, previousSha, file);

   if (loaded)
   {
      mControls->enableDiff();
      showDiffView();
   }
}

void GitQlientRepo::showHistoryView()
{
   mPreviousView = mStackedLayout->currentIndex();

   mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::History]);
   mControls->toggleButton(ControlsMainViews::History);
}

void GitQlientRepo::showBlameView()
{
   mPreviousView = mStackedLayout->currentIndex();

   mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::Blame]);
   mControls->toggleButton(ControlsMainViews::Blame);
}

void GitQlientRepo::showDiffView()
{
   mPreviousView = mStackedLayout->currentIndex();

   mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::Diff]);
   mControls->toggleButton(ControlsMainViews::Diff);
}

// TODO: Optimize
void GitQlientRepo::showWarningMerge()
{
   showMergeView();

   const auto wipCommit = mGitQlientCache->commitInfo(ZERO_SHA);

   WipHelper::update(mGitBase, mGitQlientCache);

   const auto file = mGitQlientCache->revisionFile(ZERO_SHA, wipCommit.firstParent());

   if (file)
      mMergeWidget->configure(file.value(), MergeWidget::ConflictReason::Merge);
}

void GitQlientRepo::showRebaseConflict()
{
   showMergeView();

   const auto wipCommit = mGitQlientCache->commitInfo(ZERO_SHA);

   WipHelper::update(mGitBase, mGitQlientCache);

   const auto files = mGitQlientCache->revisionFile(ZERO_SHA, wipCommit.firstParent());

   if (files)
      mMergeWidget->configureForRebase();
}

// TODO: Optimize
void GitQlientRepo::showCherryPickConflict(const QStringList &shas)
{
   showMergeView();

   const auto wipCommit = mGitQlientCache->commitInfo(ZERO_SHA);

   WipHelper::update(mGitBase, mGitQlientCache);

   const auto files = mGitQlientCache->revisionFile(ZERO_SHA, wipCommit.firstParent());

   if (files)
      mMergeWidget->configureForCherryPick(files.value(), shas);
}

// TODO: Optimize
void GitQlientRepo::showPullConflict()
{
   showMergeView();

   const auto wipCommit = mGitQlientCache->commitInfo(ZERO_SHA);

   WipHelper::update(mGitBase, mGitQlientCache);

   const auto files = mGitQlientCache->revisionFile(ZERO_SHA, wipCommit.firstParent());

   if (files)
      mMergeWidget->configure(files.value(), MergeWidget::ConflictReason::Pull);
}

void GitQlientRepo::showMergeView()
{
   mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::Merge]);
   mControls->toggleButton(ControlsMainViews::Merge);
}

bool GitQlientRepo::configureGitServer() const
{
   bool isConfigured = false;

   if (mSettings->localValue("GitServerEnabled", false).toBool())
   {
      const auto remoteBranches = mGitQlientCache->getBranches(References::Type::RemoteBranches);

      if (!mGitServerWidget->isConfigured())
      {
         QScopedPointer<GitConfig> gitConfig(new GitConfig(mGitBase));
         const auto serverUrl = gitConfig->getServerHost();
         const auto repoInfo = gitConfig->getCurrentRepoAndOwner();

         GitQlientSettings settings("");
         const auto user = settings.globalValue(QString("%1/user").arg(serverUrl)).toString();
         const auto token = settings.globalValue(QString("%1/token").arg(serverUrl)).toString();
         const auto endPoint = settings.globalValue(QString("%1/endpoint").arg(serverUrl)).toString();

         GitServerPlugin::ConfigData data;
         data.user = user;
         data.token = token;
         data.serverUrl = serverUrl;
         data.repoOwner = repoInfo.first;
         data.repoName = repoInfo.second;
         data.endPoint = endPoint;

         isConfigured = mGitServerWidget->configure(data, remoteBranches, GitQlientStyles::getStyles());
      }
      else
         isConfigured = true;

      if (isConfigured)
         mGitServerWidget->start(remoteBranches);
   }

   return isConfigured;
}

void GitQlientRepo::showGitServerView()
{
   if (configureGitServer())
   {
      mGitServerCache = mGitServerWidget->getCache();
      mHistoryWidget->enableGitServerFeatures(mGitServerCache);

      mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::GitServer]);
      mControls->toggleButton(ControlsMainViews::GitServer);
   }
   else
      showPreviousView();
}

void GitQlientRepo::showGitServerPrView(int prNumber)
{
   if (configureGitServer())
   {
      showGitServerView();
      mGitServerWidget->openPullRequest(prNumber);
   }
}

void GitQlientRepo::showBuildSystemView()
{
   if (mSettings->localValue("BuildSystemEnabled", false).toBool())
   {
      if (mJenkins->isConfigured())
         mJenkins->update();
      else
      {
         mJenkins->configure({ mSettings->localValue("BuildSystemUser", "").toString(),
                               mSettings->localValue("BuildSystemToken", "").toString(),
                               mSettings->localValue("BuildSystemUrl", "").toString() },
                             GitQlientStyles::getStyles());
         mJenkins->start();
      }

      mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::BuildSystem]);
   }

   mControls->toggleButton(ControlsMainViews::BuildSystem);
}

void GitQlientRepo::buildSystemActivationToggled(bool enabled)
{
   mControls->enableJenkins(enabled);
}

void GitQlientRepo::gitServerActivationToggled(bool enabled)
{
   mControls->enableGitServer(enabled);
}

void GitQlientRepo::terminalActivationToggled(bool enabled)
{
   mControls->enableTerminal(enabled);
}

void GitQlientRepo::showConfig()
{
   mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::Config]);
   mControls->toggleButton(ControlsMainViews::Config);
}

void GitQlientRepo::showTerminal()
{
   mStackedLayout->setCurrentIndex(mIndexMap[ControlsMainViews::Terminal]);
   mControls->toggleButton(ControlsMainViews::Terminal);
}

void GitQlientRepo::showPreviousView()
{
   mStackedLayout->setCurrentIndex(mPreviousView);
   mControls->toggleButton(static_cast<ControlsMainViews>(mPreviousView));
}

void GitQlientRepo::updateWip()
{
   mHistoryWidget->resetWip();

   WipHelper::update(mGitBase, mGitQlientCache);

   mHistoryWidget->updateUiFromWatcher();
}

void GitQlientRepo::focusHistoryOnBranch(const QString &branch)
{
   auto found = false;
   const auto fullBranch = QString("origin/%1").arg(branch);
   auto remoteBranches = mGitQlientCache->getBranches(References::Type::RemoteBranches);

   for (const auto &remote : remoteBranches)
   {
      if (remote.second.contains(fullBranch))
      {
         found = true;
         mHistoryWidget->focusOnCommit(remote.first);
         showHistoryView();
      }
   }

   remoteBranches.clear();
   remoteBranches.squeeze();

   if (!found)
      QMessageBox::information(
          this, tr("Branch not found"),
          tr("The branch couldn't be found. Please, make sure you fetched and have the latest changes."));
}

void GitQlientRepo::focusHistoryOnPr(int prNumber)
{
   const auto pr = mGitServerCache->getPullRequest(prNumber);

   mHistoryWidget->focusOnCommit(pr.state.sha);
   showHistoryView();
}

void GitQlientRepo::reconfigureAutoFetch(int newInterval)
{
   if (newInterval > 0)
      mAutoFetch->start(newInterval * 60 * 1000);
   else
      mAutoFetch->stop();
}

void GitQlientRepo::onChangesCommitted()
{
   mHistoryWidget->selectCommit(ZERO_SHA);
   mHistoryWidget->loadBranches(false);
   showHistoryView();
}

void GitQlientRepo::closeEvent(QCloseEvent *ce)
{
   QLog_Info("UI", QString("Closing GitQlient for repository {%1}").arg(mCurrentDir));

   mGitLoader->cancelAll();

   QWidget::closeEvent(ce);
}
