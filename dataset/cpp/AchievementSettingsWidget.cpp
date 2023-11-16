/*  PCSX2 - PS2 Emulator for PCs
 *  Copyright (C) 2002-2023 PCSX2 Dev Team
 *
 *  PCSX2 is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU Lesser General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  PCSX2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with PCSX2.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

#include "PrecompiledHeader.h"

#include "AchievementSettingsWidget.h"
#include "AchievementLoginDialog.h"
#include "MainWindow.h"
#include "SettingsWindow.h"
#include "SettingWidgetBinder.h"
#include "QtUtils.h"

#include "pcsx2/Achievements.h"
#include "pcsx2/Host.h"

#include "common/StringUtil.h"

#include <QtCore/QDateTime>
#include <QtWidgets/QMessageBox>

AchievementSettingsWidget::AchievementSettingsWidget(SettingsWindow* dialog, QWidget* parent)
	: QWidget(parent)
	, m_dialog(dialog)
{
	SettingsInterface* sif = dialog->getSettingsInterface();

	m_ui.setupUi(this);

	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.enable, "Achievements", "Enabled", false);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.hardcoreMode, "Achievements", "ChallengeMode", false);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.achievementNotifications, "Achievements", "Notifications", true);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.leaderboardNotifications, "Achievements", "LeaderboardNotifications", true);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.soundEffects, "Achievements", "SoundEffects", true);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.overlays, "Achievements", "Overlays", true);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.encoreMode, "Achievements", "EncoreMode", false);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.spectatorMode, "Achievements", "SpectatorMode", false);
	SettingWidgetBinder::BindWidgetToBoolSetting(sif, m_ui.unofficialAchievements, "Achievements", "UnofficialTestMode",false);
	SettingWidgetBinder::BindWidgetToIntSetting(sif, m_ui.achievementNotificationsDuration, "Achievements", "NotificationsDuration", Pcsx2Config::AchievementsOptions::DEFAULT_NOTIFICATION_DURATION);
	SettingWidgetBinder::BindWidgetToIntSetting(sif, m_ui.leaderboardNotificationsDuration, "Achievements", "LeaderboardsDuration", Pcsx2Config::AchievementsOptions::DEFAULT_LEADERBOARD_DURATION);

	dialog->registerWidgetHelp(m_ui.enable, tr("Enable Achievements"), tr("Unchecked"), tr("When enabled and logged in, PCSX2 will scan for achievements on startup."));
	dialog->registerWidgetHelp(m_ui.hardcoreMode, tr("Enable Hardcore Mode"), tr("Unchecked"), tr("\"Challenge\" mode for achievements, including leaderboard tracking. Disables save state, cheats, and slowdown functions."));
	dialog->registerWidgetHelp(m_ui.achievementNotifications, tr("Show Achievement Notifications"), tr("Checked"), tr("Displays popup messages on events such as achievement unlocks and game completion."));
	dialog->registerWidgetHelp(m_ui.leaderboardNotifications, tr("Show Leaderboard Notifications"), tr("Checked"), tr("Displays popup messages when starting, submitting, or failing a leaderboard challenge."));
	dialog->registerWidgetHelp(m_ui.soundEffects, tr("Enable Sound Effects"), tr("Checked"), tr("Plays sound effects for events such as achievement unlocks and leaderboard submissions."));
	dialog->registerWidgetHelp(m_ui.overlays, tr("Enable In-Game Overlays"), tr("Checked"), tr("Shows icons in the lower-right corner of the screen when a challenge/primed achievement is active."));
	dialog->registerWidgetHelp(m_ui.encoreMode, tr("Enable Encore Mode"), tr("Unchecked"),tr("When enabled, each session will behave as if no achievements have been unlocked."));
	dialog->registerWidgetHelp(m_ui.spectatorMode, tr("Enable Spectator Mode"), tr("Unchecked"), tr("When enabled, PCSX2 will assume all achievements are locked and not send any unlock notifications to the server."));
	dialog->registerWidgetHelp(m_ui.unofficialAchievements, tr("Test Unofficial Achievements"), tr("Unchecked"), tr("When enabled, PCSX2 will list achievements from unofficial sets. Please note that these achievements are not tracked by RetroAchievements, so they unlock every time."));

	connect(m_ui.enable, &QCheckBox::stateChanged, this, &AchievementSettingsWidget::updateEnableState);
	connect(m_ui.hardcoreMode, &QCheckBox::stateChanged, this, &AchievementSettingsWidget::updateEnableState);
	connect(m_ui.hardcoreMode, &QCheckBox::stateChanged, this, &AchievementSettingsWidget::onHardcoreModeStateChanged);
	connect(m_ui.achievementNotifications, &QCheckBox::stateChanged, this, &AchievementSettingsWidget::updateEnableState);
	connect(m_ui.leaderboardNotifications, &QCheckBox::stateChanged, this, &AchievementSettingsWidget::updateEnableState);
	connect(m_ui.achievementNotificationsDuration, &QSlider::valueChanged, this, &AchievementSettingsWidget::onAchievementsNotificationDurationSliderChanged);
	connect(m_ui.leaderboardNotificationsDuration, &QSlider::valueChanged, this, &AchievementSettingsWidget::onLeaderboardsNotificationDurationSliderChanged);

	if (!m_dialog->isPerGameSettings())
	{
		connect(m_ui.loginButton, &QPushButton::clicked, this, &AchievementSettingsWidget::onLoginLogoutPressed);
		connect(m_ui.viewProfile, &QPushButton::clicked, this, &AchievementSettingsWidget::onViewProfilePressed);
		connect(g_emu_thread, &EmuThread::onAchievementsRefreshed, this, &AchievementSettingsWidget::onAchievementsRefreshed);
		updateLoginState();

		// force a refresh of game info
		Host::RunOnCPUThread(Host::OnAchievementsRefreshed);
	}
	else
	{
		// remove login and game info, not relevant for per-game
		m_ui.verticalLayout->removeWidget(m_ui.gameInfoBox);
		m_ui.gameInfoBox->deleteLater();
		m_ui.gameInfoBox = nullptr;
		m_ui.verticalLayout->removeWidget(m_ui.loginBox);
		m_ui.loginBox->deleteLater();
		m_ui.loginBox = nullptr;
	}

	updateEnableState();
	onAchievementsNotificationDurationSliderChanged();
	onLeaderboardsNotificationDurationSliderChanged();
}

AchievementSettingsWidget::~AchievementSettingsWidget() = default;

void AchievementSettingsWidget::updateEnableState()
{
	const bool enabled = m_dialog->getEffectiveBoolValue("Achievements", "Enabled", false);
	const bool notifications = enabled && m_dialog->getEffectiveBoolValue("Achievements", "Notifications", true);
	const bool lb_notifications = enabled && m_dialog->getEffectiveBoolValue("Achievements", "LeaderboardNotifications", true);
	m_ui.hardcoreMode->setEnabled(enabled);
	m_ui.achievementNotifications->setEnabled(enabled);
	m_ui.leaderboardNotifications->setEnabled(enabled);
	m_ui.achievementNotificationsDuration->setEnabled(notifications);
	m_ui.achievementNotificationsDurationLabel->setEnabled(notifications);
	m_ui.leaderboardNotificationsDuration->setEnabled(lb_notifications);
	m_ui.leaderboardNotificationsDurationLabel->setEnabled(lb_notifications);
	m_ui.soundEffects->setEnabled(enabled);
	m_ui.overlays->setEnabled(enabled);
	m_ui.encoreMode->setEnabled(enabled);
	m_ui.spectatorMode->setEnabled(enabled);
	m_ui.unofficialAchievements->setEnabled(enabled);
}

void AchievementSettingsWidget::onHardcoreModeStateChanged()
{
	if (!QtHost::IsVMValid())
		return;

	const bool enabled = m_dialog->getEffectiveBoolValue("Achievements", "Enabled", false);
	const bool challenge = m_dialog->getEffectiveBoolValue("Achievements", "ChallengeMode", false);
	if (!enabled || !challenge)
		return;

	// don't bother prompting if the game doesn't have achievements
	auto lock = Achievements::GetLock();
	if (!Achievements::HasActiveGame() || !Achievements::HasAchievementsOrLeaderboards())
		return;

	if (QMessageBox::question(
			QtUtils::GetRootWidget(this), tr("Reset System"),
			tr("Hardcore mode will not be enabled until the system is reset. Do you want to reset the system now?")) !=
		QMessageBox::Yes)
	{
		return;
	}

	g_emu_thread->resetVM();
}

void AchievementSettingsWidget::onAchievementsNotificationDurationSliderChanged()
{
	const float duration = m_dialog->getEffectiveFloatValue("Achievements", "NotificationsDuration",
		Pcsx2Config::AchievementsOptions::DEFAULT_NOTIFICATION_DURATION);
	m_ui.achievementNotificationsDurationLabel->setText(tr("%n seconds", nullptr, static_cast<int>(duration)));
}

void AchievementSettingsWidget::onLeaderboardsNotificationDurationSliderChanged()
{
	const float duration = m_dialog->getEffectiveFloatValue("Achievements", "LeaderboardsDuration",
		Pcsx2Config::AchievementsOptions::DEFAULT_LEADERBOARD_DURATION);
	m_ui.leaderboardNotificationsDurationLabel->setText(tr("%n seconds", nullptr, static_cast<int>(duration)));
}

void AchievementSettingsWidget::updateLoginState()
{
	const std::string username(Host::GetBaseStringSettingValue("Achievements", "Username"));
	const bool logged_in = !username.empty();

	if (logged_in)
	{
		const u64 login_unix_timestamp =
			StringUtil::FromChars<u64>(Host::GetBaseStringSettingValue("Achievements", "LoginTimestamp", "0")).value_or(0);
		const QDateTime login_timestamp(QDateTime::fromSecsSinceEpoch(static_cast<qint64>(login_unix_timestamp)));
		m_ui.loginStatus->setText(tr("Username: %1\nLogin token generated on %2.")
									  .arg(QString::fromStdString(username))
									  .arg(login_timestamp.toString(Qt::TextDate)));
		m_ui.loginButton->setText(tr("Logout"));
	}
	else
	{
		m_ui.loginStatus->setText(tr("Not Logged In."));
		m_ui.loginButton->setText(tr("Login..."));
	}

	m_ui.viewProfile->setEnabled(logged_in);
}

void AchievementSettingsWidget::onLoginLogoutPressed()
{
	if (!Host::GetBaseStringSettingValue("Achievements", "Username").empty())
	{
		Host::RunOnCPUThread([]() { Achievements::Logout(); }, true);
		updateLoginState();
		return;
	}

	AchievementLoginDialog login(this, Achievements::LoginRequestReason::UserInitiated);
	int res = login.exec();
	if (res != 0)
		return;

	updateLoginState();
}

void AchievementSettingsWidget::onViewProfilePressed()
{
	const std::string username(Host::GetBaseStringSettingValue("Achievements", "Username"));
	if (username.empty())
		return;

	const QByteArray encoded_username(QUrl::toPercentEncoding(QString::fromStdString(username)));
	QtUtils::OpenURL(
		QtUtils::GetRootWidget(this),
		QUrl(QStringLiteral("https://retroachievements.org/user/%1").arg(QString::fromUtf8(encoded_username))));
}

void AchievementSettingsWidget::onAchievementsRefreshed(quint32 id, const QString& game_info_string)
{
	m_ui.gameInfo->setText(game_info_string);
}
