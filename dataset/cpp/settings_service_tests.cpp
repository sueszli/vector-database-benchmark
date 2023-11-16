#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <QFile>
#include <QString>
#include "setting_groups.hpp"
#include "setting_keys.hpp"
#include "settings_service.hpp"

using namespace testing;
using namespace application::services;

namespace tests::application
{

struct ASettingsService : public ::testing::Test
{
    void SetUp() override
    {
        settingsService = std::make_unique<SettingsService>();
        settingsService->loadUserSettings("", testingEmail);
    }

    void TearDown() override
    {
        QFile::remove(settingsService->getSettingsFilePath());
        settingsService->clearUserData();
    }

    std::unique_ptr<SettingsService> settingsService;

private:
    QString testingEmail = "LibrumTestEmail@librum.fake";
};

TEST_F(ASettingsService, SucceedsSettingASetting)
{
    // Arrange
    auto key = SettingKeys::PageSpacing;
    QVariant value = 20;
    auto group = SettingGroups::Appearance;


    // Act
    settingsService->setSetting(key, value, group);
}

TEST_F(ASettingsService, SucceedsOverridingASetting)
{
    // Arrange
    auto key = SettingKeys::PageSpacing;
    QVariant initialValue = 20;
    QVariant newValue = 50;
    auto group = SettingGroups::Appearance;

    settingsService->setSetting(key, initialValue, group);


    // Act
    settingsService->setSetting(key, newValue, group);

    // Assert
    auto actualValue = settingsService->getSetting(key, group);
    EXPECT_EQ(newValue, actualValue);
}

TEST_F(ASettingsService, SucceedsGettingAnIntegerSetting)
{
    // Arrange
    auto key = SettingKeys::PageSpacing;
    QVariant value = 20;
    auto group = SettingGroups::Appearance;

    settingsService->setSetting(key, value, group);


    // Act
    auto result = settingsService->getSetting(key, group);

    // Assert
    EXPECT_EQ(value, result.toInt());
}

TEST_F(ASettingsService, SucceedsGettingAStringSetting)
{
    // Arrange
    auto key = SettingKeys::PageSpacing;
    QVariant value = QString("SomeString");
    auto group = SettingGroups::Appearance;

    settingsService->setSetting(key, value, group);


    // Act
    auto result = settingsService->getSetting(key, group);

    // Assert
    EXPECT_EQ(value, result);
}

}  // namespace tests::application