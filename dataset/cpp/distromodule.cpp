/*
 * Copyright (C) 2018-2023 QuasarApp.
 * Distributed under the lgplv3 software license, see the accompanying
 * Everyone is permitted to copy and distribute verbatim copies
 * of this license document, but changing it is not allowed.
 */

#include "deployconfig.h"
#include "deploycore.h"
#include "distromodule.h"

DistroModule::DistroModule(const QString& key) {
    setKey(key);
}

const QSet<QString>& DistroModule::targets() const {
    return _targets;
}

void DistroModule::setTargets(const QSet<QString> &targets) {
    _targets = targets;
}

void DistroModule::addTarget(const QString &target) {
    _targets.insert(target);
}

QSet<QString> DistroModule::qmlInput() const {
    return _qmlInput;
}

void DistroModule::setQmlInput(const QSet<QString> &qmlInput) {
    _qmlInput = qmlInput;
}

void DistroModule::addQmlInput(const QString &target) {
    _qmlInput.insert(DeployCore::transportPathToSnapRoot(target));
}

QString DistroModule::name() const {
    return _name;
}

void DistroModule::setName(const QString &name) {
    _name = name;
}

QString DistroModule::description() const {
    return _description;
}

void DistroModule::setDescription(const QString &description) {
    _description = description;
}

QString DistroModule::version() const {
    return _version;
}

void DistroModule::setVersion(const QString &version) {
    _version = version;
}

QString DistroModule::releaseData() const {
    return _releaseData;
}

void DistroModule::setReleaseData(const QString &releaseData) {
    _releaseData = releaseData;
}

QString DistroModule::publisher() const {
    return _publisher;
}

void DistroModule::setPublisher(const QString &publisher) {
    _publisher = publisher;
}

const QSet<QString>& DistroModule::enabledPlugins() const {
    return _enabled;
}

void DistroModule::setEnabledPlugins(const QSet<QString> &enabled) {
    _enabled = enabled;
}

void DistroModule::addEnabledPlugins(const QString &enabled) {
    _enabled += enabled;
}

const QSet<QString>& DistroModule::disabledPlugins() const {
    return _disabled;
}

void DistroModule::setDisabledPlugins(const QSet<QString> &disabled) {
    _disabled = disabled;
}

void DistroModule::addDisabledPlugins(const QString &disabled) {
    _disabled += disabled;
}

const QSet<QString> &DistroModule::extraPlugins() const {
    return _extraPlugins;
}

void DistroModule::setExtraPlugins(const QSet<QString> &extraPlugins) {
    _extraPlugins = extraPlugins;
}

void DistroModule::addExtraPlugins(const QString &extraPlugin) {
    _extraPlugins += DeployCore::transportPathToSnapRoot(extraPlugin);

}

QString DistroModule::homePage() const {
    return _homePage;
}

void DistroModule::setHomePage(const QString &homePage) {
    _homePage = homePage;
}

QString DistroModule::prefix() const {
    return _prefix;
}

void DistroModule::setPrefix(const QString &location) {
    _prefix = location;
}

QString DistroModule::key() const {
    return _key;
}

bool DistroModule::isDefaultModule() const {
    const DeployConfig *cfg = DeployCore::_config;

    return key() == cfg->getDefaultPackage();
}

bool DistroModule::isValid() {
    return _key.size();
}

void DistroModule::setKey(const QString &key) {
    _key = key;
}

const QString &DistroModule::zipOut() const {
    return _zipOut;
}

void DistroModule::setZipOut(const QString &newZipOut) {
    _zipOut = newZipOut;
}

const QString &DistroModule::debOut() const {
    return _debOut;
}

void DistroModule::setDebOut(const QString &newDebOut) {
    _debOut = newDebOut;
}

QString DistroModule::installDirDEB() const {
    if (_installDirDEB.isEmpty())
        return "/opt";

    return _installDirDEB;
}

void DistroModule::setInstallDirDEB(const QString &newInstallDir) {
    _installDirDEB = newInstallDir;
}

const QSet<QString> &DistroModule::translation() const {
    return _tr;
}

void DistroModule::setTranslation(const QSet<QString> &tr) {
    _tr = tr;
}

void DistroModule::addTranslation(const QString &tr) {
    _tr += tr;
}

const QSet<QString> &DistroModule::extraData() const {
    return _extraData;
}

void DistroModule::setExtraData(const QSet<QString> &extraFiles) {
    _extraData = extraFiles;
}

void DistroModule::addExtraData(const QString &extraFile) {
    _extraData += extraFile;
}

