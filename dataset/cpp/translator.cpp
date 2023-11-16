#include "translator.h"
#include <QDirIterator>
#include <QGuiApplication>
#include <QSettings>

Translator::Translator(QQmlEngine *engine, QObject *parent)
    : QObject(parent), m_engine(engine) {
  m_translator = new QTranslator(this);
  m_dir = QDir(QGuiApplication::applicationDirPath(), "*" + extension,
               QDir::Name | QDir::IgnoreCase, QDir::Files);
  m_dir.cd("i18n");
  m_languages.clear();
  for (QString entry : m_dir.entryList()) {
    entry.remove(0, QGuiApplication::applicationName().length() + 1);
    entry.chop(extension.length());
    m_languages.append(entry);
  }
  emit languagesChanged();
  QSettings settings;
  QString lang =
      settings.value("Language/current", QLocale::system().bcp47Name())
          .toString();
  selectLanguage(lang);
}

QStringList Translator::languages() const { return m_languages; }

QString Translator::currentLanguage() const { return m_currentLanguage; }

QString Translator::languageByCode(const QString &code) {
  QLocale lo(code);
  return QLocale::languageToString(lo.language());
}

void Translator::selectLanguage(const QString &language) {
  qApp->removeTranslator(m_translator);
  if (m_languages.contains(language)) {
    QString file = QString("%1_%2%3")
                       .arg(QGuiApplication::applicationName())
                       .arg(language)
                       .arg(extension);
    if (m_translator->load(m_dir.absoluteFilePath(file))) {
      m_currentLanguage = language;
      QSettings settings;
      settings.setValue("Language/current", language);
      emit currentLanguageChanged();
    }
  }
  qApp->installTranslator(m_translator);
  m_engine->retranslate();
  emit languageChanged();
}
