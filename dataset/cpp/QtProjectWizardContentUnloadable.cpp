#include "QtProjectWizardContentUnloadable.h"

#include "SourceGroupSettingsUnloadable.h"

QtProjectWizardContentUnloadable::QtProjectWizardContentUnloadable(
	std::shared_ptr<SourceGroupSettingsUnloadable> settings, QtProjectWizardWindow* window)
	: QtProjectWizardContent(window), m_settings(settings)
{
}

void QtProjectWizardContentUnloadable::populate(QGridLayout* layout, int& row)
{
	QHBoxLayout* layoutHorz = new QHBoxLayout();
	layout->addLayout(
		layoutHorz,
		row,
		QtProjectWizardWindow::FRONT_COL,
		1,
		1 + QtProjectWizardWindow::BACK_COL - QtProjectWizardWindow::FRONT_COL,
		Qt::AlignTop);

	layoutHorz->addSpacing(60);

	QLabel* infoLabel = new QLabel(QString::fromStdString(
		"<p>The type \"" + m_settings->getTypeString() +
		"\" of the selected Source Group is not supported by this version of Sourcetrail.</p>"));
	infoLabel->setObjectName(QStringLiteral("info"));
	infoLabel->setWordWrap(true);
	layoutHorz->addWidget(infoLabel);

	layoutHorz->addSpacing(40);

	row++;
}
