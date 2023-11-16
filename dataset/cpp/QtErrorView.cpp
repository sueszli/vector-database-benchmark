#include "QtErrorView.h"

#include <QBoxLayout>
#include <QCheckBox>
#include <QFrame>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QLineEdit>
#include <QPalette>
#include <QPushButton>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStyledItemDelegate>

#include "ColorScheme.h"
#include "MessageProjectEdit.h"
#include "QtHelpButton.h"
#include "QtSelfRefreshIconButton.h"
#include "QtTable.h"
#include "QtViewWidgetWrapper.h"
#include "ResourcePaths.h"
#include "TabId.h"
#include "utilityQt.h"

QIcon QtErrorView::s_errorIcon;

QtErrorView::QtErrorView(ViewLayout* viewLayout)
	: ErrorView(viewLayout), m_controllerProxy(this, TabId::app())
{
	s_errorIcon = QIcon(QString::fromStdWString(
		ResourcePaths::getGuiDirectoryPath().concatenate(L"indexing_dialog/error.png").wstr()));

	setWidgetWrapper(std::make_shared<QtViewWidgetWrapper>(new QFrame()));

	QWidget* widget = QtViewWidgetWrapper::getWidgetOfView(this);

	QBoxLayout* layout = new QVBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);
	widget->setLayout(layout);

	m_table = new QtTable(widget);
	m_model = new QStandardItemModel(widget);
	m_table->setSortingEnabled(true);
	m_table->setModel(m_model);

	// Setup Table Headers
	m_model->setColumnCount(COLUMN_MAX + 1);
	m_table->setColumnWidth(Column::ID, 40);
	m_table->setColumnWidth(Column::TYPE, 80);
	m_table->setColumnWidth(Column::MESSAGE, 450);
	m_table->setColumnWidth(Column::FILE, 300);
	m_table->setColumnWidth(Column::LINE, 50);
	m_table->setColumnWidth(Column::TRANSLATION_UNIT, 300);

	m_table->setColumnHidden(Column::ID, true);

	QStringList headers;
	headers << QStringLiteral("ID") << QStringLiteral("Type") << QStringLiteral("Message")
			<< QStringLiteral("File") << QStringLiteral("Line") << QStringLiteral("Indexed")
			<< QStringLiteral("Translation Unit");
	m_model->setHorizontalHeaderLabels(headers);

	connect(m_table, &QTableView::clicked, [=](const QModelIndex& index) {
		if (index.isValid())
		{
			if (m_model->item(index.row(), Column::FILE) == nullptr)
			{
				return;
			}

			const Id errorId = static_cast<Id>(
				m_model->item(index.row(), Column::ID)->text().toLongLong());

			m_controllerProxy.executeAsTaskWithArgs(&ErrorController::showError, errorId);
		}
	});

	layout->addWidget(m_table);

	// Setup Checkboxes
	QBoxLayout* checkboxes = new QHBoxLayout();
	checkboxes->setContentsMargins(10, 3, 0, 3);
	checkboxes->setSpacing(0);

	{
		m_showFatals = createFilterCheckbox(
			QStringLiteral("Fatals"), m_errorFilter.fatal, checkboxes);
		m_showErrors = createFilterCheckbox(
			QStringLiteral("Errors"), m_errorFilter.error, checkboxes);
		m_showNonIndexedFatals = createFilterCheckbox(
			QStringLiteral("Fatals in non-indexed files"), m_errorFilter.unindexedFatal, checkboxes);
		m_showNonIndexedErrors = createFilterCheckbox(
			QStringLiteral("Errors in non-indexed files"), m_errorFilter.unindexedError, checkboxes);

		m_helpButton = new QtHelpButton(createErrorHelpButtonInfo());
		m_helpButton->setObjectName(QStringLiteral("help_button"));
		checkboxes->addWidget(m_helpButton);
	}

	checkboxes->addStretch();

	{
		m_allLabel = new QLabel(QLatin1String(""));
		checkboxes->addWidget(m_allLabel);
		m_allLabel->hide();

		checkboxes->addSpacing(5);

		m_allButton = new QPushButton(QLatin1String(""));
		m_allButton->setObjectName(QStringLiteral("screen_button"));
		connect(m_allButton, &QPushButton::clicked, [=]() {
			m_errorFilter.limit = 0;
			errorFilterChanged();
		});
		checkboxes->addWidget(m_allButton);
		m_allButton->hide();

		m_errorLabel = new QLabel(QLatin1String(""));
		checkboxes->addWidget(m_errorLabel);
	}

	checkboxes->addSpacing(10);

	{
		m_editButton = new QtSelfRefreshIconButton(
			QStringLiteral("Edit Project"),
			ResourcePaths::getGuiDirectoryPath().concatenate(L"code_view/images/edit.png"),
			"window/button");
		m_editButton->setObjectName(QStringLiteral("screen_button"));
		m_editButton->setToolTip(QStringLiteral("edit project"));
		connect(m_editButton, &QPushButton::clicked, []() { MessageProjectEdit().dispatch(); });

		checkboxes->addWidget(m_editButton);
	}

	checkboxes->addSpacing(10);

	layout->addLayout(checkboxes);
}

void QtErrorView::createWidgetWrapper() {}

void QtErrorView::refreshView()
{
	m_onQtThread([=]() { setStyleSheet(); });
}

void QtErrorView::clear()
{
	m_onQtThread([=]() {
		if (!m_model->index(0, 0).data(Qt::DisplayRole).toString().isEmpty())
		{
			m_model->removeRows(0, m_model->rowCount());
		}

		m_table->updateRows();

		m_allLabel->setVisible(false);
		m_allButton->setVisible(false);
		m_errorLabel->setVisible(false);
	});
}

void QtErrorView::addErrors(
	const std::vector<ErrorInfo>& errors, const ErrorCountInfo& errorCount, bool scrollTo)
{
	m_onQtThread([=]() {
		for (const ErrorInfo& error: errors)
		{
			addErrorToTable(error);
		}
		m_table->updateRows();

		if (scrollTo)
		{
			m_table->showLastRow();
		}
		else
		{
			m_table->showFirstRow();
		}

		bool limited = m_errorFilter.limit > 0 && errorCount.total > m_errorFilter.limit;

		m_allLabel->setVisible(limited);
		m_allLabel->setText(
			"<b>Only displaying first " + QString::number(m_errorFilter.limit) + " errors</b>");

		m_allButton->setVisible(limited);
		m_allButton->setText("Show all " + QString::number(errorCount.total));

		m_errorLabel->setVisible(!limited);
		m_errorLabel->setText(
			"<b>displaying " + QString::number(errorCount.total) + " error" +
			(errorCount.total != 1 ? "s" : "") +
			(errorCount.fatal > 0 ? " (" + QString::number(errorCount.fatal) + " fatal)"
								  : QLatin1String("")) +
			"</b>");
	});
}

void QtErrorView::setErrorId(Id errorId)
{
	m_onQtThread([=]() {
		QList<QStandardItem*> items = m_model->findItems(
			QString::number(errorId), Qt::MatchExactly, Column::ID);

		if (items.size() == 1)
		{
			m_table->selectRow(items.at(0)->row());
		}
	});
}

ErrorFilter QtErrorView::getErrorFilter() const
{
	return m_errorFilter;
}

void QtErrorView::setErrorFilter(const ErrorFilter& filter)
{
	if (m_errorFilter == filter)
	{
		return;
	}

	m_errorFilter = filter;

	m_onQtThread([=]() {
		m_showErrors->blockSignals(true);
		m_showFatals->blockSignals(true);
		m_showNonIndexedErrors->blockSignals(true);
		m_showNonIndexedFatals->blockSignals(true);

		m_showErrors->setChecked(m_errorFilter.error);
		m_showFatals->setChecked(m_errorFilter.fatal);
		m_showNonIndexedErrors->setChecked(m_errorFilter.unindexedError);
		m_showNonIndexedFatals->setChecked(m_errorFilter.unindexedFatal);

		m_showErrors->blockSignals(false);
		m_showFatals->blockSignals(false);
		m_showNonIndexedErrors->blockSignals(false);
		m_showNonIndexedFatals->blockSignals(false);
	});
}

void QtErrorView::errorFilterChanged(int i)
{
	Q_UNUSED(i)
	m_table->selectionModel()->clearSelection();

	m_errorFilter.error = m_showErrors->isChecked();
	m_errorFilter.fatal = m_showFatals->isChecked();
	m_errorFilter.unindexedError = m_showNonIndexedErrors->isChecked();
	m_errorFilter.unindexedFatal = m_showNonIndexedFatals->isChecked();

	m_controllerProxy.executeAsTaskWithArgs(&ErrorController::errorFilterChanged, m_errorFilter);
}

void QtErrorView::setStyleSheet() const
{
	QWidget* widget = QtViewWidgetWrapper::getWidgetOfView(this);
	utility::setWidgetBackgroundColor(
		widget, ColorScheme::getInstance()->getColor("window/background"));

	QPalette palette(m_showErrors->palette());
	palette.setColor(
		QPalette::WindowText,
		QColor(ColorScheme::getInstance()->getColor("table/text/normal").c_str()));

	m_showErrors->setPalette(palette);
	m_showFatals->setPalette(palette);
	m_showNonIndexedErrors->setPalette(palette);
	m_showNonIndexedFatals->setPalette(palette);

	m_helpButton->setColor(QColor(ColorScheme::getInstance()->getColor("table/text/normal").c_str()));

	m_table->updateRows();
}

void QtErrorView::addErrorToTable(const ErrorInfo& error)
{
	if (!isShownError(error))
	{
		return;
	}

	int rowNumber = m_table->getFilledRowCount();
	if (rowNumber < m_model->rowCount())
	{
		m_model->insertRow(rowNumber);
	}

	QStandardItem* item = new QStandardItem();
	item->setData(QVariant(qlonglong(error.id)), Qt::DisplayRole);
	m_model->setItem(rowNumber, Column::ID, item);

	m_model->setItem(
		rowNumber,
		Column::TYPE,
		new QStandardItem(error.fatal ? QStringLiteral("FATAL") : QStringLiteral("ERROR")));
	if (error.fatal)
	{
		m_model->item(rowNumber, Column::TYPE)->setForeground(QBrush(Qt::red));
	}
	m_model->item(rowNumber, Column::TYPE)->setIcon(s_errorIcon);

	m_model->setItem(
		rowNumber, Column::MESSAGE, new QStandardItem(QString::fromStdWString(error.message)));

	m_model->setItem(
		rowNumber, Column::FILE, new QStandardItem(QString::fromStdWString(error.filePath)));
	m_model->item(rowNumber, Column::FILE)->setToolTip(QString::fromStdWString(error.filePath));

	item = new QStandardItem();
	item->setData(QVariant(qlonglong(error.lineNumber)), Qt::DisplayRole);
	m_model->setItem(rowNumber, Column::LINE, item);

	m_model->setItem(
		rowNumber,
		Column::INDEXED,
		new QStandardItem(error.indexed ? QStringLiteral("yes") : QStringLiteral("no")));

	m_model->setItem(
		rowNumber,
		Column::TRANSLATION_UNIT,
		new QStandardItem(QString::fromStdWString(error.translationUnit)));
	m_model->item(rowNumber, Column::TRANSLATION_UNIT)
		->setToolTip(QString::fromStdWString(error.translationUnit));
}

QCheckBox* QtErrorView::createFilterCheckbox(const QString& name, bool checked, QBoxLayout* layout)
{
	QCheckBox* checkbox = new QCheckBox(name);
	checkbox->setChecked(checked);

	connect(
		checkbox,
		&QCheckBox::stateChanged,
		this,
		static_cast<void (QtErrorView::*)(int)>(&QtErrorView::errorFilterChanged));

	layout->addWidget(checkbox);
	layout->addSpacing(25);

	return checkbox;
}

bool QtErrorView::isShownError(const ErrorInfo& error)
{
	if (!error.fatal && error.indexed && m_showErrors->isChecked())
	{
		return true;
	}
	if (error.fatal && error.indexed && m_showFatals->isChecked())
	{
		return true;
	}
	if (!error.fatal && !error.indexed && m_showNonIndexedErrors->isChecked())
	{
		return true;
	}
	if (error.fatal && !error.indexed && m_showNonIndexedFatals->isChecked())
	{
		return true;
	}
	return false;
}
