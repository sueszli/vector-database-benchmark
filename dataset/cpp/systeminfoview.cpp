#include "systeminfoview.hpp"
#include "mainwindow.hpp"

SystemInfoView::SystemInfoView(QWidget *parent)
	: QWidget(parent)
{
	auto *layout = new QVBoxLayout();
	setLayout(layout);

	auto *textInfo = new QTextEdit(this);
	textInfo->setHtml(systemInfo().to_html());
	textInfo->setReadOnly(true);
	layout->addWidget(textInfo);

	auto *infoLayout = new QHBoxLayout();

	auto *infoAbout = new QLabel(
		"This information could be useful when reporting bugs. "
		"Additional information, depending on the type of issue, may be more helpful.", this);
	infoAbout->setWordWrap(true);
	infoLayout->addWidget(infoAbout, 1);

	auto *copy = new QPushButton(QStringLiteral("Copy to clipboard"), this);
	QPushButton::connect(copy, &QPushButton::clicked,
		this, &SystemInfoView::copyToClipboard);
	infoLayout->addWidget(copy);

	layout->addLayout(infoLayout);
}

auto SystemInfoView::systemInfo() -> lib::qt::system_info
{
	SystemInfo info;

	// Device
	auto *mainWindow = MainWindow::find(parentWidget());
	if (mainWindow != nullptr)
	{
		auto device = mainWindow->playback().device;
		if (!device.name.empty() && !device.type.empty())
		{
			info.add(QStringLiteral("Device"),
				QString::fromStdString(lib::fmt::format("{} ({})",
					device.name, device.type)));
		}
	}

	return info;
}

void SystemInfoView::copyToClipboard(bool /*checked*/)
{
	QApplication::clipboard()->setText(systemInfo().to_text());
}
