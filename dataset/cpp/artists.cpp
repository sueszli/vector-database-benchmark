#include "view/search/artists.hpp"
#include "mainwindow.hpp"

Search::Artists::Artists(QWidget *parent)
	: QListWidget(parent)
{
	QListWidget::connect(this, &QListWidget::itemClicked,
		this, &Search::Artists::onItemClicked);
}

void Search::Artists::add(const lib::spt::artist &artist)
{
	auto name = QString::fromStdString(artist.name);
	auto id = QString::fromStdString(artist.id);

	auto *item = new QListWidgetItem(name, this);
	item->setData(static_cast<int>(DataRole::ArtistId), id);
	item->setToolTip(name);
}

void Search::Artists::onItemClicked(QListWidgetItem *item)
{
	auto *mainWindow = MainWindow::find(parentWidget());
	mainWindow->openArtist(item->data(static_cast<int>(DataRole::ArtistId))
		.toString().toStdString());
}
