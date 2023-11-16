#pragma once

#include "lib/spotify/artist.hpp"

#include <QListWidget>

namespace Search
{
	class Artists: public QListWidget
	{
	Q_OBJECT

	public:
		explicit Artists(QWidget *parent);

		void add(const lib::spt::artist &artist);

	private:
		void onItemClicked(QListWidgetItem *item);
	};
}
