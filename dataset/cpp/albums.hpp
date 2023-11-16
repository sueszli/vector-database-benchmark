#pragma once

#include "lib/spotify/api.hpp"
#include "lib/cache.hpp"
#include "view/search/searchtabtree.hpp"
#include "util/tooltip.hpp"

namespace Search
{
	class Albums: public SearchTabTree
	{
	Q_OBJECT

	public:
		Albums(lib::spt::api &spotify, lib::cache &cache, const lib::http_client &httpClient,
			lib::settings &settings, QWidget *parent);

		void add(const lib::spt::album &album);

	private:
		lib::spt::api &spotify;
		lib::cache &cache;
		const lib::http_client &httpClient;
		Tooltip tooltip;

		void onItemClicked(QTreeWidgetItem *item, int column);
		void onContextMenu(const QPoint &pos);
		void onItemEntered(QTreeWidgetItem *item, int column);
	};
}
