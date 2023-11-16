#pragma once

#include "lib/spotify/track.hpp"
#include "menu/track.hpp"

#include <QLabel>
#include <QVBoxLayout>

namespace Context
{
	class NowPlaying: public QWidget
	{
	Q_OBJECT

	public:
		explicit NowPlaying(QWidget *parent);

		void setTrack(const lib::spt::track &track);
		void setNoPlaying();

		auto getTextShadow() const -> bool;
		void setTextShadow(bool value);

	private:
		static constexpr float nameScale = 1.1F;
		static constexpr float artistScale = 0.9F;

		QVBoxLayout *layout = nullptr;

		QLabel *artist = nullptr;
		QLabel *name = nullptr;

		auto newLabel(float scale) -> QLabel *;
		static void setText(QLabel *label, const std::string &text);

		static void addTextShadow(QLabel *label);
		static void removeTextShadow(QLabel *label);

		void onMenu(const QPoint &pos);
	};
}
