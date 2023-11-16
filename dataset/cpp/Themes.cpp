/*  PCSX2 - PS2 Emulator for PCs
 *  Copyright (C) 2002-2023  PCSX2 Dev Team
 *
 *  PCSX2 is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU Lesser General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  PCSX2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with PCSX2.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

#include "PrecompiledHeader.h"
#include "QtHost.h"

#include "pcsx2/Config.h"
#include "pcsx2/Host.h"

#include "common/Path.h"

#include <QtGui/QPalette>
#include <QtWidgets/QApplication>
#include <QtWidgets/QStyle>
#include <QtWidgets/QStyleFactory>

namespace QtHost
{
	static void SetStyleFromSettings();
} // namespace QtHost

static QString s_unthemed_style_name;
static bool s_unthemed_style_name_set;

const char* QtHost::GetDefaultThemeName()
{
#ifdef __APPLE__
	return "";
#else
	return "darkfusion";
#endif
}

void QtHost::UpdateApplicationTheme()
{
	if (!s_unthemed_style_name_set)
	{
		s_unthemed_style_name_set = true;
		s_unthemed_style_name = QApplication::style()->objectName();
	}

	SetStyleFromSettings();
	SetIconThemeFromStyle();
}

void QtHost::SetStyleFromSettings()
{
	const std::string theme(Host::GetBaseStringSettingValue("UI", "Theme", GetDefaultThemeName()));

	if (theme == "fusion")
	{
		// setPalette() shouldn't be necessary, as the documentation claims that setStyle() resets the palette, but it
		// is here, to work around a bug in 6.4.x and 6.5.x where the palette doesn't restore after changing themes.
		qApp->setPalette(QPalette());
		qApp->setStyle(QStyleFactory::create("Fusion"));
		qApp->setStyleSheet(QString());
	}
	else if (theme == "darkfusion")
	{
		// adapted from https://gist.github.com/QuantumCD/6245215
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor lighterGray(75, 75, 75);
		const QColor darkGray(53, 53, 53);
		const QColor gray(128, 128, 128);
		const QColor black(25, 25, 25);
		const QColor blue(198, 238, 255);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, darkGray);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, black);
		darkPalette.setColor(QPalette::AlternateBase, darkGray);
		darkPalette.setColor(QPalette::ToolTipBase, darkGray);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, darkGray);
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, blue);
		darkPalette.setColor(QPalette::Highlight, lighterGray);
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);
		darkPalette.setColor(QPalette::PlaceholderText, QColor(Qt::white).darker());

		darkPalette.setColor(QPalette::Active, QPalette::Button, darkGray);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, darkGray);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "darkfusionblue")
	{
		// adapted from https://gist.github.com/QuantumCD/6245215
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor darkGray(53, 53, 53);
		const QColor gray(128, 128, 128);
		const QColor black(25, 25, 25);
		const QColor blue(198, 238, 255);
		const QColor blue2(0, 88, 208);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, darkGray);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, black);
		darkPalette.setColor(QPalette::AlternateBase, darkGray);
		darkPalette.setColor(QPalette::ToolTipBase, blue2);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, darkGray);
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, blue);
		darkPalette.setColor(QPalette::Highlight, blue2);
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);
		darkPalette.setColor(QPalette::PlaceholderText, QColor(Qt::white).darker());

		darkPalette.setColor(QPalette::Active, QPalette::Button, darkGray);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, darkGray);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "UntouchedLagoon")
	{
		// Custom palette by RedDevilus, Tame (Light/Washed out) Green as main color and Grayish Blue as complimentary.
		// Alternative white theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor black(25, 25, 25);
		const QColor darkteal(0, 77, 77);
		const QColor teal(0, 128, 128);
		const QColor tameTeal(160, 190, 185);
		const QColor grayBlue(160, 180, 190);

		QPalette standardPalette;
		standardPalette.setColor(QPalette::Window, tameTeal);
		standardPalette.setColor(QPalette::WindowText, black.lighter());
		standardPalette.setColor(QPalette::Base, grayBlue);
		standardPalette.setColor(QPalette::AlternateBase, tameTeal);
		standardPalette.setColor(QPalette::ToolTipBase, tameTeal);
		standardPalette.setColor(QPalette::ToolTipText, grayBlue);
		standardPalette.setColor(QPalette::Text, black);
		standardPalette.setColor(QPalette::Button, tameTeal);
		standardPalette.setColor(QPalette::ButtonText, black);
		standardPalette.setColor(QPalette::Link, black.lighter());
		standardPalette.setColor(QPalette::Highlight, teal);
		standardPalette.setColor(QPalette::HighlightedText, grayBlue.lighter());

		standardPalette.setColor(QPalette::Active, QPalette::Button, tameTeal);
		standardPalette.setColor(QPalette::Disabled, QPalette::ButtonText, darkteal);
		standardPalette.setColor(QPalette::Disabled, QPalette::WindowText, darkteal.lighter());
		standardPalette.setColor(QPalette::Disabled, QPalette::Text, darkteal.lighter());
		standardPalette.setColor(QPalette::Disabled, QPalette::Light, tameTeal);

		qApp->setPalette(standardPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "BabyPastel")
	{
		// Custom palette by RedDevilus, Blue as main color and blue as complimentary.
		// Alternative light theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor gray(150, 150, 150);
		const QColor black(25, 25, 25);
		const QColor redpinkish(200, 75, 132);
		const QColor pink(255, 174, 201);
		const QColor brightPink(255, 230, 255);
		const QColor congoPink(255, 127, 121);
		const QColor blue(221, 225, 239);

		QPalette standardPalette;
		standardPalette.setColor(QPalette::Window, pink);
		standardPalette.setColor(QPalette::WindowText, black);
		standardPalette.setColor(QPalette::Base, brightPink);
		standardPalette.setColor(QPalette::AlternateBase, blue);
		standardPalette.setColor(QPalette::ToolTipBase, pink);
		standardPalette.setColor(QPalette::ToolTipText, brightPink);
		standardPalette.setColor(QPalette::Text, black);
		standardPalette.setColor(QPalette::Button, pink);
		standardPalette.setColor(QPalette::ButtonText, black);
		standardPalette.setColor(QPalette::Link, black);
		standardPalette.setColor(QPalette::Highlight, congoPink);
		standardPalette.setColor(QPalette::HighlightedText, black);

		standardPalette.setColor(QPalette::Active, QPalette::Button, pink);
		standardPalette.setColor(QPalette::Disabled, QPalette::ButtonText, redpinkish);
		standardPalette.setColor(QPalette::Disabled, QPalette::WindowText, redpinkish);
		standardPalette.setColor(QPalette::Disabled, QPalette::Text, redpinkish);
		standardPalette.setColor(QPalette::Disabled, QPalette::Light, gray);

		qApp->setPalette(standardPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "PizzaBrown")
	{
		// Custom palette by KamFretoZ, a Pizza Tower Reference!
		// With a mixtures of Light Brown, Peachy/Creamy White, Latte-like Color.
		// Thanks to Jordan for the idea :P
		// Alternative light theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor gray(128, 128, 128);
		const QColor extr(248, 192, 88);
		const QColor main(233, 187, 147);
		const QColor comp(248, 230, 213);
		const QColor highlight(188, 100, 60);

		QPalette standardPalette;
		standardPalette.setColor(QPalette::Window, main);
		standardPalette.setColor(QPalette::WindowText, Qt::black);
		standardPalette.setColor(QPalette::Base, comp);
		standardPalette.setColor(QPalette::AlternateBase, extr);
		standardPalette.setColor(QPalette::ToolTipBase, comp);
		standardPalette.setColor(QPalette::ToolTipText, Qt::black);
		standardPalette.setColor(QPalette::Text, Qt::black);
		standardPalette.setColor(QPalette::Button, extr);
		standardPalette.setColor(QPalette::ButtonText, Qt::black);
		standardPalette.setColor(QPalette::Link, highlight.darker());
		standardPalette.setColor(QPalette::Highlight, highlight);
		standardPalette.setColor(QPalette::HighlightedText, Qt::white);
		standardPalette.setColor(QPalette::Active, QPalette::Button, extr);
		standardPalette.setColor(QPalette::Disabled, QPalette::ButtonText, gray.darker());
		standardPalette.setColor(QPalette::Disabled, QPalette::WindowText, gray.darker());
		standardPalette.setColor(QPalette::Disabled, QPalette::Text, Qt::gray);
		standardPalette.setColor(QPalette::Disabled, QPalette::Light, gray.lighter());

		qApp->setPalette(standardPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #cc3f18; border: 1px solid white; }");
	}
	else if (theme == "PCSX2Blue")
	{
		// Custom palette by RedDevilus, White as main color and Blue as complimentary.
		// Alternative light theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor blackish(35, 35, 35);
		const QColor darkBlue(73, 97, 177);
		const QColor blue2(80, 120, 200);
		const QColor blue(106, 156, 255);
		const QColor lightBlue(130, 155, 241);

		QPalette standardPalette;
		standardPalette.setColor(QPalette::Window, blue2.lighter());
		standardPalette.setColor(QPalette::WindowText, blackish);
		standardPalette.setColor(QPalette::Base, lightBlue);
		standardPalette.setColor(QPalette::AlternateBase, blue2.lighter());
		standardPalette.setColor(QPalette::ToolTipBase, blue2);
		standardPalette.setColor(QPalette::ToolTipText, Qt::white);
		standardPalette.setColor(QPalette::Text, blackish);
		standardPalette.setColor(QPalette::Button, blue);
		standardPalette.setColor(QPalette::ButtonText, blackish);
		standardPalette.setColor(QPalette::Link, darkBlue);
		standardPalette.setColor(QPalette::Highlight, Qt::white);
		standardPalette.setColor(QPalette::HighlightedText, blackish);

		standardPalette.setColor(QPalette::Active, QPalette::Button, blue);
		standardPalette.setColor(QPalette::Disabled, QPalette::ButtonText, darkBlue);
		standardPalette.setColor(QPalette::Disabled, QPalette::WindowText, darkBlue);
		standardPalette.setColor(QPalette::Disabled, QPalette::Text, darkBlue);
		standardPalette.setColor(QPalette::Disabled, QPalette::Light, darkBlue);

		qApp->setPalette(standardPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "ScarletDevilRed")
	{
		// Custom palette by RedDevilus, Red as main color and Purple as complimentary.
		// Alternative dark theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor darkRed(80, 45, 69);
		const QColor purplishRed(120, 45, 69);
		const QColor brightRed(200, 45, 69);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, darkRed);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, purplishRed);
		darkPalette.setColor(QPalette::AlternateBase, darkRed);
		darkPalette.setColor(QPalette::ToolTipBase, darkRed);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, purplishRed.darker());
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, brightRed);
		darkPalette.setColor(QPalette::Highlight, brightRed);
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);

		darkPalette.setColor(QPalette::Active, QPalette::Button, purplishRed.darker());
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, brightRed);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, brightRed);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, brightRed);
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, darkRed);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "CobaltSky")
	{
		// Custom palette by KamFretoZ, A soothing deep royal blue
		// that are meant to be easy on the eyes as the main color.
		// Alternative dark theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor gray(150, 150, 150);
		const QColor royalBlue(29, 41, 81);
		const QColor darkishBlue(17, 30, 108);
		const QColor lighterBlue(25, 32, 130);
		const QColor highlight(36, 93, 218);
		const QColor link(0, 202, 255);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, royalBlue);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, royalBlue.lighter());
		darkPalette.setColor(QPalette::AlternateBase, darkishBlue);
		darkPalette.setColor(QPalette::ToolTipBase, darkishBlue);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, lighterBlue);
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, link);
		darkPalette.setColor(QPalette::Highlight, highlight);
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);

		darkPalette.setColor(QPalette::Active, QPalette::Button, lighterBlue);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, gray);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #000080; border: 1px solid white; }");
	}
	else if (theme == "VioletAngelPurple")
	{
		// Custom palette by RedDevilus, Blue as main color and Purple as complimentary.
		// Alternative dark theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor blackishblue(50, 25, 70);
		const QColor darkerPurple(90, 30, 105);
		const QColor nauticalPurple(110, 30, 125);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, blackishblue);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, nauticalPurple);
		darkPalette.setColor(QPalette::AlternateBase, blackishblue);
		darkPalette.setColor(QPalette::ToolTipBase, nauticalPurple);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, nauticalPurple.darker());
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, darkerPurple.lighter());
		darkPalette.setColor(QPalette::Highlight, darkerPurple.lighter());
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);

		darkPalette.setColor(QPalette::Active, QPalette::Button, nauticalPurple.darker());
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, darkerPurple.lighter());
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, darkerPurple.lighter());
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, darkerPurple.darker());
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, nauticalPurple);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "Ruby")
	{
		// Custom palette by Daisouji, Black as main color and Red as complimentary.
		// Alternative dark (black) theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor gray(128, 128, 128);
		const QColor slate(18, 18, 18);
		const QColor rubyish(172, 21, 31);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, slate);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, slate.lighter());
		darkPalette.setColor(QPalette::AlternateBase, slate.lighter());
		darkPalette.setColor(QPalette::ToolTipBase, slate);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, slate);
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, Qt::white);
		darkPalette.setColor(QPalette::Highlight, rubyish);
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);

		darkPalette.setColor(QPalette::Active, QPalette::Button, slate);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, slate.lighter());

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "Sapphire")
	{
		// Custom palette by RedDevilus, Black as main color and Blue as complimentary.
		// Alternative dark (black) theme.
		qApp->setStyle(QStyleFactory::create("Fusion"));

		const QColor gray(128, 128, 128);
		const QColor slate(18, 18, 18);
		const QColor persianBlue(32, 35, 204);

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, slate);
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, slate.lighter());
		darkPalette.setColor(QPalette::AlternateBase, slate.lighter());
		darkPalette.setColor(QPalette::ToolTipBase, slate);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, slate);
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::Link, Qt::white);
		darkPalette.setColor(QPalette::Highlight, persianBlue);
		darkPalette.setColor(QPalette::HighlightedText, Qt::white);

		darkPalette.setColor(QPalette::Active, QPalette::Button, slate);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, gray);
		darkPalette.setColor(QPalette::Disabled, QPalette::Light, slate.lighter());

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}
	else if (theme == "Custom")
	{
		//Additional Theme option than loads .qss from main PCSX2 Directory
		qApp->setStyle(QStyleFactory::create("Fusion"));

		QString sheet_content;
		QFile sheets(QString::fromStdString(Path::Combine(EmuFolders::DataRoot, "custom.qss")));

		if (sheets.open(QFile::ReadOnly))
		{
			QString sheet_content = QString::fromUtf8(sheets.readAll().data());
			qApp->setStyleSheet(sheet_content);
		}
		else
		{
			qApp->setStyle(QStyleFactory::create("Fusion"));
		}
	}
	else
	{
		// setPalette() shouldn't be necessary, as the documentation claims that setStyle() resets the palette, but it
		// is here, to work around a bug in 6.4.x and 6.5.x where the palette doesn't restore after changing themes.
		qApp->setPalette(QPalette());
		qApp->setStyle(s_unthemed_style_name);
		qApp->setStyleSheet(QString());
	}
}

void QtHost::SetIconThemeFromStyle()
{
	QPalette palette = qApp->palette();
	bool dark = palette.windowText().color().value() > palette.window().color().value();
	QIcon::setThemeName(dark ? QStringLiteral("white") : QStringLiteral("black"));
}
