#include "GitQlientStyles.h"

#include <Colors.h>
#include <GitQlientSettings.h>

#include <QFile>

GitQlientStyles *GitQlientStyles::INSTANCE = nullptr;

GitQlientStyles *GitQlientStyles::getInstance()
{
   if (INSTANCE == nullptr)
      INSTANCE = new GitQlientStyles();

   return INSTANCE;
}

QString GitQlientStyles::getStyles()
{
   QString styles;
   QFile stylesFile(":/stylesheet");

   if (stylesFile.open(QIODevice::ReadOnly))
   {
      const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();
      QFile colorsFile(QString(":/colors_%1").arg(QString::fromUtf8(colorSchema ? "bright" : "dark")));
      QString colorsCss;

      if (colorsFile.open(QIODevice::ReadOnly))
      {
         colorsCss = QString::fromUtf8(colorsFile.readAll());
         colorsFile.close();
      }

      styles = stylesFile.readAll() + colorsCss;

      stylesFile.close();
   }

   return styles;
}

QColor GitQlientStyles::getTextColor()
{
   const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorSchema == 1 ? textColorBright : textColorDark;
}

QColor GitQlientStyles::getGraphSelectionColor()
{
   const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorSchema == 0 ? graphSelectionColorDark : graphSelectionColorBright;
}

QColor GitQlientStyles::getGraphHoverColor()
{
   const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorSchema == 0 ? graphHoverColorDark : graphHoverColorBright;
}

QColor GitQlientStyles::getBackgroundColor()
{
   const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorSchema == 0 ? graphBackgroundColorDark : graphBackgroundColorBright;
}

QColor GitQlientStyles::getTabColor()
{
   const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorSchema == 0 ? graphHoverColorDark : graphBackgroundColorBright;
}

QColor GitQlientStyles::getBlue()
{
   const auto colorSchema = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorSchema == 0 ? graphBlueDark : graphBlueBright;
}

QColor GitQlientStyles::getRed()
{
   return graphRed;
}

QColor GitQlientStyles::getGreen()
{
   return graphGreen;
}

QColor GitQlientStyles::getOrange()
{
   return graphOrange;
}

QColor GitQlientStyles::getShadowedRed()
{
   const auto colorScheme = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorScheme == 0 ? editorRedShadowDark : editorRedShadowBright;
}

QColor GitQlientStyles::getShadowedGreen()
{
   const auto colorScheme = GitQlientSettings().globalValue("colorSchema", 0).toInt();

   return colorScheme == 0 ? editorGreenShadowDark : editorGreenShadowBright;
}

std::array<QColor, GitQlientStyles::kBranchColors> GitQlientStyles::getBranchColors()
{
   static std::array<QColor, kBranchColors> colors { { getTextColor(), graphRed, getBlue(), graphGreen, graphOrange,
                                                       graphAubergine, graphCoral, graphGrey, graphTurquoise, graphPink,
                                                       graphPastel } };

   return colors;
}

QColor GitQlientStyles::getBranchColorAt(int index)
{
   if (index < kBranchColors && index >= 0)
      return getBranchColors().at(static_cast<size_t>(index));

   return QColor();
}
