/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */
#include <eventql/sql/extensions/chartsql/linechartbuilder.h>

namespace csql {

LineChartBuilder::LineChartBuilder(
    util::chart::Canvas* canvas,
    RefPtr<DrawStatementNode> draw_stmt) :
    ChartBuilder(canvas, draw_stmt) {}

util::chart::Drawable* LineChartBuilder::getChart() const {
  auto chart = dynamic_cast<util::chart::LineChart*>(findChartType());
  setLabels(chart);
  return chart;
}

util::chart::Drawable* LineChartBuilder::findChartType() const {
  preconditionCheck();

/*
  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::TimeType,
        SValue::TimeType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::TimeType,
        SValue::FloatType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::TimeType,
        SValue::StringType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::FloatType,
        SValue::StringType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::FloatType,
        SValue::FloatType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::FloatType,
        SValue::TimeType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::FloatType,
        SValue::StringType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::StringType,
        SValue::TimeType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::StringType,
        SValue::FloatType>>())
    return c;

  if (auto c = tryType2D<util::chart::LineChart2D<
        SValue::StringType,
        SValue::StringType>>())
    return c;
*/
  invalidType();
  return nullptr;
}

std::string LineChartBuilder::chartName() const {
  return "LineChart";
}

void LineChartBuilder::setLabels(util::chart::LineChart* chart) const {
  auto prop = draw_stmt_->getProperty(Token::T_LABELS);
  chart->setLabels(prop != nullptr);
}

}
