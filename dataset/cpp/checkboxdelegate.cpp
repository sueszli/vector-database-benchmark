/*
Copyright 2011 Simone Tobia

This file is part of AppSet.

AppSet is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

AppSet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AppSet; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/
#include "checkboxdelegate.h"

#include <QApplication>
#include <QCheckBox>

CheckBoxDelegate::CheckBoxDelegate( QObject *parent )
    : QStyledItemDelegate(parent)
{
}

void CheckBoxDelegate::paint( QPainter *painter,
                              const QStyleOptionViewItem &option,
                              const QModelIndex &index ) const
{
    bool value = index.data().toBool();
    QStyleOptionButton cbo;

    cbo.rect = option.rect;
    cbo.rect.setLeft( option.rect.center().x() - 10 );
    cbo.state = value ? QStyle::State_On : QStyle::State_Off;
    cbo.text = QLatin1String("");
    cbo.state = cbo.state | QStyle::State_Enabled;


    qApp->style()->drawControl( QStyle::CE_CheckBox,&cbo, painter );
}

bool CheckBoxDelegate::editorEvent( QEvent *event,
                                    QAbstractItemModel *model,
                                    const QStyleOptionViewItem &option,
                                    const QModelIndex &index )
{
    Q_UNUSED(option);

    if( event->type() == QEvent::MouseButtonRelease )
        return false;

    model->setData( index, !index.data().toBool() );

    return event->type() == QEvent::MouseButtonDblClick;
}
