#include "fixedscrollarea.h"
//#include <QWheelEvent>
#include <QScrollBar>

void FixedScrollArea::resizeEvent(QResizeEvent *event) {
    if (widget()) {
        widget()->setFixedWidth(width() - (verticalScrollBar()->isVisible() ? (verticalScrollBar()->width() - 2) : 0));
    }
    QScrollArea::resizeEvent(event);
}

//void FixedScrollArea::wheelEvent(QWheelEvent *event) {
//    if (event->modifiers() & Qt::ShiftModifier) {
//        emit ScaleTriggered(event->delta());
//        return;
//    }
//    QScrollArea::wheelEvent(event);
//}
