#include "SeerWatchpointsBrowserWidget.h"
#include "SeerWatchpointCreateDialog.h"
#include "SeerUtl.h"
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItemIterator>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QInputDialog>
#include <QtGui/QFontDatabase>
#include <QtCore/QFileInfo>
#include <QtCore/QDebug>

SeerWatchpointsBrowserWidget::SeerWatchpointsBrowserWidget (QWidget* parent) : QWidget(parent) {

    // Construct the UI.
    setupUi(this);

    // Setup the widgets
    watchpointsTreeWidget->clear();

    watchpointsTreeWidget->setSortingEnabled(false);
    watchpointsTreeWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    watchpointsTreeWidget->resizeColumnToContents(0); // number
    watchpointsTreeWidget->resizeColumnToContents(1); // original-location / expression
    watchpointsTreeWidget->resizeColumnToContents(2); // value
    watchpointsTreeWidget->resizeColumnToContents(3); // new value
    watchpointsTreeWidget->resizeColumnToContents(4); // type
    watchpointsTreeWidget->resizeColumnToContents(5); // disp
    watchpointsTreeWidget->resizeColumnToContents(6); // enabled
    watchpointsTreeWidget->resizeColumnToContents(7); // addr
  //watchpointsTreeWidget->resizeColumnToContents(8); // func      Too long to show
    watchpointsTreeWidget->resizeColumnToContents(9); // file
  //watchpointsTreeWidget->resizeColumnToContents(10); // fullname  Too long to show
    watchpointsTreeWidget->resizeColumnToContents(11); // line
    watchpointsTreeWidget->resizeColumnToContents(12); // thread-groups
    watchpointsTreeWidget->resizeColumnToContents(13); // cond
    watchpointsTreeWidget->resizeColumnToContents(14); // times
    watchpointsTreeWidget->resizeColumnToContents(15); // ignore
  //watchpointsTreeWidget->resizeColumnToContents(16); // script   Too long to show
    watchpointsTreeWidget->resizeColumnToContents(17); // used

    watchpointsTreeWidget->setColumnHidden(7, true); // Hide the 'addr' column.
    watchpointsTreeWidget->setColumnHidden(8, true); // Hide the 'func' column.
    watchpointsTreeWidget->setColumnHidden(17, true); // Hide the 'used' column.
    watchpointsTreeWidget->clear();

    // Connect things.
    QObject::connect(watchpointsTreeWidget,         &QTreeWidget::itemDoubleClicked,    this,  &SeerWatchpointsBrowserWidget::handleItemDoubleClicked);
    QObject::connect(refreshWatchpointsToolButton,  &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleRefreshToolButton);
    QObject::connect(addWatchpointToolButton,       &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleAddToolButton);
    QObject::connect(deleteWatchpointsToolButton,   &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleDeleteToolButton);
    QObject::connect(enableWatchpointsToolButton,   &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleEnableToolButton);
    QObject::connect(disableWatchpointsToolButton,  &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleDisableToolButton);
    QObject::connect(conditionWatchpointToolButton, &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleConditionToolButton);
    QObject::connect(ignoreWatchpointToolButton,    &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleIgnoreToolButton);
    QObject::connect(commandsWatchpointToolButton,  &QToolButton::clicked,              this,  &SeerWatchpointsBrowserWidget::handleCommandsToolButton);
}

SeerWatchpointsBrowserWidget::~SeerWatchpointsBrowserWidget () {
}

bool SeerWatchpointsBrowserWidget::isEmpty() const {

    return (watchpointsTreeWidget->topLevelItemCount() == 0);
}

void SeerWatchpointsBrowserWidget::handleText (const QString& text) {

    // Don't do any work if the widget is hidden.
    if (isHidden()) {
        return;
    }

    QApplication::setOverrideCursor(Qt::BusyCursor);

    if (text.startsWith("^done,BreakpointTable={") && text.endsWith("}")) {

        //
        // ^done,BreakpointTable={
        //     nr_rows="2",nr_cols="6",
        //
        //     hdr=[
        //             {width="7",alignment="-1",col_name="number",colhdr="Num"},
        //             {width="14",alignment="-1",col_name="type",colhdr="Type"},
        //             {width="4",alignment="-1",col_name="disp",colhdr="Disp"},
        //             {width="3",alignment="-1",col_name="enabled",colhdr="Enb"},
        //             {width="18",alignment="-1",col_name="addr",colhdr="Address"},
        //             {width="40",alignment="2",col_name="what",colhdr="What"}
        //         ],
        //
        //     body=[
        //             bkpt={number="2",type="breakpoint",disp="keep",enabled="y",addr="0x0000000000400c17",func="main(int, char**)",file="helloworld.cpp",fullname="/home/erniep/Development/Peak/src/Seer/helloworld/helloworld.cpp",line="8",thread-groups=["i1"],times="0",original-location="main"},
        //             bkpt={number="3",
        //                   type="breakpoint",
        //                   disp="keep",
        //                   enabled="y",
        //                   addr="0x0000000000400d72",
        //                   func="function1(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)",
        //                   file="function1.cpp",
        //                   fullname="/home/erniep/Development/Peak/src/Seer/helloworld/function1.cpp",
        //                   line="7",
        //                   thread-groups=["i1"],
        //                   times="0",
        //                   original-location="function1"}
        //          ]
        // }
        //

        QString newtext = Seer::filterEscapes(text); // Filter escaped characters.

        QString body_text = Seer::parseFirst(newtext, "body=", '[', ']', false);

        //qDebug() << body_text;

        // No rows? Just clear the tree.
        if (body_text == "") {
            watchpointsTreeWidget->clear();

        // Otherwise, populate it.
        }else{

            // Mark each entry initially as "unused".
            // Later, some will be marked as "reused" or "new". Then the "unused" ones will
            // be deleted.
            QTreeWidgetItemIterator it(watchpointsTreeWidget);
            while (*it) {
                (*it)->setText(17, "unused");
                ++it;
            }

            QStringList bkpt_list = Seer::parse(newtext, "bkpt=", '{', '}', false);

            for ( const auto& bkpt_text : bkpt_list  ) {

                QString number_text            = Seer::parseFirst(bkpt_text, "number=",            '"', '"', false);
                QString type_text              = Seer::parseFirst(bkpt_text, "type=",              '"', '"', false);
                QString disp_text              = Seer::parseFirst(bkpt_text, "disp=",              '"', '"', false);
                QString enabled_text           = Seer::parseFirst(bkpt_text, "enabled=",           '"', '"', false);
                QString addr_text              = Seer::parseFirst(bkpt_text, "addr=",              '"', '"', false);
                QString func_text              = Seer::parseFirst(bkpt_text, "func=",              '"', '"', false);
                QString file_text              = Seer::parseFirst(bkpt_text, "file=",              '"', '"', false);
                QString fullname_text          = Seer::parseFirst(bkpt_text, "fullname=",          '"', '"', false);
                QString line_text              = Seer::parseFirst(bkpt_text, "line=",              '"', '"', false);
                QString thread_groups_text     = Seer::parseFirst(bkpt_text, "thread-groups=",     '[', ']', false);
                QString cond_text              = Seer::parseFirst(bkpt_text, "cond=",              '"', '"', false);
                QString times_text             = Seer::parseFirst(bkpt_text, "times=",             '"', '"', false);
                QString ignore_text            = Seer::parseFirst(bkpt_text, "ignore=",            '"', '"', false);
                QString script_text            = Seer::parseFirst(bkpt_text, "script=",            '{', '}', false);
                QString original_location_text = Seer::parseFirst(bkpt_text, "original-location=", '"', '"', false);

                // Only look for 'watchpoint' type break points.
                if (type_text != "hw watchpoint" && type_text != "watchpoint" && type_text != "read watchpoint" && type_text != "acc watchpoint") {
                    continue;
                }

                script_text = Seer::filterBookends(Seer::parseCommaList(script_text, '{', '}'), '"', '"').join('\n');

                // Instead of creating a new tree each time, we will reuse existing items, if they are there.
                // This allows the expanded items to remain expanded.
                QList<QTreeWidgetItem*> matches = watchpointsTreeWidget->findItems(number_text, Qt::MatchExactly, 0);

                // No matches. So can't reuse. Add the new entry.
                if (matches.size() == 0) {

                    // Add the level to the tree.
                    QTreeWidgetItem* topItem = new QTreeWidgetItem;
                    topItem->setText(0, number_text);
                    topItem->setText(1, original_location_text);
                    topItem->setText(2, "");
                    topItem->setText(3, "");
                    topItem->setText(4, type_text);
                    topItem->setText(5, disp_text);
                    topItem->setText(6, enabled_text);
                    topItem->setText(7, addr_text);
                    topItem->setText(8, func_text);
                    topItem->setText(9, QFileInfo(file_text).fileName());
                    topItem->setText(10, fullname_text);
                    topItem->setText(11, line_text);
                    topItem->setText(12, thread_groups_text);
                    topItem->setText(13, cond_text);
                    topItem->setText(14, times_text);
                    topItem->setText(15, ignore_text);
                    topItem->setText(16, script_text);
                    topItem->setText(17, "new");

                    topItem->setFont(2, QFontDatabase::systemFont(QFontDatabase::FixedFont));
                    topItem->setFont(3, QFontDatabase::systemFont(QFontDatabase::FixedFont));

                    for (int i=0; i<topItem->columnCount(); i++) {
                        topItem->setTextAlignment(i, Qt::AlignLeft|Qt::AlignTop);
                    }

                    watchpointsTreeWidget->addTopLevelItem(topItem);

                // Found a match. Reuse it.
                // But don't overwrite the file, fullname, line, value, and new value.
                }else{

                    QTreeWidgetItem* topItem = matches.takeFirst();

                    topItem->setText(0, number_text);
                    topItem->setText(1, original_location_text);
                  //topItem->setText(2, "");
                  //topItem->setText(3, "");
                    topItem->setText(4, type_text);
                    topItem->setText(5, disp_text);
                    topItem->setText(6, enabled_text);
                    topItem->setText(7, addr_text);
                    topItem->setText(8, func_text);
                  //topItem->setText(9, QFileInfo(file_text).fileName());
                  //topItem->setText(10, fullname_text);
                  //topItem->setText(11, line_text);
                    topItem->setText(12, thread_groups_text);
                    topItem->setText(13, cond_text);
                    topItem->setText(14, times_text);
                    topItem->setText(15, ignore_text);
                    topItem->setText(16, script_text);
                    topItem->setText(17, "reused");
                }
            }

            // At this point, there are some new entries, some reused entries, and some unused ones.
            // Delete the unused ones. They are obsolete.
            QList<QTreeWidgetItem*> matches = watchpointsTreeWidget->findItems("unused", Qt::MatchExactly, 17);

            qDeleteAll(matches);
        }

    }else if (text.startsWith("*stopped,reason=\"") || text.startsWith("*stopped,hw-awpt={")) {

        QString reason_text = Seer::parseFirst(text, "reason=", '"', '"', false);

        if (reason_text == "watchpoint-trigger") {
            //*stopped,reason="watchpoint-trigger",wpt={number="3",exp="i"},value={old="32767",new="42"},frame={addr="0x0000000000400d79",func="function1",args=[{name="text",value="\"Hello, World!\""}],file="function1.cpp",fullname="/home/erniep/Development/Peak/src/Seer/helloworld/function1.cpp",line="9",arch="i386:x86-64"},thread-id="1",stopped-threads="all",core="0"

            QString wpt_text       = Seer::parseFirst(text,       "wpt=",       '{', '}', false);
            QString number_text    = Seer::parseFirst(wpt_text,   "number=",    '"', '"', false);
            QString exp_text       = Seer::parseFirst(wpt_text,   "exp=",       '"', '"', false);
            QString value_text     = Seer::parseFirst(text,       "value=",     '{', '}', false);
            QString old_text       = Seer::parseFirst(value_text, "old=",       '"', '"', false);
            QString new_text       = Seer::parseFirst(value_text, "new=",       '"', '"', false);
            QString frame_text     = Seer::parseFirst(text,       "frame=",     '{', '}', false);
            QString file_text      = Seer::parseFirst(frame_text, "file=",      '"', '"', false);
            QString fullname_text  = Seer::parseFirst(frame_text, "fullname=",  '"', '"', false);
            QString line_text      = Seer::parseFirst(frame_text, "line=",      '"', '"', false);

            // Find watchpoint number in the tree
            QList<QTreeWidgetItem*> matches = watchpointsTreeWidget->findItems(number_text, Qt::MatchExactly, 0);
            if (matches.size() > 0) {
                //qDebug() << text;
                QTreeWidgetItem* item = matches.first();
                item->setText(2, old_text);
                item->setText(3, new_text);
                item->setText(9, QFileInfo(file_text).fileName());
                item->setText(10, fullname_text);
                item->setText(11, line_text);
            }

        }else if (reason_text == "read-watchpoint-trigger") {
            //*stopped,reason="read-watchpoint-trigger",hw-rwpt={number="5",exp="i"},value={value="42"},frame={addr="0x0000000000400d9a",func="function1",args=[{name="text",value="\"Hello, World!\""}],file="function1.cpp",fullname="/home/erniep/Development/Peak/src/Seer/helloworld/function1.cpp",line="11",arch="i386:x86-64"},thread-id="1",stopped-threads="all",core="4"

            QString hwwpt_text     = Seer::parseFirst(text,       "hw-rwpt=",   '{', '}', false);
            QString number_text    = Seer::parseFirst(hwwpt_text, "number=",    '"', '"', false);
            QString exp_text       = Seer::parseFirst(hwwpt_text, "exp=",       '"', '"', false);
            QString value_text     = Seer::parseFirst(text,       "value=",     '{', '}', false);
            QString value_text2    = Seer::parseFirst(value_text, "value=",     '"', '"', false);
            QString frame_text     = Seer::parseFirst(text,       "frame=",     '{', '}', false);
            QString file_text      = Seer::parseFirst(frame_text, "file=",      '"', '"', false);
            QString fullname_text  = Seer::parseFirst(frame_text, "fullname=",  '"', '"', false);
            QString line_text      = Seer::parseFirst(frame_text, "line=",      '"', '"', false);

            // Find watchpoint number in the tree
            QList<QTreeWidgetItem*> matches = watchpointsTreeWidget->findItems(number_text, Qt::MatchExactly, 0);
            if (matches.size() > 0) {
                //qDebug() << text;
                QTreeWidgetItem* item = matches.first();
                item->setText(2, value_text2);
                item->setText(3, "");
                item->setText(9, QFileInfo(file_text).fileName());
                item->setText(10, fullname_text);
                item->setText(11, line_text);
            }

        }else if (reason_text == "access-watchpoint-trigger") {
            //*stopped,reason="access-watchpoint-trigger",hw-awpt={number="3",exp="v"},value={old="1",new="11"},frame={addr="0x000000000040059a",func="bar",args=[{name="v",value="11"}],file="helloonefile.cpp",fullname="/home/erniep/Development/Peak/src/Seer/helloonefile/helloonefile.cpp",line="15",arch="i386:x86-64"},thread-id="1",stopped-threads="all",core="3"

            QString hwawpt_text    = Seer::parseFirst(text,        "hw-awpt=",  '{', '}', false);
            QString number_text    = Seer::parseFirst(hwawpt_text, "number=",   '"', '"', false);
            QString exp_text       = Seer::parseFirst(hwawpt_text, "exp=",      '"', '"', false);
            QString value_text     = Seer::parseFirst(text,        "value=",    '{', '}', false);
            QString old_text       = Seer::parseFirst(value_text,  "old=",      '"', '"', false);
            QString new_text       = Seer::parseFirst(value_text,  "new=",      '"', '"', false);
            QString frame_text     = Seer::parseFirst(text,       "frame=",     '{', '}', false);
            QString file_text      = Seer::parseFirst(frame_text, "file=",      '"', '"', false);
            QString fullname_text  = Seer::parseFirst(frame_text, "fullname=",  '"', '"', false);
            QString line_text      = Seer::parseFirst(frame_text, "line=",      '"', '"', false);

            // Find watchpoint number in the tree
            QList<QTreeWidgetItem*> matches = watchpointsTreeWidget->findItems(number_text, Qt::MatchExactly, 0);
            if (matches.size() > 0) {
                //qDebug() << text;
                QTreeWidgetItem* item = matches.first();
                item->setText(2, old_text);
                item->setText(3, new_text);
                item->setText(9, QFileInfo(file_text).fileName());
                item->setText(10, fullname_text);
                item->setText(11, line_text);
            }
        }

    }else if (text.startsWith("^error,msg=\"No registers.\"")) {
        // Ignore.

    }else{
        // Ignore others.
    }

    watchpointsTreeWidget->resizeColumnToContents(0);
    watchpointsTreeWidget->resizeColumnToContents(1);
    watchpointsTreeWidget->resizeColumnToContents(2);
    watchpointsTreeWidget->resizeColumnToContents(3);
    watchpointsTreeWidget->resizeColumnToContents(4);
    watchpointsTreeWidget->resizeColumnToContents(5);
    watchpointsTreeWidget->resizeColumnToContents(6);
    watchpointsTreeWidget->resizeColumnToContents(7);
  //watchpointsTreeWidget->resizeColumnToContents(8);
    watchpointsTreeWidget->resizeColumnToContents(9);
  //watchpointsTreeWidget->resizeColumnToContents(10);
    watchpointsTreeWidget->resizeColumnToContents(11);
    watchpointsTreeWidget->resizeColumnToContents(12);
    watchpointsTreeWidget->resizeColumnToContents(13);
    watchpointsTreeWidget->resizeColumnToContents(14);
    watchpointsTreeWidget->resizeColumnToContents(15);
  //watchpointsTreeWidget->resizeColumnToContents(16);
    watchpointsTreeWidget->resizeColumnToContents(17);

    QApplication::restoreOverrideCursor();
}

void SeerWatchpointsBrowserWidget::handleStoppingPointReached () {

    // Don't do any work if the widget is hidden.
    if (isHidden()) {
        return;
    }

    emit refreshWatchpointsList();
}

void SeerWatchpointsBrowserWidget::handleItemDoubleClicked (QTreeWidgetItem* item, int column) {

    Q_UNUSED(column);

    int lineno = item->text(11).toInt();

    emit selectedFile(item->text(9), item->text(10), lineno);
}

void SeerWatchpointsBrowserWidget::handleRefreshToolButton () {

    emit refreshWatchpointsList();
}

void SeerWatchpointsBrowserWidget::handleAddToolButton () {

    SeerWatchpointCreateDialog dlg(this);

    int ret = dlg.exec();

    if (ret == 0) {
        return;
    }

    // Build a watchpoint specification.
    QString watchpointParameters = dlg.watchpointText();

    // If nothing, just return.
    if (watchpointParameters == "") {
        return;
    }

    // Otherwise send the command to create the watchpoint.
    emit insertWatchpoint(watchpointParameters);
}

void SeerWatchpointsBrowserWidget::handleDeleteToolButton () {

    // Get selected tree items.
    QList<QTreeWidgetItem*> items =  watchpointsTreeWidget->selectedItems();

    // Build a string that is a list of watchpoints.
    QString watchpoints;

    QList<QTreeWidgetItem*>::iterator i;
    for (i = items.begin(); i != items.end(); ++i) {
        if (i != items.begin()) {
            watchpoints += " ";
        }
        watchpoints += (*i)->text(0);
    }

    // Don't do anything if the list of watchpoints is empty.
    if (watchpoints == "") {
        return;
    }

    // Send the signal.
    emit deleteWatchpoints(watchpoints);
}

void SeerWatchpointsBrowserWidget::handleEnableToolButton () {

    // Get selected tree items.
    QList<QTreeWidgetItem*> items = watchpointsTreeWidget->selectedItems();

    // Build a string that is a list of watchpoints.
    QString watchpoints;

    QList<QTreeWidgetItem*>::iterator i;
    for (i = items.begin(); i != items.end(); ++i) {

        if (i != items.begin()) {
            watchpoints += " ";
        }

        watchpoints += (*i)->text(0);
    }

    // Don't do anything if the list of watchpoints is empty.
    if (watchpoints == "") {
        return;
    }

    // Send the signal.
    emit enableWatchpoints(watchpoints);
}

void SeerWatchpointsBrowserWidget::handleDisableToolButton () {

    // Get selected tree items.
    QList<QTreeWidgetItem*> items = watchpointsTreeWidget->selectedItems();

    // Build a string that is a list of watchpoints.
    QString watchpoints;

    QList<QTreeWidgetItem*>::iterator i;
    for (i = items.begin(); i != items.end(); ++i) {

        if (i != items.begin()) {
            watchpoints += " ";
        }

        watchpoints += (*i)->text(0);
    }

    // Don't do anything if the list of watchpoints is empty.
    if (watchpoints == "") {
        return;
    }

    // Send the signal.
    emit disableWatchpoints(watchpoints);
}

void SeerWatchpointsBrowserWidget::handleConditionToolButton () {

    // Get selected tree items. Only allow one.
    QList<QTreeWidgetItem*> items = watchpointsTreeWidget->selectedItems();

    if (items.count() == 0) {
        return;
    }

    if (items.count() > 1) {
        QMessageBox::warning(this, "Seer", "Select only one watchpoint when adding a condition.", QMessageBox::Ok);
        return;
    }

    // Get the condition text.
    bool ok;
    QString condition = QInputDialog::getText(this, "Seer", "Enter the condition for this watchpoint.\nA blank condition will remove an existing one.", QLineEdit::Normal, items.front()->text(13), &ok);

    if (ok == false) {
        return;
    }

    // Get the selected watchpoint number.
    QString watchpoint = items.front()->text(0);

    // Send the signal.
    emit addBreakpointCondition(watchpoint, condition);
}

void SeerWatchpointsBrowserWidget::handleIgnoreToolButton () {

    // Get selected tree items. Only allow one.
    QList<QTreeWidgetItem*> items = watchpointsTreeWidget->selectedItems();

    if (items.count() == 0) {
        return;
    }

    if (items.count() > 1) {
        QMessageBox::warning(this, "Seer", "Select only one watchpoint when adding an ignore count.", QMessageBox::Ok);
        return;
    }

    // Get the ignore text.
    bool ok;
    int count = QInputDialog::getInt(this, "Seer", "Enter the ignore count for this watchpoint.\nA count of 0 will remove an existing one.", items.front()->text(15).toInt(), 0, 2147483647, 1, &ok);

    if (ok == false) {
        return;
    }

    // Get the selected watchpoint number.
    QString watchpoint = items.front()->text(0);

    // Send the signal.
    emit addBreakpointIgnore(watchpoint, QString::number(count));
}

void SeerWatchpointsBrowserWidget::handleCommandsToolButton () {

    // Get selected tree items. Only allow one.
    QList<QTreeWidgetItem*> items = watchpointsTreeWidget->selectedItems();

    if (items.count() == 0) {
        return;
    }

    if (items.count() > 1) {
        QMessageBox::warning(this, "Seer", "Select only one watcpoint when adding commands.", QMessageBox::Ok);
        return;
    }

    // Get the ignore text.
    bool ok;
    QString commandstr = QInputDialog::getMultiLineText(this, "Seer", "Enter the commands to execute for this watchpoint.\nA blank list will remove existing ones.", items.front()->text(16), &ok);

    if (ok == false) {
        return;
    }

    // Get the selected watchpoint number.
    QString watchpoint = items.front()->text(0);

    QStringList commands = Seer::quoteChars(commandstr.split('\n', Qt::SkipEmptyParts), "\"");

    // Send the signal.
    emit addBreakpointCommands(watchpoint, commands);
}

void SeerWatchpointsBrowserWidget::showEvent (QShowEvent* event) {

    QWidget::showEvent(event);

    emit refreshWatchpointsList();
}

