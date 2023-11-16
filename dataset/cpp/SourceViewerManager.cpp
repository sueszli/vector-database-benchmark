#include "SourceViewerManager.h"
#include "Bridge.h"
#include <QFileInfo>
#include <QDir>
#include <QTimer>

SourceViewerManager::SourceViewerManager(QWidget* parent) : QTabWidget(parent)
{
    setMovable(true);
    setTabsClosable(true);

    //Close All Tabs
    mCloseAllTabs = new QPushButton(this);
    mCloseAllTabs->setIcon(DIcon("close-all-tabs"));
    mCloseAllTabs->setToolTip(tr("Close All Tabs"));
    connect(mCloseAllTabs, SIGNAL(clicked()), this, SLOT(closeAllTabs()));
    setCornerWidget(mCloseAllTabs, Qt::TopLeftCorner);

    connect(this, SIGNAL(tabCloseRequested(int)), this, SLOT(closeTab(int)));
    connect(Bridge::getBridge(), SIGNAL(loadSourceFile(QString, duint)), this, SLOT(loadSourceFile(QString, duint)));
    connect(Bridge::getBridge(), SIGNAL(dbgStateChanged(DBGSTATE)), this, SLOT(dbgStateChanged(DBGSTATE)));
}

void SourceViewerManager::loadSourceFile(QString path, duint addr)
{
    for(int i = 0; i < count(); i++)
    {
        SourceView* curView = (SourceView*)this->widget(i);
        if(curView->getSourcePath().compare(path, Qt::CaseInsensitive) == 0) //file already loaded
        {
            curView->setSelection(addr);
            setCurrentIndex(i); //show that loaded tab
            QTimer::singleShot(50, [curView]()
            {
                curView->setFocus();
            });
            return;
        }
    }
    //check if file actually exists
    if(!QFileInfo(path).exists())
    {
        return; //error?
    }
    //load the new file
    QString title = path;
    int idx = path.lastIndexOf(QDir::separator());
    if(idx != -1)
        title = path.mid(idx + 1);
    SourceView* newView = new SourceView(path, addr, this);
    addTab(newView, title);
    setCurrentIndex(count() - 1);
    // https://forum.qt.io/post/132664
    // For some reason the viewport() in the AbstractTableView does not have the right size which means setSelection completely fails
    QTimer::singleShot(50, [newView, addr]()
    {
        newView->setSelection(addr);
        newView->setFocus();
    });
}

void SourceViewerManager::closeTab(int index)
{
    auto sourceView = qobject_cast<SourceView*>(widget(index));
    removeTab(index);
    if(sourceView)
        delete sourceView;
}

void SourceViewerManager::closeAllTabs()
{
    while(count())
    {
        closeTab(0);
    }
}

void SourceViewerManager::dbgStateChanged(DBGSTATE state)
{
    if(state == stopped)
        closeAllTabs();
}
