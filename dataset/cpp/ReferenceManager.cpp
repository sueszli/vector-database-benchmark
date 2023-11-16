#include "ReferenceManager.h"
#include "Bridge.h"

ReferenceManager::ReferenceManager(QWidget* parent) : QTabWidget(parent)
{
    setMovable(true);
    setTabsClosable(true);
    mCurrentReferenceView = 0;

    //Close All Tabs
    mCloseAllTabs = new QPushButton(this);
    mCloseAllTabs->setIcon(DIcon("close-all-tabs"));
    mCloseAllTabs->setToolTip(tr("Close All Tabs"));
    connect(mCloseAllTabs, SIGNAL(clicked()), this, SLOT(closeAllTabs()));
    setCornerWidget(mCloseAllTabs, Qt::TopLeftCorner);
    setContextMenuPolicy(Qt::CustomContextMenu);

    connect(Bridge::getBridge(), SIGNAL(referenceInitialize(QString)), this, SLOT(newReferenceView(QString)));
    connect(this, SIGNAL(tabCloseRequested(int)), this, SLOT(closeTab(int)));
}

ReferenceView* ReferenceManager::currentReferenceView()
{
    //get the current index, disconnects the previous view if it's not the current one, set and connect the new current view, then return it
    int currentIndex = QTabWidget::currentIndex();

    if(mCurrentReferenceView && mCurrentReferenceView != widget(currentIndex))
    {
        mCurrentReferenceView->disconnectBridge();
        mCurrentReferenceView = qobject_cast<ReferenceView*>(widget(currentIndex));

        if(mCurrentReferenceView)
            mCurrentReferenceView->connectBridge();
    }

    return mCurrentReferenceView;
}

void ReferenceManager::newReferenceView(QString name)
{
    if(mCurrentReferenceView) //disconnect previous reference view
        mCurrentReferenceView->disconnectBridge();
    mCurrentReferenceView = new ReferenceView(false, this);
    mCurrentReferenceView->connectBridge();
    connect(mCurrentReferenceView, SIGNAL(showCpu()), this, SIGNAL(showCpu()));
    insertTab(0, mCurrentReferenceView, name);
    setCurrentIndex(0);
    Bridge::getBridge()->setResult(BridgeResult::RefInitialize, 1);
}

void ReferenceManager::closeTab(int index)
{
    auto view = qobject_cast<ReferenceView*>(widget(index));
    removeTab(index);
    if(mCurrentReferenceView == view)
        mCurrentReferenceView = nullptr;
    if(view)
        delete view;
    if(count() <= 0)
        emit showCpu();
}

void ReferenceManager::closeAllTabs()
{
    while(count())
    {
        closeTab(0);
    }
}
