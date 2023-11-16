#include "ztimeline.h"
#include "zslider.h"
#include "zenomainwindow.h"
#include "zenoapplication.h"
#include "viewport/viewportwidget.h"
#include <comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/effect/innershadoweffect.h>
#include <zeno/utils/envconfig.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/zwidgetfactory.h>
#include "../viewport/zenovis.h"
#include "ui_ztimeline.h"
#include <zenoui/comctrl/view/zcomboboxitemdelegate.h>
#include "viewport/zenovis.h"
#include <zenovis/DrawOptions.h>
#include <iostream>
#include "viewport/displaywidget.h"


//////////////////////////////////////////////
ZTimeline::ZTimeline(QWidget* parent)
    : QWidget(parent)
{
    m_ui = new Ui::Timeline;
    m_ui->setupUi(this);

    QStringList items = { "23.5 fps", "24 fps", "25 fps", "30 fps", "60 fps" };
    m_ui->comboBox->addItems(items);
    m_ui->comboBox->setItemDelegate(new ZComboBoxItemDelegate2(m_ui->comboBox));
    m_ui->comboBox->setProperty("cssClass", "newstyle");
 
    setFocusPolicy(Qt::ClickFocus);
    //QPalette pal = palette();
    //pal.setColor(QPalette::Window, QColor(42, 42, 42));
    //setAutoFillBackground(true);
    //setPalette(pal);

    int deflFrom = 0, deflTo = 0;
    m_ui->editFrom->setText(QString::number(deflFrom));
    m_ui->editTo->setText(QString::number(deflTo));
    m_ui->timeliner->setFromTo(deflFrom, deflTo);
    m_ui->fpsEdit->setText("24");
    
    initStyleSheet();
    initSignals();
    initButtons();
    initSize();
    m_ui->fpsEdit->setValidator(new QIntValidator);
    m_ui->editFrom->setValidator(new QIntValidator);
    m_ui->editTo->setValidator(new QIntValidator);
    m_ui->editFrame->setValidator(new QIntValidator);
    m_ui->comboBox->hide();
}

void ZTimeline::initSignals()
{
    connect(m_ui->btnPlay, &ZToolButton::toggled, this, [=](bool toggle) {
        emit playForward(toggle);
    });
    connect(m_ui->editFrom, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->editTo, SIGNAL(editingFinished()), this, SLOT(onFrameEditted()));
    connect(m_ui->timeliner, SIGNAL(sliderValueChange(int)), this, SIGNAL(sliderValueChanged(int)));

//merge from master.
#if 0
    connect(m_ui->editSR, &QLineEdit::editingFinished, this, [=]() {
        auto srTime = std::abs(m_ui->editSR->text().toInt());  // Avoid negative
        //std::cout << "SR: SimpleRender " << srTime << "\n";
        Zenovis* pZenovis = views[0]->getZenoVis();
        auto scene = pZenovis->get_scene();
        viewport->simpleRenderTime = srTime;
        if(srTime == 0){
            scene->drawOptions->simpleRender = false;
        }else{
            scene->drawOptions->simpleRender = true;
        }
        scene->drawOptions->needRefresh = true;
    });
#endif

    connect(m_ui->btnBackward, &ZToolButton::clicked, this, [=]() {
        int frame = m_ui->timeliner->value();
        auto ft = fromTo();
        int frameFrom = ft.first, frameTo = ft.second;
        if (frame > frameFrom && frameFrom >= 0)
        {
            m_ui->timeliner->setSliderValue(frame - 1);
        }
    });
    connect(m_ui->btnForward, &ZToolButton::clicked, this, [=]() {
        int frame = m_ui->timeliner->value();
        auto ft = fromTo();
        int frameFrom = ft.first, frameTo = ft.second;
        if (frame < frameTo)
        {
            m_ui->timeliner->setSliderValue(frame + 1);
        }
    });
    connect(m_ui->editFrame, &QLineEdit::editingFinished, this, [=]() {
        int frame = m_ui->editFrame->text().toInt();
        int frameFrom = m_ui->editFrom->text().toInt();
        int frameTo = m_ui->editTo->text().toInt();
        if (frame < frameFrom)
        {
            BlockSignalScope scope(m_ui->editFrame);
            frame = frameFrom;
            m_ui->editFrame->setText(QString::number(frame));
            m_ui->timeliner->setSliderValue(frame);
            return;
        }
        else if (frame > frameTo)
        {
            BlockSignalScope scope(m_ui->editFrame);
            frame = frameTo;
            m_ui->editFrame->setText(QString::number(frame));
            m_ui->timeliner->setSliderValue(frame);
            return;
        }
        m_ui->timeliner->setSliderValue(frame);
    });
    connect(this, &ZTimeline::sliderValueChanged, this, [=]() {
        QString numText = QString::number(m_ui->timeliner->value());
        m_ui->editFrame->setText(numText);
    });
    connect(m_ui->fpsEdit, &QLineEdit::editingFinished, this, [=]() {
        if (m_ui->fpsEdit->text().toInt() < 1)
            m_ui->fpsEdit->setText("1");
        auto main = zenoApp->getMainWindow();
        ZASSERT_EXIT(main);
        for (auto view : main->viewports())
        {
            view->setSliderFeq(1000 / m_ui->fpsEdit->text().toInt());
        }
    });
}

void ZTimeline::onFrameEditted()
{
    if (m_ui->editFrom->text().isEmpty() || m_ui->editTo->text().isEmpty())
        return;

    QObject* pSender = sender();
    int frameFrom = m_ui->editFrom->text().toInt();
    int frameTo = m_ui->editTo->text().toInt();
    if (frameFrom > frameTo)
    {
        if (m_ui->editFrom == pSender)
        {
            frameTo = frameFrom;
            BlockSignalScope scope(m_ui->editTo);
            m_ui->editTo->setText(QString::number(frameTo));
        }
        else if (m_ui->editTo == pSender)
        {
            frameFrom = frameTo;
            BlockSignalScope scope(m_ui->editFrom);
            m_ui->editFrom->setText(QString::number(frameFrom));
        }
    }
    m_ui->timeliner->setFromTo(frameFrom, frameTo);
}

void ZTimeline::initStyleSheet()
{
    auto editors = findChildren<QLineEdit *>(QString(), Qt::FindChildrenRecursively);
    for (QLineEdit *pLineEdit : editors) {
        pLineEdit->setProperty("cssClass", "FCurve-lineedit");
    }
    m_ui->fpslabel->setProperty("cssClass", "proppanel");
}

void ZTimeline::initButtons()
{
    QSize sz = ZenoStyle::dpiScaledSize(QSize(24, 24));

    QColor hoverBg("#4F5963");

    m_ui->btnBackToStart->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnBackToStart->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_startFrame_idle.svg",
        ":/icons/timeline_startFrame_light.svg",
        "",
        "");
    m_ui->btnBackToStart->setMargins(ZenoStyle::dpiScaledMargins(QMargins(3, 2, 2, 3)));
    m_ui->btnBackToStart->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);
    m_ui->btnBackToStart->setToolTip(tr("Back To Start"));

    m_ui->btnBackward->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnBackward->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_previousFrame_idle.svg",
        ":/icons/timeline_previousFrame_light.svg",
        "",
        "");
    m_ui->btnBackward->setMargins(ZenoStyle::dpiScaledMargins(QMargins(3, 2, 2, 3)));
    m_ui->btnBackward->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);
    m_ui->btnBackward->setToolTip(tr("Backward"));

    m_ui->btnPlay->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnPlay->setIcon(
        ZenoStyle::dpiScaledSize(QSize(26, 26)),
        ":/icons/timeline_pause_idle.svg",
        ":/icons/timeline_pause_hover.svg",
        ":/icons/timeline_play_idle.svg",
        ":/icons/timeline_play_hover.svg");
    m_ui->btnPlay->setMargins(ZenoStyle::dpiScaledMargins(QMargins(3, 2, 2, 3)));
    m_ui->btnPlay->setBackgroundClr(QColor(), QColor(), QColor(), QColor());
    m_ui->btnPlay->setToolTip(tr("Play"));

    m_ui->btnForward->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnForward->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_nextFrame_idle.svg",
        ":/icons/timeline_nextFrame_light.svg",
        "",
        "");
    m_ui->btnForward->setMargins(ZenoStyle::dpiScaledMargins(QMargins(3, 2, 2, 3)));
    m_ui->btnForward->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);
    m_ui->btnForward->setToolTip(tr("Forward"));

    m_ui->btnForwardToEnd->setButtonOptions(ZToolButton::Opt_HasIcon);
    m_ui->btnForwardToEnd->setIcon(
        ZenoStyle::dpiScaledSize(QSize(24, 24)),
        ":/icons/timeline_endFrame_idle.svg",
        ":/icons/timeline_endFrame_light.svg",
        "",
        "");
    m_ui->btnForwardToEnd->setMargins(ZenoStyle::dpiScaledMargins(QMargins(3, 2, 2, 3)));
    m_ui->btnForwardToEnd->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);
    m_ui->btnForwardToEnd->setToolTip(tr("Forward To End"));

    QColor bg(35, 40, 47);
    m_ui->btnLoopPlay->setButtonOptions(ZToolButton::Opt_HasIcon | ZToolButton::Opt_Checkable);
    m_ui->btnLoopPlay->setIcon(
        ZenoStyle::dpiScaledSize(QSize(16, 16)),
        ":/icons/loop_off.png",
        "",
        ":/icons/loop_on.png",
        "");
    m_ui->btnLoopPlay->setMargins(QMargins(3, 2, 2, 3));
    m_ui->btnLoopPlay->setBackgroundClr(bg, hoverBg, bg, hoverBg);
    m_ui->btnLoopPlay->setToolTip(tr("Loop Play"));
    connect(m_ui->btnLoopPlay, &ZToolButton::toggled, this, [=](bool bChecked) {
        ZenoMainWindow* pMainWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pMainWin);
    for (auto view: pMainWin->viewports())
    {
        view->setLoopPlaying(bChecked);
    }
        });


    //m_ui->btnRecycle->setButtonOptions(ZToolButton::Opt_HasIcon);
    //m_ui->btnRecycle->setIcon(
    //    ZenoStyle::dpiScaledSize(QSize(24, 24)),
    //    ":/icons/timeline_loopMethod_loop.svg",
    //    ":/icons/timeline_loopMethod_loop.svg",
    //    "",
    //    "");
    //m_ui->btnRecycle->setMargins(QMargins(3, 2, 2, 3));
    //m_ui->btnRecycle->setBackgroundClr(QColor(), hoverBg, QColor(), hoverBg);

    ////m_ui->btnSimpleRender->setProperty("cssClass", "grayButton");
    //m_ui->btnSimpleRender->setFont(font);
}

void ZTimeline::initSize()
{
    m_ui->comboBox->setFixedSize(ZenoStyle::dpiScaledSize(QSize(96, 20)));
    m_ui->editFrame->setFixedSize(ZenoStyle::dpiScaledSize(QSize(38, 20)));
    m_ui->btnPlay->setFixedSize(ZenoStyle::dpiScaledSize(QSize(26, 26)));
    m_ui->fpsEdit->setFixedSize(ZenoStyle::dpiScaledSize(QSize(38, 20)));
}

void ZTimeline::onTimelineUpdate(int frameid)
{
    bool blocked = m_ui->timeliner->signalsBlocked();
    BlockSignalScope scope(m_ui->timeliner);
    BlockSignalScope scope2(m_ui->editFrame);
    m_ui->timeliner->setSliderValue(frameid);
    m_ui->editFrame->setText(QString::number(frameid));
}

void ZTimeline::setSliderValue(int frameid)
{
    m_ui->timeliner->setSliderValue(frameid);
}

void ZTimeline::setPlayButtonChecked(bool bToggle)
{
    m_ui->btnPlay->setChecked(bToggle);
}

void ZTimeline::togglePlayButton(bool bOn)
{
    m_ui->btnPlay->toggle(bOn);
}

QPair<int, int> ZTimeline::fromTo() const
{
    bool bOk = false;
    int frameFrom = m_ui->editFrom->text().toInt(&bOk);
    int frameTo = m_ui->editTo->text().toInt(&bOk);
    return { frameFrom, frameTo };
}

void ZTimeline::initFromTo(int frameFrom, int frameTo)
{
    BlockSignalScope s1(m_ui->timeliner);
    BlockSignalScope s2(m_ui->editFrom);
    BlockSignalScope s3(m_ui->editTo);

    m_ui->editFrom->setText(QString::number(frameFrom));
    m_ui->editTo->setText(QString::number(frameTo));
    if (frameTo >= frameFrom)
        m_ui->timeliner->setFromTo(frameFrom, frameTo);
}

void ZTimeline::initFps(int fps)
{
    BlockSignalScope s1(m_ui->fpsEdit);
    m_ui->fpsEdit->setText(QString::number(fps));
}

int ZTimeline::fps()
{
    return m_ui->fpsEdit->text().toInt();
}

void ZTimeline::resetSlider()
{
    m_ui->timeliner->setSliderValue(0);
}

int ZTimeline::value() const
{
    return m_ui->timeliner->value();
}

bool ZTimeline::isPlayToggled() const
{
    return m_ui->btnPlay->isChecked();
}

void ZTimeline::updateKeyFrames(const QVector<int>& keys) 
{
    m_ui->timeliner->updateKeyFrames(keys);
}

void ZTimeline::updateCachedFrame()
{
    m_ui->timeliner->update();
}

void ZTimeline::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    QPen pen(QColor("#000000"), 1);
    painter.setPen(pen);
    painter.setBrush(QColor("#2d3239"));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
}