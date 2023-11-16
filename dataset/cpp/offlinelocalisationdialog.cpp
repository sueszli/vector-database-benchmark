#include "offlinelocalisationdialog.h"
#include "OfflineLocalisation.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTextBrowser>
#include <QPushButton>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QLabel>
#include <QProgressDialog>
#include <QFileDialog>
#include <QAction>
#include "FileAccess/SplitStreamFileFormatReader.h"

OfflineLocalisationDialog::OfflineLocalisationDialog(QWidget *parent) :
    QDialog(parent)
{
    m_reader = new LogFileReader();
    m_external_reader = false;
    m_offline_loc = new OfflineLocalisation(m_reader);
    MakeLayout();
}

OfflineLocalisationDialog::OfflineLocalisationDialog(LogFileReader* reader, QWidget *parent): QDialog(parent)
{
    m_reader = reader;
    m_external_reader = true;
    m_offline_loc = new OfflineLocalisation(m_reader);
    MakeLayout();
}

OfflineLocalisationDialog::~OfflineLocalisationDialog()
{
    if(m_offline_loc) delete m_offline_loc;
}

void OfflineLocalisationDialog::MakeLayout()
{
    this->setWindowTitle("Offline Localisation");

    QVBoxLayout *buttonsLayout = new QVBoxLayout();

    QPushButton *openFileButton = new QPushButton("&Open Log...");
    connect(openFileButton,SIGNAL(clicked()), this, SLOT(OpenLogFiles()));
    buttonsLayout->addWidget(openFileButton);

    QPushButton *runSimulationButton = new QPushButton("&Run simulation");
    connect(runSimulationButton,SIGNAL(clicked()), this, SLOT(BeginSimulation()));
    buttonsLayout->addWidget(runSimulationButton);

    QPushButton *saveLogButton = new QPushButton("Save &Log...");
    saveLogButton->setEnabled(false);
    connect(m_offline_loc,SIGNAL(SimDataChanged(bool)), saveLogButton, SLOT(setEnabled(bool)));
    connect(saveLogButton,SIGNAL(clicked()), this, SLOT(SaveAsLocalisationLog()));

    buttonsLayout->addWidget(saveLogButton);

    QPushButton *saveReportButton = new QPushButton("Save Report...");
    saveReportButton->setEnabled(false);
    connect(m_offline_loc,SIGNAL(SimDataChanged(bool)), saveReportButton, SLOT(setEnabled(bool)));
    //connect(runSimulationButton,SIGNAL(pressed()), this, SLOT(BeginSimulation()));
    buttonsLayout->addWidget(saveReportButton);


    QVBoxLayout *displayLayout = new QVBoxLayout();

    QLabel *fileLabel = new QLabel("Log files");
    displayLayout->addWidget(fileLabel);

    m_fileListDisplay = new QTextBrowser(this);
    m_fileListDisplay->setWordWrapMode(QTextOption::NoWrap);

    displayLayout->addWidget(m_fileListDisplay);


    QHBoxLayout *overallLayout = new QHBoxLayout();

    overallLayout->addLayout(buttonsLayout);
    overallLayout->addLayout(displayLayout,1);

    m_progressBar = new QProgressDialog("Runing localisation...","Cancel",0, m_offline_loc->NumberOfLogFrames(),this);
    m_progressBar->setWindowModality(Qt::WindowModal);
    m_progressBar->setValue(0);
    m_progressBar->setMinimumDuration(100);
    connect(m_offline_loc, SIGNAL(updateProgress(int,int)), this, SLOT(DiplayProgress(int,int)));
    connect(m_offline_loc, SIGNAL(finished()), this, SLOT(CompleteSimulation()));
    connect(m_progressBar, SIGNAL(canceled()), this, SLOT(CancelProgress()));

    connect(m_reader, SIGNAL(OpenLogFilesChanged(std::vector<QFileInfo>)), this, SLOT(SetOpenFileList(std::vector<QFileInfo>)));

    setLayout(overallLayout);
}

void OfflineLocalisationDialog::OpenLogFiles()
{
    QString filename = QFileDialog::getOpenFileName(this,
                            tr("Open Replay File"), ".",
                            tr("All NUbot Image Files(*.nul;*.nif;*.nurf;*.strm);;NUbot Log Files (*.nul);;NUbot Image Files (*.nif);;NUbot Replay Files (*.nurf);;Stream File(*.strm);;All Files(*.*)"));

    if (!filename.isEmpty()){
        m_offline_loc->OpenLogs(filename.toStdString());
    }
}

void OfflineLocalisationDialog::SaveAsLocalisationLog()
{
    if(!m_offline_loc->hasSimData()) return;
    QString save_name = QFileDialog::getSaveFileName(this,"Save Log",QString(),"Stream (*.strm)");
    m_offline_loc->WriteLog(save_name.toStdString());
    return;
}

void OfflineLocalisationDialog::BeginSimulation()
{
    if(m_offline_loc->isRunning()) return;
    m_progressBar->setRange(0, m_offline_loc->NumberOfLogFrames());
    m_progressBar->setValue(0);

//    m_progressBar = new QProgressDialog("Runing localisation...","Cancel",0, m_offline_loc->NumberOfLogFrames(),this);
//    m_progressBar->setWindowModality(Qt::WindowModal);
//    m_progressBar->setValue(0);
//    m_progressBar->setMinimumDuration(100);
//    connect(m_offline_loc, SIGNAL(updateProgress(int,int)), this, SLOT(DiplayProgress(int,int)));
//    connect(m_offline_loc, SIGNAL(finished()), this, SLOT(CompleteSimulation()));
//    connect(m_progressBar, SIGNAL(canceled()), this, SLOT(CancelProgress()));

    //m_offline_loc->Run();
    m_offline_loc->start();
}

void OfflineLocalisationDialog::CompleteSimulation()
{
//    disconnect(m_offline_loc, SIGNAL(finished()), this, SLOT(CompleteSimulation()));
//    disconnect(m_offline_loc, SIGNAL(updateProgress(int,int)), this, SLOT(DiplayProgress(int,int)));
    //delete m_progressBar;
    //m_progressBar = NULL;
}

void OfflineLocalisationDialog::DiplayProgress(int frame, int total)
{
    if(m_progressBar && !m_offline_loc->wasStopped())
    {
        QString labelText = QString("Running Localisation... Frame %1 of %2").arg(frame).arg(total);
        m_progressBar->setLabelText(labelText);
        m_progressBar->setValue(frame);
    }
}

void OfflineLocalisationDialog::CancelProgress()
{
    if(m_offline_loc->isRunning()) m_offline_loc->stop();
}

void OfflineLocalisationDialog::SetFrame(int frameNumber, int total)
{
    if(m_offline_loc->hasSimData())
    {
        const Localisation* temp = m_offline_loc->GetFrame(frameNumber);
        emit LocalisationChanged(temp);
        QString message = m_offline_loc->GetFrameInfo(frameNumber);
        emit LocalisationInfoChanged(message);
    }
}

void OfflineLocalisationDialog::SetOpenFileList(std::vector<QFileInfo> files)
{
    QString displayString;

    std::vector<QFileInfo>::const_iterator fileIt;
    for (fileIt = files.begin(); fileIt != files.end(); ++fileIt)
    {
        displayString += (*fileIt).filePath() + '\n';
    }
    m_fileListDisplay->setText(displayString);
}
