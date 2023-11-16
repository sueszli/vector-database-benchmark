#include "PullDlg.h"
#include "ui_PullDlg.h"

#include <GitBase.h>
#include <GitQlientSettings.h>
#include <GitQlientStyles.h>
#include <GitRemote.h>

#include <QMessageBox>
#include <QPushButton>

PullDlg::PullDlg(QSharedPointer<GitBase> git, const QString &text, QWidget *parent)
   : QDialog(parent)
   , ui(new Ui::PullDlg)
   , mGit(git)
{
   ui->setupUi(this);

   ui->lText->setText(text);
   ui->lQuestion->setText(tr("<strong>Would you like to pull the last changes?</strong>"));
   ui->buttonBox->button(QDialogButtonBox::Ok)->setText("Pull");

   setStyleSheet(GitQlientStyles::getStyles());
}

PullDlg::~PullDlg()
{
   delete ui;
}

void PullDlg::accept()
{
   GitQlientSettings settings(mGit->getGitDir());
   const auto updateOnPull = settings.localValue("UpdateOnPull", true).toBool();

   QScopedPointer<GitRemote> git(new GitRemote(mGit));

   QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
   const auto ret = git->pull(updateOnPull);
   QApplication::restoreOverrideCursor();

   if (ret.success)
   {
      emit signalRepositoryUpdated();

      QDialog::accept();
   }
   else
   {
      if (ret.output.contains("error: could not apply", Qt::CaseInsensitive)
          && ret.output.contains("causing a conflict", Qt::CaseInsensitive))
      {
         emit signalPullConflict();
      }
      else
      {
         QMessageBox msgBox(QMessageBox::Critical, tr("Error while pulling"),
                            QString(tr("There were problems during the pull operation. Please, see the detailed "
                                       "description for more information.")),
                            QMessageBox::Ok, this);
         msgBox.setDetailedText(ret.output);
         msgBox.setStyleSheet(GitQlientStyles::getStyles());
         msgBox.exec();
      }
   }
}
