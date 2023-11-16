#include "AddSubtreeDlg.h"
#include "ui_AddSubtreeDlg.h"

#include <GitBase.h>
#include <GitQlientSettings.h>
#include <GitQlientStyles.h>
#include <GitSubtree.h>
#include <QLogger.h>

#include <QMessageBox>

using namespace QLogger;

AddSubtreeDlg::AddSubtreeDlg(const QSharedPointer<GitBase> &git, QWidget *parent)
   : QDialog(parent)
   , ui(new Ui::AddSubtreeDlg)
   , mGit(git)
{
   setStyleSheet(GitQlientStyles::getStyles());

   ui->setupUi(this);

   connect(ui->lePath, &QLineEdit::returnPressed, this, &AddSubtreeDlg::accept);
   connect(ui->leUrl, &QLineEdit::returnPressed, this, &AddSubtreeDlg::accept);
   connect(ui->leUrl, &QLineEdit::editingFinished, this, &AddSubtreeDlg::proposeName);
   connect(ui->pbAccept, &QPushButton::clicked, this, &AddSubtreeDlg::accept);
   connect(ui->pbCancel, &QPushButton::clicked, this, &QDialog::reject);
}

AddSubtreeDlg::AddSubtreeDlg(const QString &prefix, const QSharedPointer<GitBase> &git, QWidget *parent)
   : AddSubtreeDlg(git, parent)
{
   disconnect(ui->leUrl, &QLineEdit::editingFinished, this, &AddSubtreeDlg::proposeName);

   ui->lePath->setText(prefix);
   ui->lePath->setReadOnly(true);
   ui->chSquash->setVisible(false);

   setWindowTitle(tr("Configure subtree"));
}

AddSubtreeDlg::AddSubtreeDlg(const QString &prefix, const QString &url, const QString &reference,
                             const QSharedPointer<GitBase> &git, QWidget *parent)
   : AddSubtreeDlg(prefix, git, parent)
{
   disconnect(ui->leUrl, &QLineEdit::editingFinished, this, &AddSubtreeDlg::proposeName);

   ui->leReference->setText(reference);
   ui->leUrl->setText(url);
}

AddSubtreeDlg::~AddSubtreeDlg()
{
   delete ui;
}

void AddSubtreeDlg::accept()
{
   const auto subtreeName = ui->lePath->text();
   const auto subtreeUrl = ui->leUrl->text();
   const auto subtreeRef = ui->leReference->text();

   QScopedPointer<GitSubtree> git(new GitSubtree(mGit));

   if (subtreeName.isEmpty() || subtreeUrl.isEmpty() || subtreeRef.isEmpty())
   {
      QMessageBox::warning(
          this, tr("Invalid fields"),
          tr("The information provided is incorrect. Please fix the URL and/or the name and submit again."));
   }
   else
   {
      QLog_Debug("UI", "Adding a subtree");
      GitQlientSettings settings(mGit->getGitDir());
      GitExecResult ret;

      for (auto i = 0;; ++i)
      {
         const auto repo = settings.localValue(QString("Subtrees/%1.prefix").arg(i)).toString();

         if (repo == subtreeName)
         {
            settings.setLocalValue(QString("Subtrees/%1.url").arg(i), subtreeUrl);
            settings.setLocalValue(QString("Subtrees/%1.ref").arg(i), subtreeRef);

            ret = git->add(subtreeUrl, subtreeRef, subtreeName, ui->chSquash->isChecked());
         }
         else if (repo.isEmpty())
         {
            settings.setLocalValue(QString("Subtrees/%1.prefix").arg(i), subtreeName);
            settings.setLocalValue(QString("Subtrees/%1.url").arg(i), subtreeUrl);
            settings.setLocalValue(QString("Subtrees/%1.ref").arg(i), subtreeRef);

            QLog_Trace("Git", QString("Updating subtree info: {%1}").arg(subtreeName));
         }
      }

      if (ret.success)
         QDialog::accept();
      else
         QMessageBox::warning(this, tr("Error when adding a subtree."), ret.output);
   }
}

void AddSubtreeDlg::proposeName()
{
   auto url = ui->leUrl->text();
   QString proposedName;

   if (url.startsWith("https"))
   {
      url.remove("https://");
      const auto fields = url.split("/");

      if (fields.count() > 1)
      {
         proposedName = fields.at(2);
         proposedName = proposedName.split(".").constFirst();
      }
   }
   else if (url.contains("@"))
   {
      const auto fields = url.split(":");

      if (fields.count() > 0)
      {
         proposedName = fields.constLast().split("/").constLast();
         proposedName = proposedName.split(".").constFirst();
      }
   }

   ui->lePath->setText(proposedName);
}
