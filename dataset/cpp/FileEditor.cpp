#include "FileEditor.h"

#include <FileDiffEditor.h>
#include <GitQlientSettings.h>
#include <GitQlientStyles.h>
#include <Highlighter.h>

#include <QFile>
#include <QMessageBox>
#include <QVBoxLayout>

FileEditor::FileEditor(bool highlighter, QWidget *parent)
   : QFrame(parent)
   , mFileEditor(new FileDiffEditor())
{
   if (highlighter)
      mHighlighter = new Highlighter(mFileEditor->document());

   const auto layout = new QVBoxLayout(this);
   layout->setContentsMargins(QMargins());
   layout->setSpacing(0);
   layout->addWidget(mFileEditor);
}

FileEditor::~FileEditor()
{
   if (mFileEditor)
      delete mFileEditor;
}

void FileEditor::editFile(const QString &fileName)
{
   mFileName = fileName;

   QFile f(mFileName);

   if (f.open(QIODevice::ReadOnly))
   {
      mLoadedContent = QString::fromUtf8(f.readAll());
      f.close();
   }

   mFileEditor->loadDiff(mLoadedContent, {});

   isEditing = true;
}

void FileEditor::finishEdition()
{
   if (isEditing)
   {
      const auto currentContent = mFileEditor->toPlainText();
      QFile f(mFileName);
      QString fileContent;

      if (f.open(QIODevice::ReadOnly))
      {
         fileContent = QString::fromUtf8(f.readAll());
         f.close();
      }

      if (currentContent != fileContent)
      {
         const auto alert = new QMessageBox(QMessageBox::Question, tr("Unsaved changes"),
                                            tr("The current text was modified. Do you want to save the changes?"));
         alert->setStyleSheet(GitQlientStyles::getInstance()->getStyles());
         alert->addButton(tr("Discard"), QMessageBox::ButtonRole::RejectRole);
         alert->addButton(tr("Save"), QMessageBox::ButtonRole::AcceptRole);

         if (alert->exec() == QMessageBox::Accepted)
            saveTextInFile(currentContent);
      }

      isEditing = false;

      emit signalEditionClosed();
   }
}

void FileEditor::saveFile()
{
   const auto currentContent = mFileEditor->toPlainText();

   saveTextInFile(currentContent);
}

void FileEditor::changeFontSize()
{
   GitQlientSettings settings;
   const auto fontSize = settings.globalValue("FileDiffView/FontSize", 8).toInt();

   auto font = mFileEditor->font();
   font.setPointSize(fontSize);

   const auto cursor = mFileEditor->textCursor();
   mFileEditor->selectAll();
   mFileEditor->setFont(font);
   mFileEditor->setTextCursor(cursor);
}

void FileEditor::saveTextInFile(const QString &content) const
{
   QFile f(mFileName);

   if (f.open(QIODevice::WriteOnly))
   {
      f.write(content.toUtf8());
      f.close();
   }
}
