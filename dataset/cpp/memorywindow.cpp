#include "debuggershortcuts.h"
#include "memorywindow.h"
#include "memorywidget.h"

#include "ui_memorywindow.h"

MemoryWindow::MemoryWindow(DebuggerShortcuts *debuggerShortcuts,
                           QWidget *parent) :
   QWidget(parent),
   ui(new Ui::MemoryWindow { })
{
   ui->setupUi(this);
   ui->widget->setAddressRange(0, 0xFFFFFFFF);
   ui->widget->navigateToAddress(0x02000000);
   ui->widget->setBytesPerLine(16);

   ui->widget->addAction(debuggerShortcuts->navigateBackward);
   ui->widget->addAction(debuggerShortcuts->navigateForward);
   ui->widget->addAction(debuggerShortcuts->navigateToAddress);
}

MemoryWindow::~MemoryWindow()
{
   delete ui;
}

void
MemoryWindow::navigateToAddress(uint32_t address)
{
   ui->widget->navigateToAddress(address);
}

void
MemoryWindow::navigateForward()
{
   ui->widget->navigateForward();
}

void
MemoryWindow::navigateBackward()
{
   ui->widget->navigateBackward();
}

void
MemoryWindow::addressChanged()
{
   auto text = ui->lineEditAddress->text();
   auto value = 0ull;
   auto valid = false;

   if (text.startsWith("0x")) {
      value = text.toULongLong(&valid, 0);
   } else {
      value = text.toULongLong(&valid, 16);
   }

   if (valid) {
      ui->widget->navigateToAddress(value);
      ui->widget->setFocus();
   }
}

void
MemoryWindow::columnsChanged()
{
   if (ui->comboBoxColumns->currentIndex() == 0) {
      ui->widget->setAutoBytesPerLine(true);
   } else {
      auto valid = false;
      auto value = ui->comboBoxColumns->currentText().toInt(&valid, 0);
      if (valid) {
         ui->widget->setBytesPerLine(value);
      }
   }
}
