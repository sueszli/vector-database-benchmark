#include "debuggerwindow.h"
#include "ui_debuggerwindow.h"

#include "debugdata.h"

#include "breakpointswindow.h"
#include "disassemblywindow.h"
#include "functionswindow.h"
#include "jitprofilingwindow.h"
#include "memorywindow.h"
#include "registerswindow.h"
#include "segmentswindow.h"
#include "stackwindow.h"
#include "threadswindow.h"
#include "voiceswindow.h"

#include <DockAreaWidget.h>
#include <DockWidgetTab.h>
#include <QInputDialog>
#include <QMessageBox>
#include <QModelIndex>
#include <QTimer>

#include <libdecaf/decaf.h>
#include <libdecaf/decaf_config.h>
#include <libdecaf/decaf_debug_api.h>
#include <libgpu/gpu_config.h>

DebuggerWindow::DebuggerWindow(QWidget *parent) :
   QMainWindow(parent),
   ui(new Ui::DebuggerWindow { })
{
   connect(qApp, &QApplication::focusChanged, this, &DebuggerWindow::focusChanged);
   ui->setupUi(this);

   mDebuggerShortcuts.navigateBackward = ui->actionNavigateBackward;
   mDebuggerShortcuts.navigateForward = ui->actionNavigateForward;
   mDebuggerShortcuts.navigateToAddress = ui->actionNavigateToAddress;
   mDebuggerShortcuts.navigateToOperand = ui->actionNavigateToOperand;
   mDebuggerShortcuts.toggleBreakpoint = ui->actionToggleBreakpoint;

   mDockManager = new ads::CDockManager { this };

   mDebugData = new DebugData { this };
   connect(mDebugData, &DebugData::entry, this, &DebuggerWindow::onEntry);
   connect(mDebugData, &DebugData::pm4CaptureStateChanged, this, &DebuggerWindow::pm4CaptureStateChanged);
   connect(mDebugData, &DebugData::executionStateChanged, this, &DebuggerWindow::executionStateChanged);
   connect(mDebugData, &DebugData::activeThreadIndexChanged, this, &DebuggerWindow::activeThreadChanged);

   mBreakpointsDockWidget = new ads::CDockWidget{ tr("Breakpoints") };
   mBreakpointsWindow = new BreakpointsWindow { mBreakpointsDockWidget };
   mBreakpointsWindow->setDebugData(mDebugData);
   mBreakpointsDockWidget->setWidget(mBreakpointsWindow, ads::CDockWidget::ForceNoScrollArea);

   mDisassemblyDockWidget = new ads::CDockWidget { tr("Disassembly") };
   mDisassemblyWindow = new DisassemblyWindow { &mDebuggerShortcuts, mDisassemblyDockWidget };
   mDisassemblyWindow->setDebugData(mDebugData);
   mDisassemblyDockWidget->setWidget(mDisassemblyWindow, ads::CDockWidget::ForceNoScrollArea);

   mFunctionsDockWidget = new ads::CDockWidget { tr("Functions") };
   mFunctionsWindow = new FunctionsWindow { mFunctionsDockWidget };
   mFunctionsWindow->setDebugData(mDebugData);
   mFunctionsDockWidget->setWidget(mFunctionsWindow, ads::CDockWidget::ForceNoScrollArea);

   mJitProfilingDockWidget = new ads::CDockWidget { tr("JIT Profiling") };
   mJitProfilingWindow = new JitProfilingWindow { mJitProfilingDockWidget };
   mJitProfilingWindow->setDebugData(mDebugData);
   mJitProfilingDockWidget->setWidget(mJitProfilingWindow, ads::CDockWidget::ForceNoScrollArea);

   mMemoryDockWidget = new ads::CDockWidget { tr("Memory") };
   mMemoryWindow = new MemoryWindow { &mDebuggerShortcuts, mRegistersDockWidget };
   mMemoryDockWidget->setWidget(mMemoryWindow, ads::CDockWidget::ForceNoScrollArea);

   mRegistersDockWidget = new ads::CDockWidget { tr("Registers") };
   mRegistersWindow = new RegistersWindow { mRegistersDockWidget };
   mRegistersWindow->setDebugData(mDebugData);
   mRegistersDockWidget->setWidget(mRegistersWindow, ads::CDockWidget::ForceScrollArea);

   mSegmentsDockWidget = new ads::CDockWidget { tr("Segments") };
   mSegmentsWindow = new SegmentsWindow { mSegmentsDockWidget };
   mSegmentsWindow->setDebugData(mDebugData);
   mSegmentsDockWidget->setWidget(mSegmentsWindow, ads::CDockWidget::ForceNoScrollArea);

   mStackDockWidget = new ads::CDockWidget { tr("Stack") };
   mStackWindow = new StackWindow { &mDebuggerShortcuts, mStackDockWidget };
   mStackWindow->setDebugData(mDebugData);
   mStackDockWidget->setWidget(mStackWindow, ads::CDockWidget::ForceNoScrollArea);

   mThreadsDockWidget = new ads::CDockWidget { tr("Threads") };
   mThreadsWindow = new ThreadsWindow { mThreadsDockWidget };
   mThreadsWindow->setDebugData(mDebugData);
   mThreadsDockWidget->setWidget(mThreadsWindow, ads::CDockWidget::ForceNoScrollArea);

   mVoicesDockWidget = new ads::CDockWidget { tr("Voices") };
   mVoicesWindow = new VoicesWindow { mVoicesDockWidget };
   mVoicesWindow->setDebugData(mDebugData);
   mVoicesDockWidget->setWidget(mVoicesWindow, ads::CDockWidget::ForceNoScrollArea);

   connect(mBreakpointsWindow, &BreakpointsWindow::navigateToTextAddress, this, &DebuggerWindow::gotoTextAddress);
   connect(mFunctionsWindow, &FunctionsWindow::navigateToTextAddress, this, &DebuggerWindow::gotoTextAddress);
   connect(mJitProfilingWindow, &JitProfilingWindow::navigateToTextAddress, this, &DebuggerWindow::gotoTextAddress);
   connect(mSegmentsWindow, &SegmentsWindow::navigateToDataAddress, this, &DebuggerWindow::gotoDataAddress);
   connect(mSegmentsWindow, &SegmentsWindow::navigateToTextAddress, this, &DebuggerWindow::gotoTextAddress);
   connect(mStackWindow, &StackWindow::navigateToTextAddress, this, &DebuggerWindow::gotoTextAddress);

   // Setup default dock layout
   {
      auto mainDockArea = mDockManager->addDockWidgetTab(ads::CenterDockWidgetArea, mDisassemblyDockWidget);
      mDockManager->addDockWidgetTabToArea(mMemoryDockWidget, mainDockArea);
      mDockManager->addDockWidgetTabToArea(mThreadsDockWidget, mainDockArea);
      mDockManager->addDockWidgetTabToArea(mSegmentsDockWidget, mainDockArea);
      mDockManager->addDockWidgetTabToArea(mVoicesDockWidget, mainDockArea);
      mDockManager->addDockWidgetTabToArea(mJitProfilingDockWidget, mainDockArea);
      mDockManager->addDockWidgetTabToArea(mBreakpointsDockWidget, mainDockArea);
      mDisassemblyDockWidget->dockAreaWidget()->setCurrentDockWidget(mDisassemblyDockWidget);

      auto sizePolicy = mainDockArea->sizePolicy();
      sizePolicy.setHorizontalStretch(1);
      sizePolicy.setVerticalStretch(1);
      mainDockArea->setSizePolicy(sizePolicy);
   }

   {
      auto leftDockArea = mDockManager->addDockWidget(ads::LeftDockWidgetArea, mFunctionsDockWidget);
   }

   {
      auto rightDockArea = mDockManager->addDockWidget(ads::RightDockWidgetArea, mRegistersDockWidget);
      mDockManager->addDockWidget(ads::BottomDockWidgetArea, mStackDockWidget, rightDockArea);
   }

   // Setup shortcuts
   mBreakpointsDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+b"));
   mDisassemblyDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+i"));
   mMemoryDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+m"));
   mRegistersDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+r"));
   mSegmentsDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+s"));
   mStackDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+e"));
   mThreadsDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+t"));
   mVoicesDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+p"));
   mJitProfilingDockWidget->toggleViewAction()->setShortcut(tr("Ctrl+j"));

   // Add view toggles to menu
   ui->menuView->addAction(mBreakpointsDockWidget->toggleViewAction());
   ui->menuView->addAction(mDisassemblyDockWidget->toggleViewAction());
   ui->menuView->addAction(mFunctionsDockWidget->toggleViewAction());
   ui->menuView->addAction(mThreadsDockWidget->toggleViewAction());
   ui->menuView->addAction(mSegmentsDockWidget->toggleViewAction());
   ui->menuView->addAction(mVoicesDockWidget->toggleViewAction());
   ui->menuView->addAction(mRegistersDockWidget->toggleViewAction());
   ui->menuView->addAction(mStackDockWidget->toggleViewAction());
   ui->menuView->addAction(mMemoryDockWidget->toggleViewAction());
   ui->menuView->addAction(mJitProfilingDockWidget->toggleViewAction());

   // Create a timer to poll debug data from decaf
   mUpdateModelTimer = new QTimer { this };
   connect(mUpdateModelTimer, SIGNAL(timeout()), this, SLOT(updateModel()));
   mUpdateModelTimer->start(100);
}

DebuggerWindow::~DebuggerWindow()
{
   delete ui;
}

void
DebuggerWindow::updateModel()
{
   if (!mDebugData->update()) {
      return;
   }
}

void
DebuggerWindow::gotoTextAddress(DebugData::VirtualAddress address)
{
   mDisassemblyWindow->navigateToAddress(address);
   mDisassemblyDockWidget->dockAreaWidget()->setCurrentDockWidget(mDisassemblyDockWidget);
}

void
DebuggerWindow::gotoDataAddress(DebugData::VirtualAddress address)
{
   mMemoryWindow->navigateToAddress(address);
   mMemoryDockWidget->dockAreaWidget()->setCurrentDockWidget(mMemoryDockWidget);
}

void
DebuggerWindow::onEntry()
{
   auto textStartAddress = 0x02000000u;
   auto dataStartAddress = 0x10000000u;

   if (mDebugData->loadedModule().textAddr) {
      textStartAddress = mDebugData->loadedModule().textAddr;
   }

   if (mDebugData->loadedModule().dataAddr) {
      dataStartAddress = mDebugData->loadedModule().dataAddr;
   }

   if (!mDebugData->paused()) {
      gotoDataAddress(dataStartAddress);
      gotoTextAddress(textStartAddress);
   }
}

void
DebuggerWindow::debugPause()
{
   decaf::debug::pause();
}

void
DebuggerWindow::debugResume()
{
   decaf::debug::resume();
}

void
DebuggerWindow::debugStepOver()
{
   if (auto activeThread = mDebugData->activeThread()) {
      decaf::debug::stepOver(activeThread->coreId);
   }
}

void
DebuggerWindow::debugStepInto()
{
   if (auto activeThread = mDebugData->activeThread()) {
      decaf::debug::stepInto(activeThread->coreId);
   }
}

void
DebuggerWindow::debugToggleBreakpoint()
{
   mDisassemblyWindow->toggleBreakpointUnderCursor();
}

void
DebuggerWindow::setHleTraceEnabled(bool enabled)
{
   auto config = *decaf::config();
   config.log.hle_trace = enabled;
   decaf::setConfig(config);
}

void
DebuggerWindow::setGpuShaderBinaryDumpOnly(bool enabled)
{
   auto config = *gpu::config();
   config.debug.dump_shader_binaries_only = enabled;
   gpu::setConfig(config);
}

void
DebuggerWindow::setGpuShaderDumpEnabled(bool enabled)
{
   auto config = *gpu::config();
   config.debug.dump_shaders = enabled;
   gpu::setConfig(config);
}

void
DebuggerWindow::setGx2ShaderDumpEnabled(bool enabled)
{
   auto config = *decaf::config();
   config.gx2.dump_shaders = enabled;
   decaf::setConfig(config);
}

void
DebuggerWindow::setGx2TextureDumpEnabled(bool enabled)
{
   auto config = *decaf::config();
   config.gx2.dump_textures = enabled;
   decaf::setConfig(config);
}

void
DebuggerWindow::setPm4TraceEnabled(bool enabled)
{
   if (enabled) {
      decaf::debug::pm4CaptureBegin();
      ui->actionPm4CaptureNextFrame->setEnabled(false);
      ui->actionPm4TraceEnabled->setEnabled(false);
   } else {
      decaf::debug::pm4CaptureEnd();
   }
}

void
DebuggerWindow::pm4CaptureNextFrame()
{
   decaf::debug::pm4CaptureNextFrame();
   ui->actionPm4CaptureNextFrame->setEnabled(false);
   ui->actionPm4TraceEnabled->setEnabled(false);
}

void
DebuggerWindow::pm4CaptureStateChanged(decaf::debug::Pm4CaptureState state)
{
   if (state == decaf::debug::Pm4CaptureState::Disabled) {
      ui->actionPm4CaptureNextFrame->setEnabled(true);
      ui->actionPm4TraceEnabled->setEnabled(true);
      ui->actionPm4TraceEnabled->setChecked(false);
   } else if (state == decaf::debug::Pm4CaptureState::Enabled) {
      ui->actionPm4CaptureNextFrame->setEnabled(false);
      ui->actionPm4TraceEnabled->setEnabled(true);
      ui->actionPm4TraceEnabled->setChecked(true);
   } else {
      ui->actionPm4CaptureNextFrame->setEnabled(false);
      ui->actionPm4TraceEnabled->setEnabled(false);
      ui->actionPm4TraceEnabled->setChecked(true);
   }
}

void
DebuggerWindow::activeThreadChanged()
{
   auto activeThread = mDebugData->activeThread();
   if (!activeThread) {
      return;
   }

   if (mDebugData->paused()) {
      gotoTextAddress(activeThread->nia);
   }
}

void
DebuggerWindow::executionStateChanged(bool paused,
                                      int pauseInitiatorCoreId,
                                      decaf::debug::VirtualAddress pauseNia)
{
   if (!paused) {
      ui->actionResume->setEnabled(false);
      ui->actionPause->setEnabled(true);
      return;
   }

   ui->actionPause->setEnabled(false);

   auto canResume = !decaf::stopping();
   ui->actionResume->setEnabled(canResume);
   ui->actionStepInto->setEnabled(canResume);
   ui->actionStepOver->setEnabled(canResume);

   // Set active thread to the one that initiated pause - this will lead to
   // activeThreadChanged being called and thus follow pauseNia implicitly
   auto &threads = mDebugData->threads();
   for (auto i = 0u; i < threads.size(); ++i) {
      if (threads[i].coreId == pauseInitiatorCoreId) {
         mDebugData->setActiveThreadIndex(i);
      }
   }
}

void
DebuggerWindow::navigateBackward()
{
   if (isDockWidgetFocused(mDisassemblyWindow)) {
      mDisassemblyWindow->navigateBackward();
   } else if (isDockWidgetFocused(mMemoryWindow)) {
      mMemoryWindow->navigateBackward();
   } else if (isDockWidgetFocused(mStackWindow)) {
      mStackWindow->navigateBackward();
   }
}

void
DebuggerWindow::navigateForward()
{
   if (isDockWidgetFocused(mDisassemblyWindow)) {
      mDisassemblyWindow->navigateForward();
   } else if (isDockWidgetFocused(mMemoryWindow)) {
      mMemoryWindow->navigateForward();
   } else if (isDockWidgetFocused(mStackWindow)) {
      mStackWindow->navigateForward();
   }
}

void
DebuggerWindow::navigateAddress()
{
   auto address = 0u;
   while (true) {
      auto ok = false;
      auto text = QInputDialog::getText(this, tr("Navigate to Address"),
                                        tr("Address:"), QLineEdit::Normal, {},
                                        &ok);
      if (!ok) {
         return; // Cancelled
      }

      address = text.toUInt(&ok, 16);
      if (!ok) {
         QMessageBox::warning(this, "Navigate to address", "Invalid address");
         continue; // Try again
      }

      break;
   }

   if (isDockWidgetFocused(mDisassemblyWindow)) {
      mDisassemblyWindow->navigateToAddress(address);
   } else if (isDockWidgetFocused(mMemoryWindow)) {
      mMemoryWindow->navigateToAddress(address);
   } else if (isDockWidgetFocused(mStackWindow)) {
      mStackWindow->navigateToAddress(address);
   }
}

void
DebuggerWindow::navigateOperand()
{
   if (isDockWidgetFocused(mDisassemblyWindow)) {
      mDisassemblyWindow->navigateOperand();
   } else if (isDockWidgetFocused(mStackWindow)) {
      mStackWindow->navigateOperand();
   }
}

void
DebuggerWindow::focusChanged(QWidget *old, QWidget *now)
{
   const auto updateActionState =
      [](QAction *action) {
         for (auto widget : action->associatedWidgets()) {
            if (widget->hasFocus()) {
               action->setEnabled(true);
               return;
            }
         }

         action->setEnabled(false);
       };

   updateActionState(ui->actionToggleBreakpoint);
   updateActionState(ui->actionNavigateForward);
   updateActionState(ui->actionNavigateBackward);
   updateActionState(ui->actionNavigateToAddress);
   updateActionState(ui->actionNavigateToOperand);
}

bool
DebuggerWindow::isDockWidgetFocused(QWidget *dockWidget)
{
   auto widget = QApplication::focusWidget();
   while (widget) {
      if (widget == dockWidget) {
         return true;
      }

      widget = widget->parentWidget();
   }

   return false;
}
