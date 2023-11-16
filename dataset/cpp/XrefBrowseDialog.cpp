#include "XrefBrowseDialog.h"
#include "ui_XrefBrowseDialog.h"
#include "StringUtil.h"
#include "MiscUtil.h"
#include "MenuBuilder.h"

XrefBrowseDialog::XrefBrowseDialog(QWidget* parent) :
    QDialog(parent),
    ui(new Ui::XrefBrowseDialog)
{
    ui->setupUi(this);
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint | Qt::MSWindowsFixedSizeDialogHint);
    setWindowIcon(DIcon("xrefs"));
    setModal(false);
    mXrefInfo.refcount = 0;
    ui->listWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(Bridge::getBridge(), SIGNAL(dbgStateChanged(DBGSTATE)), this, SLOT(onDebuggerClose(DBGSTATE)));

    setupContextMenu();
}

QString XrefBrowseDialog::GetFunctionSymbol(duint addr)
{
    QString line;
    char clabel[MAX_LABEL_SIZE] = "";

    DbgGetLabelAt(addr, SEG_DEFAULT, clabel);
    if(*clabel)
        line = QString(clabel);
    else
    {
        duint start;
        if(DbgFunctionGet(addr, &start, nullptr) && DbgGetLabelAt(start, SEG_DEFAULT, clabel) && start != addr)
            line = QString("%1+%2").arg(clabel).arg(ToHexString(addr - start));
        else
            line = QString("%1").arg(ToHexString(addr));
    }

    return line;
}

void XrefBrowseDialog::setup(duint address, GotoFunction gotoFunction)
{
    if(mXrefInfo.refcount)
    {
        BridgeFree(mXrefInfo.references);
        mXrefInfo.refcount = 0;
    }
    mAddress = address;
    mGotoFunction = std::move(gotoFunction);
    mPrevSelectionSize = 0;
    ui->listWidget->clear();
    if(DbgXrefGet(address, &mXrefInfo))
    {
        std::vector<XREF_RECORD> data;
        for(duint i = 0; i < mXrefInfo.refcount; i++)
            data.push_back(mXrefInfo.references[i]);

        std::sort(data.begin(), data.end(), [](const XREF_RECORD A, const XREF_RECORD B)
        {
            if(A.type != B.type)
                return (A.type < B.type);

            return (A.addr < B.addr);
        });

        for(duint i = 0; i < mXrefInfo.refcount; i++)
            mXrefInfo.references[i] = data[i];

        data.clear();
        char disasm[GUI_MAX_DISASSEMBLY_SIZE] = "";
        setWindowTitle(QString(tr("xrefs at <%1>")).arg(GetFunctionSymbol(address)));
        for(duint i = 0; i < mXrefInfo.refcount; i++)
        {
            if(GuiGetDisassembly(mXrefInfo.references[i].addr, disasm))
                ui->listWidget->addItem(disasm);
            else
                ui->listWidget->addItem("???");
        }
        ui->listWidget->setCurrentRow(0);
    }
    ui->listWidget->setFocus();
}

void XrefBrowseDialog::setupContextMenu()
{
    mMenu = new MenuBuilder(this);
    mMenu->addAction(makeAction(DIcon("breakpoint_toggle"), tr("Toggle &Breakpoint"), SLOT(breakpointSlot())));
    //Breakpoint (hardware access) menu
    auto hardwareAccessMenu = makeMenu(DIcon("breakpoint_access"), tr("Hardware, Access"));
    hardwareAccessMenu->addAction(makeAction(DIcon("breakpoint_byte"), tr("&Byte"), SLOT(hardwareAccess1Slot())));
    hardwareAccessMenu->addAction(makeAction(DIcon("breakpoint_word"), tr("&Word"), SLOT(hardwareAccess2Slot())));
    hardwareAccessMenu->addAction(makeAction(DIcon("breakpoint_dword"), tr("&Dword"), SLOT(hardwareAccess4Slot())));
#ifdef _WIN64
    hardwareAccessMenu->addAction(makeAction(DIcon("breakpoint_qword"), tr("&Qword"), SLOT(hardwareAccess8Slot())));
#endif //_WIN64

    //Breakpoint (hardware write) menu
    auto hardwareWriteMenu = makeMenu(DIcon("breakpoint_write"), tr("Hardware, Write"));
    hardwareWriteMenu->addAction(makeAction(DIcon("breakpoint_byte"), tr("&Byte"), SLOT(hardwareWrite1Slot())));
    hardwareWriteMenu->addAction(makeAction(DIcon("breakpoint_word"), tr("&Word"), SLOT(hardwareWrite2Slot())));
    hardwareWriteMenu->addAction(makeAction(DIcon("breakpoint_dword"), tr("&Dword"), SLOT(hardwareWrite4Slot())));
#ifdef _WIN64
    hardwareWriteMenu->addAction(makeAction(DIcon("breakpoint_qword"), tr("&Qword"), SLOT(hardwareAccess8Slot())));
#endif //_WIN64

    //Breakpoint (remove hardware)
    auto hardwareRemove = makeAction(DIcon("breakpoint_remove"), tr("Remove &Hardware"), SLOT(hardwareRemoveSlot()));

    //Breakpoint (memory access) menu
    auto memoryAccessMenu = makeMenu(DIcon("breakpoint_memory_access"), tr("Memory, Access"));
    memoryAccessMenu->addAction(makeAction(DIcon("breakpoint_memory_singleshoot"), tr("&Singleshoot"), SLOT(memoryAccessSingleshootSlot())));
    memoryAccessMenu->addAction(makeAction(DIcon("breakpoint_memory_restore_on_hit"), tr("&Restore on hit"), SLOT(memoryAccessRestoreSlot())));

    //Breakpoint (memory write) menu
    auto memoryWriteMenu = makeMenu(DIcon("breakpoint_memory_write"), tr("Memory, Write"));
    memoryWriteMenu->addAction(makeAction(DIcon("breakpoint_memory_singleshoot"), tr("&Singleshoot"), SLOT(memoryWriteSingleshootSlot())));
    memoryWriteMenu->addAction(makeAction(DIcon("breakpoint_memory_restore_on_hit"), tr("&Restore on hit"), SLOT(memoryWriteRestoreSlot())));

    //Breakpoint (remove memory) menu
    auto memoryRemove = makeAction(DIcon("breakpoint_remove"), tr("Remove &Memory"), SLOT(memoryRemoveSlot()));

    //Breakpoint menu
    auto breakpointMenu = new MenuBuilder(this);

    //Breakpoint menu
    breakpointMenu->addBuilder(new MenuBuilder(this, [ = ](QMenu * menu)
    {
        duint selectedAddr = mXrefInfo.references[ui->listWidget->currentRow()].addr;
        if(DbgGetBpxTypeAt(selectedAddr) & bp_hardware) //hardware breakpoint set
        {
            menu->addAction(hardwareRemove);
        }
        else //memory breakpoint not set
        {
            menu->addMenu(hardwareAccessMenu);
            menu->addMenu(hardwareWriteMenu);
        }

        menu->addSeparator();

        if(DbgGetBpxTypeAt(selectedAddr) & bp_memory) //memory breakpoint set
        {
            menu->addAction(memoryRemove);
        }
        else //memory breakpoint not set
        {
            menu->addMenu(memoryAccessMenu);
            menu->addMenu(memoryWriteMenu);
        }
        return true;
    }));
    mMenu->addMenu(makeMenu(DIcon("breakpoint"), tr("Brea&kpoint")), breakpointMenu);
    mMenu->addAction(makeAction(DIcon("breakpoint_toggle"), tr("Toggle breakpoints on all xrefs"), SLOT(breakpointAllSlot())));
    auto mCopyMenu = new MenuBuilder(mMenu);
    mCopyMenu->addAction(makeAction(tr("Selected xref"), SLOT(copyThisSlot())));
    mCopyMenu->addAction(makeAction(tr("All xrefs"), SLOT(copyAllSlot())));
    mMenu->addMenu(makeMenu(DIcon("copy"), tr("Copy")), mCopyMenu);
    mMenu->loadFromConfig();
}

void XrefBrowseDialog::changeAddress(duint address)
{
    mGotoFunction(address);
}

XrefBrowseDialog::~XrefBrowseDialog()
{
    delete ui;
    if(mXrefInfo.refcount)
        BridgeFree(mXrefInfo.references);
}

void XrefBrowseDialog::on_listWidget_itemDoubleClicked(QListWidgetItem*)
{
    accept();
}

void XrefBrowseDialog::on_listWidget_itemSelectionChanged()
{
    if(ui->listWidget->selectedItems().size() != mPrevSelectionSize)
    {
        duint address;
        if(mPrevSelectionSize == 0)
            address = mXrefInfo.references[ui->listWidget->currentRow()].addr;
        else
            address = mAddress;

        changeAddress(address);
    }
    mPrevSelectionSize = ui->listWidget->selectedItems().size();
}

void XrefBrowseDialog::on_listWidget_currentRowChanged(int row)
{
    if(ui->listWidget->selectedItems().size() != 0)
    {
        duint address = mXrefInfo.references[row].addr;
        changeAddress(address);
    }
}

void XrefBrowseDialog::on_XrefBrowseDialog_rejected()
{
    if(DbgIsDebugging())
        mGotoFunction(mAddress);
}

void XrefBrowseDialog::on_listWidget_itemClicked(QListWidgetItem*)
{
    on_listWidget_currentRowChanged(ui->listWidget->currentRow());
}

void XrefBrowseDialog::on_listWidget_customContextMenuRequested(const QPoint & pos)
{
    QMenu menu(this);
    mMenu->build(&menu);
    menu.exec(ui->listWidget->mapToGlobal(pos));
}

/**
 * @brief XrefBrowseDialog::onDebuggerClose Close this dialog when the debuggee stops.
 * @param state The argument passed in representing the debugger state.
 */
void XrefBrowseDialog::onDebuggerClose(DBGSTATE state)
{
    if(state == stopped)
        emit close();
}

void XrefBrowseDialog::breakpointSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    if(DbgGetBpxTypeAt(mXrefInfo.references[ui->listWidget->currentRow()].addr) & bp_normal)
        DbgCmdExec(QString("bc " + addr_text));
    else
        DbgCmdExec(QString("bp " + addr_text));
}

void XrefBrowseDialog::hardwareAccess1Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", r, 1"));
}

void XrefBrowseDialog::hardwareAccess2Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", r, 2"));
}

void XrefBrowseDialog::hardwareAccess4Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", r, 4"));
}

void XrefBrowseDialog::hardwareAccess8Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", r, 8"));
}

void XrefBrowseDialog::hardwareWrite1Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", w, 1"));
}

void XrefBrowseDialog::hardwareWrite2Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", w, 2"));
}

void XrefBrowseDialog::hardwareWrite4Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", w, 4"));
}

void XrefBrowseDialog::hardwareWrite8Slot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphws " + addr_text + ", w, 8"));
}

void XrefBrowseDialog::hardwareRemoveSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bphwc " + addr_text));
}

void XrefBrowseDialog::memoryAccessSingleshootSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bpm " + addr_text + ", 0, a"));
}

void XrefBrowseDialog::memoryAccessRestoreSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bpm " + addr_text + ", 1, a"));
}

void XrefBrowseDialog::memoryWriteSingleshootSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bpm " + addr_text + ", 0, w"));
}

void XrefBrowseDialog::memoryWriteRestoreSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bpm " + addr_text + ", 1, w"));
}

void XrefBrowseDialog::memoryRemoveSlot()
{
    QString addr_text = ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr);
    DbgCmdExec(QString("bpmc " + addr_text));
}

void XrefBrowseDialog::copyThisSlot()
{
    Bridge::CopyToClipboard(ToPtrString(mXrefInfo.references[ui->listWidget->currentRow()].addr) + " " + ui->listWidget->selectedItems()[0]->text());
}

void XrefBrowseDialog::breakpointAllSlot()
{
    for(int i = 0; i < ui->listWidget->count(); i++)
    {
        QString addr_text = ToPtrString(mXrefInfo.references[i].addr);
        if(DbgGetBpxTypeAt(mXrefInfo.references[i].addr) & bp_normal)
            DbgCmdExec(QString("bc " + addr_text));
        else
            DbgCmdExec(QString("bp " + addr_text));
    }
}

void XrefBrowseDialog::copyAllSlot()
{
    QString temp;
    for(int i = 0; i < ui->listWidget->count(); i++)
    {
        temp.append(ToPtrString(mXrefInfo.references[i].addr) + " ");
        temp.append(ui->listWidget->selectedItems()[0]->text());
        temp.append("\r\n");
    }
    Bridge::CopyToClipboard(temp);
}
