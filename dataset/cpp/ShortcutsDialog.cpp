#include "ShortcutsDialog.h"
#include "ui_ShortcutsDialog.h"

#include <QMessageBox>

ShortcutsDialog::ShortcutsDialog(QWidget* parent) : QDialog(parent), ui(new Ui::ShortcutsDialog)
{
    ui->setupUi(this);
    //set window flags
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint | Qt::MSWindowsFixedSizeDialogHint);
    setModal(true);

    // x64 has no model-view-controler pattern
    QStringList tblHeader;
    tblHeader << tr("Action") << tr("Hotkey");

    currentRow = 0;

    ui->tblShortcuts->setColumnCount(2);
    ui->tblShortcuts->verticalHeader()->setVisible(false);
    ui->tblShortcuts->setHorizontalHeaderLabels(tblHeader);
    ui->tblShortcuts->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tblShortcuts->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->tblShortcuts->setSelectionMode(QAbstractItemView::SingleSelection);
    ui->tblShortcuts->setShowGrid(false);
#if QT_VERSION < QT_VERSION_CHECK(5,0,0)
    ui->tblShortcuts->horizontalHeader()->setResizeMode(0, QHeaderView::Stretch);
#else
    ui->tblShortcuts->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
#endif
    ui->tblShortcuts->verticalHeader()->setDefaultSectionSize(15);
    showShortcutsFiltered(QString());

    connect(ui->tblShortcuts, SIGNAL(itemSelectionChanged()), this, SLOT(syncTextfield()));
    connect(ui->shortcutEdit, SIGNAL(askForSave()), this, SLOT(updateShortcut()));
    connect(this, SIGNAL(rejected()), this, SLOT(rejectedSlot()));
    connect(ui->filterEdit, SIGNAL(textChanged(const QString &)), this, SLOT(on_FilterTextChanged(const QString &)));
}

void ShortcutsDialog::showShortcutsFiltered(const QString & actionName)
{
    QMap<QString, Configuration::Shortcut> shorcutsToShow;
    filterShortcutsByName(actionName, shorcutsToShow);
    ui->tblShortcuts->clearContents();
    ui->tblShortcuts->setRowCount(shorcutsToShow.count());
    currentRow = 0;

    int row = 0;
    for(auto shortcut : shorcutsToShow)
    {
        QTableWidgetItem* shortcutName = new QTableWidgetItem(shortcut.Name);
        QTableWidgetItem* shortcutKey = new QTableWidgetItem(shortcut.Hotkey.toString(QKeySequence::NativeText));
        ui->tblShortcuts->setItem(row, 0, shortcutName);
        ui->tblShortcuts->setItem(row, 1, shortcutKey);
        row++;
    }
    ui->tblShortcuts->setSortingEnabled(true);
}

void ShortcutsDialog::filterShortcutsByName(const QString & nameFilter, QMap<QString, Configuration::Shortcut> & mapToFill)
{
    for(auto shortcut : Config()->Shortcuts)
    {
        if(shortcut.Name.contains(nameFilter, Qt::CaseInsensitive) || nameFilter == QString())
        {
            mapToFill.insert(shortcut.Name, shortcut);
        }
    }
}

void ShortcutsDialog::updateShortcut()
{
    const QKeySequence newKey = ui->shortcutEdit->getKeysequence();
    if(newKey != currentShortcut.Hotkey)
    {
        bool good = true;
        if(!newKey.isEmpty())
        {
            int idx = 0;
            for(QMap<QString, Configuration::Shortcut>::iterator i = Config()->Shortcuts.begin(); i != Config()->Shortcuts.end(); ++i, idx++)
            {
                if(i.value().Name == currentShortcut.Name) //skip current shortcut in list
                    continue;
                if(i.value().GlobalShortcut && i.value().Hotkey == newKey) //newkey is trying to override a global shortcut
                {
                    good = false;
                    break;
                }
                else if(currentShortcut.GlobalShortcut && i.value().Hotkey == newKey) //current shortcut is global and overrides another local hotkey
                {
                    ui->tblShortcuts->setItem(idx, 1, new QTableWidgetItem(""));
                    Config()->setShortcut(i.key(), QKeySequence());
                }
                else if(i.value().Hotkey == newKey)  // This shortcut already exists (both are local)
                {
                    // Ask user if they want to override the shortcut.
                    QMessageBox mbox;
                    mbox.setIcon(QMessageBox::Question);
                    mbox.setText("This hotkey is already used by the action \"" + i.value().Name + "\".\n"
                                 "Do you want to override it?");
                    mbox.setStandardButtons(QMessageBox::No | QMessageBox::Yes);
                    mbox.setDefaultButton(QMessageBox::Yes);

                    good = mbox.exec() == QMessageBox::Yes;

                    if(good)
                    {
                        ui->tblShortcuts->setItem(idx, 1, new QTableWidgetItem(""));
                        Config()->setShortcut(i.key(), QKeySequence());
                    }
                }
            }
        }
        if(good)
        {
            for(QMap<QString, Configuration::Shortcut>::iterator i = Config()->Shortcuts.begin(); i != Config()->Shortcuts.end(); ++i)
            {
                if(i.value().Name == currentShortcut.Name)
                {
                    Config()->setShortcut(i.key(), newKey);
                    break;
                }
            }
            QString keyText = "";
            if(!newKey.isEmpty())
                keyText = newKey.toString(QKeySequence::NativeText);
            ui->tblShortcuts->item(currentRow, 1)->setText(keyText);
            ui->shortcutEdit->setErrorState(false);
        }
        else
        {
            ui->shortcutEdit->setErrorState(true);
        }
    }
}
void ShortcutsDialog::on_btnClearShortcut_clicked()
{
    for(QMap<QString, Configuration::Shortcut>::iterator i = Config()->Shortcuts.begin(); i != Config()->Shortcuts.end(); ++i)
    {
        if(i.value().Name == currentShortcut.Name)
        {
            Config()->setShortcut(i.key(), QKeySequence());
            break;
        }
    }
    QString emptyString;
    ui->tblShortcuts->item(currentRow, 1)->setText(emptyString);
    ui->shortcutEdit->setText(emptyString);
    ui->shortcutEdit->setErrorState(false);
}

void ShortcutsDialog::syncTextfield()
{
    QModelIndexList indexes = ui->tblShortcuts->selectionModel()->selectedRows();
    if(indexes.count() < 1)
        return;
    currentRow = indexes.at(0).row();
    for(auto shortcut : Config()->Shortcuts)
    {
        if(shortcut.Name == ui->tblShortcuts->item(currentRow, 0)->text())
        {
            currentShortcut = shortcut;
            break;
        }
    }
    ui->shortcutEdit->setErrorState(false);
    ui->shortcutEdit->setText(currentShortcut.Hotkey.toString(QKeySequence::NativeText));
    ui->shortcutEdit->setFocus();
}

ShortcutsDialog::~ShortcutsDialog()
{
    delete ui;
}

void ShortcutsDialog::on_btnSave_clicked()
{
    Config()->writeShortcuts();
    GuiAddStatusBarMessage(QString("%1\n").arg(tr("Settings saved!")).toUtf8().constData());
}

void ShortcutsDialog::rejectedSlot()
{
    Config()->readShortcuts();
}

void ShortcutsDialog::on_FilterTextChanged(const QString & actionName)
{
    showShortcutsFiltered(actionName);
}
