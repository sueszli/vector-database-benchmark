///////////////////////////////////////////////////////////////////////////////
//
// wxFormBuilder - A Visual Dialog Editor for wxWidgets.
// Copyright (C) 2005 José Antonio Hurtado
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
// Written by
//   José Antonio Hurtado - joseantonio.hurtado@gmail.com
//   Juan Antonio Ortega  - jortegalalmolda@gmail.com
//
///////////////////////////////////////////////////////////////////////////////

#include "menueditor.h"

#include "model/objectbase.h"


enum {
    ID_ADDMENUITEM = wxID_HIGHEST + 1,
    ID_ADDSEPARATOR,
    ID_MENUDOWN,
    ID_MENULEFT,
    ID_MENURIGHT,
    ID_MENUUP,
    ID_REMOVEMENUITEM,
    ID_LABEL,
    ID_MODIFYMENUITEM,
};

#define IDENTATION 4

BEGIN_EVENT_TABLE(MenuEditor, wxDialog)
EVT_BUTTON(ID_ADDMENUITEM, MenuEditor::OnAddMenuItem)
EVT_BUTTON(ID_ADDSEPARATOR, MenuEditor::OnAddSeparator)
EVT_BUTTON(ID_MODIFYMENUITEM, MenuEditor::OnModifyMenuItem)
EVT_BUTTON(ID_REMOVEMENUITEM, MenuEditor::OnRemoveMenuItem)
EVT_BUTTON(ID_MENUDOWN, MenuEditor::OnMenuDown)
EVT_BUTTON(ID_MENULEFT, MenuEditor::OnMenuLeft)
EVT_BUTTON(ID_MENURIGHT, MenuEditor::OnMenuRight)
EVT_BUTTON(ID_MENUUP, MenuEditor::OnMenuUp)
EVT_UPDATE_UI_RANGE(ID_MENUDOWN, ID_MENUUP, MenuEditor::OnUpdateMovers)
EVT_TEXT_ENTER(wxID_ANY, MenuEditor::OnEnter)
EVT_TEXT(ID_LABEL, MenuEditor::OnLabelChanged)
EVT_LIST_ITEM_ACTIVATED(wxID_ANY, MenuEditor::OnItemActivated)
END_EVENT_TABLE()


MenuEditor::MenuEditor(wxWindow* parent, int id) :
  wxDialog(parent, id, wxT("Menu Editor"), wxDefaultPosition, wxDefaultSize)
{
    wxBoxSizer* mainSizer;
    mainSizer = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* sizerTop;
    sizerTop = new wxBoxSizer(wxHORIZONTAL);
    m_menuList =
      new wxListCtrl(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT | wxLC_SINGLE_SEL | wxSTATIC_BORDER);
    m_menuList->InsertColumn(0, wxT("Label"), wxLIST_FORMAT_LEFT, 150);
    m_menuList->InsertColumn(1, wxT("Shortcut"), wxLIST_FORMAT_LEFT, 80);
    m_menuList->InsertColumn(2, wxT("Id"), wxLIST_FORMAT_LEFT, 80);
    m_menuList->InsertColumn(3, wxT("Name"), wxLIST_FORMAT_LEFT, 50);
    m_menuList->InsertColumn(4, wxT("Help String"), wxLIST_FORMAT_LEFT, 150);
    m_menuList->InsertColumn(5, wxT("Kind"), wxLIST_FORMAT_LEFT, 120);
    int width = 0;
    for (int i = 0; i < m_menuList->GetColumnCount(); ++i) { width += m_menuList->GetColumnWidth(i); }
    m_menuList->SetMinSize(wxSize(width, -1));
    sizerTop->Add(m_menuList, 1, wxALL | wxEXPAND, 5);
    wxStaticBoxSizer* sizer1;
    sizer1 = new wxStaticBoxSizer(new wxStaticBox(this, wxID_ANY, wxT("Menu item")), wxVERTICAL);
    const auto sizer11 = new wxFlexGridSizer(2, 0, 0);
    sizer11->AddGrowableCol(1);

    wxStaticText* m_stLabel;
    m_stLabel = new wxStaticText(this, wxID_ANY, wxT("Label"), wxDefaultPosition, wxDefaultSize, 0);
    sizer11->Add(m_stLabel, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);
    m_tcLabel = new wxTextCtrl(this, ID_LABEL, wxT(""), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    sizer11->Add(m_tcLabel, 0, wxALL | wxEXPAND, 5);

    wxStaticText* m_stShortcut;
    m_stShortcut = new wxStaticText(this, wxID_ANY, wxT("Shortcut"), wxDefaultPosition, wxDefaultSize, 0);
    sizer11->Add(m_stShortcut, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);
    m_tcShortcut = new wxTextCtrl(this, ID_LABEL, wxT(""), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    sizer11->Add(m_tcShortcut, 0, wxALL | wxEXPAND, 5);

    wxStaticText* m_stId;
    m_stId = new wxStaticText(this, wxID_ANY, wxT("Id"), wxDefaultPosition, wxDefaultSize, 0);
    sizer11->Add(m_stId, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);
    m_tcId = new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    sizer11->Add(m_tcId, 0, wxALL | wxEXPAND, 5);

    wxStaticText* m_stName;
    m_stName = new wxStaticText(this, wxID_ANY, wxT("Name"), wxDefaultPosition, wxDefaultSize, 0);
    sizer11->Add(m_stName, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);
    m_tcName = new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    sizer11->Add(m_tcName, 0, wxALL | wxEXPAND, 5);

    wxStaticText* m_stHelpString;
    m_stHelpString = new wxStaticText(this, wxID_ANY, wxT("Help String"), wxDefaultPosition, wxDefaultSize, 0);
    sizer11->Add(m_stHelpString, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);
    m_tcHelpString = new wxTextCtrl(this, wxID_ANY, wxT(""), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    sizer11->Add(m_tcHelpString, 0, wxALL | wxEXPAND, 5);

    sizer1->Add(sizer11, 0, wxALL | wxEXPAND, 0);

    wxString choices[] = {wxT("Normal"), wxT("Check"), wxT("Radio")};
    m_rbItemKind =
      new wxRadioBox(this, wxID_ANY, wxT("Kind"), wxDefaultPosition, wxDefaultSize, 3, choices, 1, wxRA_SPECIFY_ROWS);
    sizer1->Add(m_rbItemKind, 0, wxALL | wxEXPAND, 5);

    wxBoxSizer* sizer4;
    sizer4 = new wxBoxSizer(wxHORIZONTAL);

    wxButton* m_bAdd;
    m_bAdd = new wxButton(this, ID_ADDMENUITEM, wxT("&Add"), wxDefaultPosition, wxDefaultSize, 0);
    sizer4->Add(m_bAdd, 1, wxALL, 5);

    wxButton* m_bModify;
    m_bModify = new wxButton(this, ID_MODIFYMENUITEM, wxT("&Modify"), wxDefaultPosition, wxDefaultSize, 0);
    sizer4->Add(m_bModify, 1, wxALL, 5);

    wxButton* m_bRemove;
    m_bRemove = new wxButton(this, ID_REMOVEMENUITEM, wxT("&Remove"), wxDefaultPosition, wxDefaultSize, 0);
    sizer4->Add(m_bRemove, 1, wxALL, 5);

    sizer1->Add(sizer4, 0, wxEXPAND, 5);

    wxButton* m_bAddSep;
    m_bAddSep = new wxButton(this, ID_ADDSEPARATOR, wxT("Add &Separator"), wxDefaultPosition, wxDefaultSize, 0);
    sizer1->Add(m_bAddSep, 0, wxALL | wxEXPAND, 5);
    sizerTop->Add(sizer1, 0, wxALL | wxEXPAND, 5);
    mainSizer->Add(sizerTop, 1, wxEXPAND, 5);
    wxBoxSizer* sizerMoveButtons;
    sizerMoveButtons = new wxBoxSizer(wxHORIZONTAL);
    wxButton* m_bUp;
    m_bUp = new wxButton(this, ID_MENUUP, wxT("&Up"), wxDefaultPosition, wxDefaultSize, 0);
    sizerMoveButtons->Add(m_bUp, 0, wxALL, 5);
    wxButton* m_bDown;
    m_bDown = new wxButton(this, ID_MENUDOWN, wxT("&Down"), wxDefaultPosition, wxDefaultSize, 0);
    sizerMoveButtons->Add(m_bDown, 0, wxALL, 5);
    wxButton* m_bLeft;
    m_bLeft = new wxButton(this, ID_MENULEFT, wxT("<"), wxDefaultPosition, wxDefaultSize, 0);
    sizerMoveButtons->Add(m_bLeft, 0, wxALL, 5);
    wxButton* m_bRight;
    m_bRight = new wxButton(this, ID_MENURIGHT, wxT(">"), wxDefaultPosition, wxDefaultSize, 0);
    sizerMoveButtons->Add(m_bRight, 0, wxALL, 5);
    wxStdDialogButtonSizer* sizerOkCancel = new wxStdDialogButtonSizer();
    sizerOkCancel->AddButton(new wxButton(this, wxID_OK));
    sizerOkCancel->AddButton(new wxButton(this, wxID_CANCEL));
    sizerOkCancel->Realize();
    sizerMoveButtons->Add(sizerOkCancel, 1, wxALL, 5);
    mainSizer->Add(sizerMoveButtons, 0, wxEXPAND | wxALIGN_CENTER_VERTICAL, 5);
    this->SetSizer(mainSizer);
    mainSizer->SetSizeHints(this);
    this->SetAutoLayout(true);
    this->Layout();
    // SetClientSize(560, 368);
    CenterOnScreen();
}

void MenuEditor::AddChild(long& n, int ident, PObjectBase obj)
{
    for (unsigned int i = 0; i < obj->GetChildCount(); i++) {
        PObjectBase childObj = obj->GetChild(i);
        if (childObj->GetClassName() == wxT("wxMenuItem")) {
            InsertItem(
              n++, wxString(wxChar(' '), ident * IDENTATION) + childObj->GetPropertyAsString(wxT("label")),
              childObj->GetPropertyAsString(wxT("shortcut")), childObj->GetPropertyAsString(wxT("id")),
              childObj->GetPropertyAsString(wxT("name")), childObj->GetPropertyAsString(wxT("help")),
              childObj->GetPropertyAsString(wxT("kind")), childObj);
        } else if (childObj->GetClassName() == wxT("separator")) {
            InsertItem(
              n++, wxString(wxChar(' '), ident * IDENTATION) + wxT("---"), wxT(""), wxT(""), wxT(""), wxT(""), wxT(""),
              childObj);
        } else {
            InsertItem(
              n++, wxString(wxChar(' '), ident * IDENTATION) + childObj->GetPropertyAsString(wxT("label")), wxT(""),
              childObj->GetPropertyAsString(wxT("id")), childObj->GetPropertyAsString(wxT("name")),
              childObj->GetPropertyAsString(wxT("help")), wxT(""), childObj);
            AddChild(n, ident + 1, childObj);
        }
    }
}

void MenuEditor::Populate(PObjectBase obj)
{
    assert(obj && obj->GetClassName() == wxT("wxMenuBar"));
    long n = 0;
    AddChild(n, 0, obj);
}

bool MenuEditor::HasChildren(long n)
{
    if (n == m_menuList->GetItemCount() - 1)
        return false;
    else
        return GetItemIdentation(n + 1) > GetItemIdentation(n);
}

PObjectBase MenuEditor::GetMenu(long& n, PObjectDatabase base, bool isSubMenu)
{
    // Get item from list control
    wxString label, shortcut, id, name, help, kind;
    PObjectBase menu;
    GetItem(n, label, shortcut, id, name, help, kind, &menu);

    bool createNew = true;
    if (menu) {
        createNew = (menu->GetClassName() != (isSubMenu ? wxT("submenu") : wxT("wxMenu")));
    }

    // preserve original menu if the object types match
    // this preserves properties that are not exposed in the menu editor - like C++ scope
    if (createNew) {
        PObjectInfo info = base->GetObjectInfo(isSubMenu ? wxT("submenu") : wxT("wxMenu"));
        menu = base->NewObject(info);
    }


    label.Trim(true);
    label.Trim(false);
    menu->GetProperty(wxT("label"))->SetValue(label);
    menu->GetProperty(wxT("name"))->SetValue(name);

    int ident = GetItemIdentation(n);
    n++;
    while (n < m_menuList->GetItemCount() && GetItemIdentation(n) > ident) {
        PObjectBase menuitem;
        GetItem(n, label, shortcut, id, name, help, kind, &menuitem);

        createNew = true;

        label.Trim(true);
        label.Trim(false);
        if (label == wxT("---")) {
            if (menuitem) {
                createNew = (menuitem->GetClassName() != wxT("separator"));
            }

            if (createNew) {
                PObjectInfo info = base->GetObjectInfo(wxT("separator"));
                menuitem = base->NewObject(info);
            }
            menu->AddChild(menuitem);
            menuitem->SetParent(menu);
            n++;
        } else if (HasChildren(n)) {
            PObjectBase child = GetMenu(n, base);
            menu->AddChild(child);
            child->SetParent(menu);
        } else {
            if (menuitem) {
                createNew = (menuitem->GetClassName() != wxT("wxMenuItem"));
            }

            if (createNew) {
                PObjectInfo info = base->GetObjectInfo(wxT("wxMenuItem"));
                menuitem = base->NewObject(info);
            }
            menuitem->GetProperty(wxT("label"))->SetValue(label);
            menuitem->GetProperty(wxT("shortcut"))->SetValue(shortcut);
            menuitem->GetProperty(wxT("name"))->SetValue(name);
            menuitem->GetProperty(wxT("help"))->SetValue(help);
            menuitem->GetProperty(wxT("id"))->SetValue(id);
            menuitem->GetProperty(wxT("kind"))->SetValue(kind);
            menu->AddChild(menuitem);
            menuitem->SetParent(menu);
            n++;
        }
    }

    return menu;
}

PObjectBase MenuEditor::GetMenubar(PObjectDatabase base)
{
    // Disconnect all parent/child relationships in original objects
    for (std::vector<WPObjectBase>::iterator it = m_originalItems.begin(); it != m_originalItems.end(); ++it) {
        PObjectBase obj = it->lock();
        if (obj) {
            obj->RemoveAllChildren();
            obj->SetParent(PObjectBase());
        }
    }

    PObjectInfo info = base->GetObjectInfo(wxT("wxMenuBar"));
    PObjectBase menubar = base->NewObject(info);
    long n = 0;
    while (n < m_menuList->GetItemCount()) {
        PObjectBase child = GetMenu(n, base, false);
        menubar->AddChild(child);
        child->SetParent(menubar);
    }
    return menubar;
}

long MenuEditor::GetSelectedItem()
{
    return m_menuList->GetNextItem(-1, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
}

int MenuEditor::GetItemIdentation(long n)
{
    wxString label = m_menuList->GetItemText(n);
    size_t curIdent = 0;
    while (curIdent < label.Len() && label[curIdent] == wxChar(' ')) curIdent++;
    curIdent /= IDENTATION;

    return (int)curIdent;
}

long MenuEditor::InsertItem(
  long n, const wxString& label, const wxString& shortcut, const wxString& id, const wxString& name,
  const wxString& helpString, const wxString& kind, PObjectBase obj)
{
    long index = m_menuList->InsertItem(n, label);
    m_menuList->SetItem(index, 1, shortcut);
    m_menuList->SetItem(index, 2, id);
    m_menuList->SetItem(index, 3, name);
    m_menuList->SetItem(index, 4, helpString);
    m_menuList->SetItem(index, 5, kind);

    // Prevent loss of data not exposed by menu editor. For Example: bitmaps, events, etc.
    if (obj) {
        m_originalItems.push_back(obj);
        m_menuList->SetItemData(index, m_originalItems.size() - 1);
    } else {
        m_menuList->SetItemData(index, -1);
    }
    return index;
}

void MenuEditor::AddItem(
  const wxString& label, const wxString& shortcut, const wxString& id, const wxString& name, const wxString& help,
  const wxString& kind)
{
    int sel = GetSelectedItem();
    int identation = 0;
    if (sel >= 0)
        identation = GetItemIdentation(sel);
    wxString labelAux = label;
    labelAux.Trim(true);
    labelAux.Trim(false);
    if (sel < 0)
        sel = m_menuList->GetItemCount() - 1;

    labelAux = wxString(wxChar(' '), identation * IDENTATION) + labelAux;

    long index = InsertItem(sel + 1, labelAux, shortcut, id, name, help, kind);
    m_menuList->SetItemState(index, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
}

void MenuEditor::GetItem(
  long n, wxString& label, wxString& shortcut, wxString& id, wxString& name, wxString& help, wxString& kind,
  PObjectBase* obj)
{
    label = m_menuList->GetItemText(n);
    wxListItem item;
    item.m_itemId = n;
    item.m_col = 1;
    item.m_mask = wxLIST_MASK_TEXT;
    m_menuList->GetItem(item);
    shortcut = item.GetText();
    item.m_col++;
    m_menuList->GetItem(item);
    id = item.GetText();
    if (id.empty()) {
        id = wxT("wxID_ANY");
    }
    item.m_col++;
    m_menuList->GetItem(item);
    name = item.GetText();
    item.m_col++;
    m_menuList->GetItem(item);
    help = item.GetText();
    item.m_col++;
    m_menuList->GetItem(item);
    kind = item.GetText();
    if (kind.empty()) {
        kind = wxT("wxITEM_NORMAL");
    }
    if (NULL != obj) {
        int origIndex = m_menuList->GetItemData(item);
        if (origIndex >= 0 && origIndex < (int)m_originalItems.size()) {
            *obj = m_originalItems[origIndex].lock();
        }
    }
}

void MenuEditor::AddNewItem()
{
    wxString kind;
    switch (m_rbItemKind->GetSelection()) {
        case 0:
            kind = wxT("wxITEM_NORMAL");
            break;
        case 1:
            kind = wxT("wxITEM_CHECK");
            break;
        case 2:
            kind = wxT("wxITEM_RADIO");
            break;
    }
    AddItem(
      m_tcLabel->GetValue(), m_tcShortcut->GetValue(), m_tcId->GetValue(), m_tcName->GetValue(),
      m_tcHelpString->GetValue(), kind);
    m_tcLabel->SetValue(wxT(""));
    m_tcShortcut->SetValue(wxT(""));
    m_tcId->SetValue(wxT(""));
    m_tcName->SetValue(wxT(""));
    m_tcHelpString->SetValue(wxT(""));
    m_rbItemKind->SetSelection(0);
    m_tcLabel->SetFocus();
}

void MenuEditor::OnAddMenuItem(wxCommandEvent&)
{
    AddNewItem();
}

void MenuEditor::OnAddSeparator(wxCommandEvent&)
{
    AddItem(wxT("---"), wxT(""), wxT(""), wxT(""), wxT(""), wxT(""));
}

void MenuEditor::OnModifyMenuItem(wxCommandEvent&)
{
    long index = GetSelectedItem();
    int identation = GetItemIdentation(index);
    wxString kind;
    switch (m_rbItemKind->GetSelection()) {
        case 0:
            kind = wxT("wxITEM_NORMAL");
            break;
        case 1:
            kind = wxT("wxITEM_CHECK");
            break;
        case 2:
            kind = wxT("wxITEM_RADIO");
            break;
    }

    m_menuList->SetItem(index, 0, wxString(wxChar(' '), identation * IDENTATION) + m_tcLabel->GetValue());
    m_menuList->SetItem(index, 1, m_tcShortcut->GetValue());
    m_menuList->SetItem(index, 2, m_tcId->GetValue());
    m_menuList->SetItem(index, 3, m_tcName->GetValue());
    m_menuList->SetItem(index, 4, m_tcHelpString->GetValue());
    m_menuList->SetItem(index, 5, kind);
}

void MenuEditor::OnRemoveMenuItem(wxCommandEvent&)
{
    long sel = GetSelectedItem();
    if (sel < m_menuList->GetItemCount() - 1) {
        int curIdent = GetItemIdentation(sel);
        int nextIdent = GetItemIdentation(sel + 1);
        if (nextIdent > curIdent) {
            int res = wxMessageBox(
              wxT("The children of the selected item will be eliminated too. Are you sure you want to continue?"),
              wxT("wxFormBuilder"), wxYES_NO);
            if (res == wxYES) {
                long item = sel + 1;
                while (item < m_menuList->GetItemCount() && GetItemIdentation(item) > curIdent)
                    m_menuList->DeleteItem(item);
                m_menuList->DeleteItem(sel);
            }
        } else
            m_menuList->DeleteItem(sel);
    } else
        m_menuList->DeleteItem(sel);
}

void MenuEditor::OnMenuLeft(wxCommandEvent&)
{
    int sel = GetSelectedItem();
    int curIdent = GetItemIdentation(sel) - 1;
    int childIdent = sel < m_menuList->GetItemCount() - 1 ? GetItemIdentation(sel + 1) : -1;

    if (curIdent < 0 || (childIdent != -1 && abs(curIdent - childIdent) > 1))
        return;

    wxString label = m_menuList->GetItemText(sel);
    label.Trim(true);
    label.Trim(false);
    label = wxString(wxChar(' '), curIdent * IDENTATION) + label;
    m_menuList->SetItemText(sel, label);
}

void MenuEditor::OnMenuRight(wxCommandEvent&)
{
    int sel = GetSelectedItem();
    int curIdent = GetItemIdentation(sel) + 1;
    int parentIdent = sel > 0 ? GetItemIdentation(sel - 1) : -1;

    if (parentIdent == -1 || abs(curIdent - parentIdent) > 1)
        return;

    wxString label = m_menuList->GetItemText(sel);
    label.Trim(true);
    label.Trim(false);
    label = wxString(wxChar(' '), curIdent * IDENTATION) + label;
    m_menuList->SetItemText(sel, label);
}

void MenuEditor::OnMenuUp(wxCommandEvent&)
{
    long sel = GetSelectedItem();
    long prev = sel - 1;
    int prevIdent = GetItemIdentation(prev);
    int curIdent = GetItemIdentation(sel);
    if (prevIdent < curIdent)
        return;
    while (prevIdent > curIdent) prevIdent = GetItemIdentation(--prev);

    PObjectBase obj;
    wxString label, shortcut, id, name, help, kind;
    GetItem(sel, label, shortcut, id, name, help, kind, &obj);

    m_menuList->DeleteItem(sel);
    long newSel = InsertItem(prev, label, shortcut, id, name, help, kind, obj);
    sel++;
    prev++;
    if (sel < m_menuList->GetItemCount()) {
        long childIdent = GetItemIdentation(sel);
        while (sel < m_menuList->GetItemCount() && childIdent > curIdent) {
            PObjectBase childObj;
            GetItem(sel, label, shortcut, id, name, help, kind, &childObj);
            m_menuList->DeleteItem(sel);
            InsertItem(prev, label, shortcut, id, name, help, kind, childObj);
            sel++;
            prev++;
            if (sel < m_menuList->GetItemCount())
                childIdent = GetItemIdentation(sel);
        }
    }
    m_menuList->SetItemState(newSel, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
}

long MenuEditor::GetEndIndex(long n)
{
    long res = n;
    int ident = GetItemIdentation(n);
    res++;
    if (res >= m_menuList->GetItemCount())
        return n;
    while (GetItemIdentation(res) > ident) res++;
    return res - 1;
}

void MenuEditor::OnMenuDown(wxCommandEvent&)
{
    long sel = GetSelectedItem();
    int selIdent = GetItemIdentation(sel);
    long selAux = sel + 1;
    while (GetItemIdentation(selAux) > selIdent) selAux++;
    if (GetItemIdentation(selAux) < selIdent)
        return;
    long endIndex = GetEndIndex(selAux) + 1;

    wxString label, shortcut, id, name, help, kind;
    PObjectBase obj;
    GetItem(sel, label, shortcut, id, name, help, kind, &obj);

    m_menuList->DeleteItem(sel);
    endIndex--;
    long first = InsertItem(endIndex, label, shortcut, id, name, help, kind, obj);
    while (GetItemIdentation(sel) > selIdent) {
        PObjectBase childObj;
        GetItem(sel, label, shortcut, id, name, help, kind, &childObj);
        m_menuList->DeleteItem(sel);
        InsertItem(endIndex, label, shortcut, id, name, help, kind, childObj);
        first--;
    }
    m_menuList->SetItemState(first, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
}

void MenuEditor::OnEnter(wxCommandEvent&)
{
    AddNewItem();
}

void MenuEditor::OnLabelChanged(wxCommandEvent&)
{
    wxString label = m_tcLabel->GetValue();
    wxString id, name;
    bool nextUpper = false;
    if (!label.IsEmpty()) {
        id = wxT("ID_");
    }
    int tabPos = label.Find(wxT("\\t"));
    if (tabPos >= 0) {
        label = label.Left(tabPos);
    }

    for (size_t i = 0; i < label.Len(); i++) {
        if (isalnum(label[i]) != 0) {
            name += (nextUpper ? toupper(label[i]) : tolower(label[i]));
            nextUpper = false;
            id += toupper(label[i]);
        } else if (label[i] == wxChar(' ')) {
            nextUpper = true;
            id += wxT("_");
        }
    }
    if (name.Len() > 0 && isdigit(name[0]) != 0) {
        name = wxT("n") + name;
    }
    m_tcId->SetValue(id);
    m_tcName->SetValue(name);
}

void MenuEditor::OnUpdateMovers(wxUpdateUIEvent& e)
{
    switch (e.GetId()) {
        case ID_MENUUP:
            e.Enable(GetSelectedItem() > 0);
            break;
        case ID_MENUDOWN:
            e.Enable(GetSelectedItem() < m_menuList->GetItemCount() - 1);
            break;
        default:
            break;
    }
}

void MenuEditor::OnItemActivated(wxListEvent& e)
{
    wxString label, shortcut, id, name, helpString, kind;
    GetItem(e.GetIndex(), label, shortcut, id, name, helpString, kind);

    label.Trim(true);
    label.Trim(false);
    m_tcLabel->SetValue(label);
    m_tcShortcut->SetValue(shortcut);
    m_tcId->SetValue(id);
    m_tcName->SetValue(name);
    m_tcHelpString->SetValue(helpString);

    if (kind == wxT("wxITEM_CHECK"))
        m_rbItemKind->SetSelection(1);
    else if (kind == wxT("wxITEM_RADIO"))
        m_rbItemKind->SetSelection(2);
    else
        m_rbItemKind->SetSelection(0);
}
