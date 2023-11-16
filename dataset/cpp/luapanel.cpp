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
// Modified by
//   Vratislav Zival
//
///////////////////////////////////////////////////////////////////////////////

#include "luapanel.h"

#include <wx/fdrepdlg.h>
#include <wx/stc/stc.h>

#include "codegen/codewriter.h"
#include "codegen/luacg.h"
#include "model/objectbase.h"
#include "rad/appdata.h"
#include "rad/codeeditor/codeeditor.h"
#include "rad/wxfbevent.h"
#include "utils/typeconv.h"
#include "utils/wxfbexception.h"


BEGIN_EVENT_TABLE(LuaPanel, wxPanel)
EVT_FB_CODE_GENERATION(LuaPanel::OnCodeGeneration)
EVT_FB_PROJECT_REFRESH(LuaPanel::OnProjectRefresh)
EVT_FB_PROPERTY_MODIFIED(LuaPanel::OnPropertyModified)
EVT_FB_OBJECT_CREATED(LuaPanel::OnObjectChange)
EVT_FB_OBJECT_REMOVED(LuaPanel::OnObjectChange)
EVT_FB_OBJECT_SELECTED(LuaPanel::OnObjectChange)
EVT_FB_EVENT_HANDLER_MODIFIED(LuaPanel::OnEventHandlerModified)

EVT_FIND(wxID_ANY, LuaPanel::OnFind)
EVT_FIND_NEXT(wxID_ANY, LuaPanel::OnFind)
END_EVENT_TABLE()


LuaPanel::LuaPanel(wxWindow* parent, int id) : wxPanel(parent, id)
{
    AppData()->AddHandler(this->GetEventHandler());
    wxBoxSizer* top_sizer = new wxBoxSizer(wxVERTICAL);

    m_luaPanel = new CodeEditor(this, wxID_ANY);
    InitStyledTextCtrl(m_luaPanel->GetTextCtrl());

    top_sizer->Add(m_luaPanel, 1, wxEXPAND, 0);

    SetSizer(top_sizer);
    SetAutoLayout(true);
    // top_sizer->SetSizeHints( this );
    top_sizer->Fit(this);
    top_sizer->Layout();

    m_luaCW = PTCCodeWriter(new TCCodeWriter(m_luaPanel->GetTextCtrl()));
}

LuaPanel::~LuaPanel()
{
    // delete m_icons;
    AppData()->RemoveHandler(this->GetEventHandler());
}

void LuaPanel::InitStyledTextCtrl(wxStyledTextCtrl* stc)
{
    stc->SetLexer(wxSTC_LEX_LUA);
    stc->SetKeyWords(0, wxT("and assert break class continue def del elif else \
							   except exec finally for from global if import in \
							   is lambda not or pass print raise return try while"));

#ifdef __WXGTK__
    wxFont font(8, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    font.SetFaceName(wxT("Monospace"));
#else
    wxFont font(10, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
#endif

    stc->StyleSetForeground(wxSTC_STYLE_DEFAULT, wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
    stc->StyleSetBackground(wxSTC_STYLE_DEFAULT, wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW));
    stc->StyleSetFont(wxSTC_STYLE_DEFAULT, font);
    stc->StyleClearAll();
    stc->StyleSetBold(wxSTC_C_WORD, true);
    if (!AppData()->IsDarkMode()) {
        stc->StyleSetForeground(wxSTC_C_WORD, *wxBLUE);
        stc->StyleSetForeground(wxSTC_C_STRING, *wxRED);
        stc->StyleSetForeground(wxSTC_C_STRINGEOL, *wxRED);
        stc->StyleSetForeground(wxSTC_C_PREPROCESSOR, wxColour(49, 106, 197));
        stc->StyleSetForeground(wxSTC_C_COMMENT, wxColour(0, 128, 0));
        stc->StyleSetForeground(wxSTC_C_COMMENTLINE, wxColour(0, 128, 0));
        stc->StyleSetForeground(wxSTC_C_COMMENTDOC, wxColour(0, 128, 0));
        stc->StyleSetForeground(wxSTC_C_COMMENTLINEDOC, wxColour(0, 128, 0));
        stc->StyleSetForeground(wxSTC_C_NUMBER, *wxBLUE);
    } else {
        stc->StyleSetForeground(wxSTC_C_WORD, wxColour(221, 40, 103));
        stc->StyleSetForeground(wxSTC_C_STRING, wxColour(23, 198, 163));
        stc->StyleSetForeground(wxSTC_C_STRINGEOL, wxColour(23, 198, 163));
        stc->StyleSetForeground(wxSTC_C_PREPROCESSOR, wxColour(204, 129, 186));
        stc->StyleSetForeground(wxSTC_C_COMMENT, wxColour(98, 98, 98));
        stc->StyleSetForeground(wxSTC_C_COMMENTLINE, wxColour(98, 98, 98));
        stc->StyleSetForeground(wxSTC_C_COMMENTDOC, wxColour(98, 98, 98));
        stc->StyleSetForeground(wxSTC_C_COMMENTLINEDOC, wxColour(98, 98, 98));
        stc->StyleSetForeground(wxSTC_C_NUMBER, wxColour(104, 151, 187));
    }
    stc->SetUseTabs(true);
    stc->SetTabWidth(4);
    stc->SetTabIndents(true);
    stc->SetBackSpaceUnIndents(true);
    stc->SetIndent(4);
    stc->SetSelBackground(true, wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
    stc->SetSelForeground(true, wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHTTEXT));

    stc->SetCaretForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
    stc->SetCaretWidth(2);
    stc->SetReadOnly(true);
}

void LuaPanel::OnFind(wxFindDialogEvent& event)
{
    m_luaPanel->GetEventHandler()->ProcessEvent(event);
}

void LuaPanel::OnPropertyModified(wxFBPropertyEvent& event)
{
    // Generate code to the panel only
    event.SetId(1);
    OnCodeGeneration(event);
}

void LuaPanel::OnProjectRefresh(wxFBEvent& event)
{
    // Generate code to the panel only
    event.SetId(1);
    OnCodeGeneration(event);
}

void LuaPanel::OnObjectChange(wxFBObjectEvent& event)
{
    // Generate code to the panel only
    event.SetId(1);
    OnCodeGeneration(event);
}

void LuaPanel::OnEventHandlerModified(wxFBEventHandlerEvent& event)
{
    // Generate code to the panel only
    event.SetId(1);
    OnCodeGeneration(event);
}

void LuaPanel::OnCodeGeneration(wxFBEvent& event)
{
    PObjectBase objectToGenerate;

    // Generate code in the panel if the panel is active
    bool doPanel = IsShown();

    // Using the previously unused Id field in the event to carry a boolean
    bool panelOnly = (event.GetId() != 0);

    // Only generate to panel + panel is not shown = do nothing
    if (panelOnly && !doPanel) {
        return;
    }

    // For code preview generate only code relevant to selected form,
    // otherwise generate full project code.

    // Create copy of the original project due to possible temporary modifications
    PObjectBase project = PObjectBase(new ObjectBase(*AppData()->GetProjectData()));

    if (panelOnly) {
        objectToGenerate = AppData()->GetSelectedForm();
    }

    if (!panelOnly || !objectToGenerate) {
        objectToGenerate = project;
    }

    // If only one project item should be generated then remove the rest items
    // from the temporary project
    if (doPanel && panelOnly && (objectToGenerate != project)) {
        if (project->GetChildCount() > 0) {
            unsigned int i = 0;
            while (project->GetChildCount() > 1) {
                if (project->GetChild(i) != objectToGenerate) {
                    project->RemoveChild(i);
                } else
                    i++;
            }
        }
    }

    if (!project || !objectToGenerate)
        return;

    // Get Lua properties from the project

    // If Lua generation is not enabled, do not generate the file
    bool doFile = false;
    PProperty pCodeGen = project->GetProperty(wxT("code_generation"));
    if (pCodeGen) {
        doFile = TypeConv::FlagSet(wxT("Lua"), pCodeGen->GetValue()) && !panelOnly;
    }

    if (!(doPanel || doFile)) {
        return;
    }

    // Get First ID from Project File
    unsigned int firstID = 1000;
    PProperty pFirstID = project->GetProperty(wxT("first_id"));
    if (pFirstID) {
        firstID = pFirstID->GetValueAsInteger();
    }

    // Get the file name
    wxString file;
    PProperty pfile = project->GetProperty(wxT("file"));
    if (pfile) {
        file = pfile->GetValue();
    }
    if (file.empty()) {
        file = wxT("noname");
    }

    // Determine if the path is absolute or relative
    bool useRelativePath = false;
    PProperty pRelPath = project->GetProperty(wxT("relative_path"));
    if (pRelPath) {
        useRelativePath = (pRelPath->GetValueAsInteger() ? true : false);
    }

    // Get the output path
    wxString path;
    try {
        path = AppData()->GetOutputPath();
    } catch (wxFBException& ex) {
        if (doFile) {
            path = wxEmptyString;
            wxLogWarning(ex.what());
            return;
        }
    }

    // Generate code in the panel
    if (doPanel) {
        LuaCodeGenerator codegen;
        codegen.UseRelativePath(useRelativePath, path);

        if (pFirstID) {
            codegen.SetFirstID(firstID);
        }

        codegen.SetSourceWriter(m_luaCW);

        Freeze();

        wxStyledTextCtrl* luaEditor = m_luaPanel->GetTextCtrl();
        luaEditor->SetReadOnly(false);
        int luaLine = luaEditor->GetFirstVisibleLine() + luaEditor->LinesOnScreen() - 1;
        int luaXOffset = luaEditor->GetXOffset();

        codegen.GenerateCode(project);

        luaEditor->SetReadOnly(true);
        luaEditor->GotoLine(luaLine);
        luaEditor->SetXOffset(luaXOffset);
        luaEditor->SetAnchor(0);
        luaEditor->SetCurrentPos(0);

        Thaw();
    }

    // Generate code in the file
    if (doFile) {
        try {
            LuaCodeGenerator codegen;
            codegen.UseRelativePath(useRelativePath, path);

            if (pFirstID) {
                codegen.SetFirstID(firstID);
            }

            // Determine if Microsoft BOM should be used
            bool useMicrosoftBOM = false;

            PProperty pUseMicrosoftBOM = project->GetProperty(wxT("use_microsoft_bom"));

            if (pUseMicrosoftBOM) {
                useMicrosoftBOM = (pUseMicrosoftBOM->GetValueAsInteger() != 0);
            }

            // Determine if Utf8 or Ansi is to be created
            bool useUtf8 = false;
            PProperty pUseUtf8 = project->GetProperty(_("encoding"));

            if (pUseUtf8) {
                useUtf8 = (pUseUtf8->GetValueAsString() != wxT("ANSI"));
            }

            PCodeWriter lua_cw(new FileCodeWriter(path + file + wxT(".lua"), useMicrosoftBOM, useUtf8));

            codegen.SetSourceWriter(lua_cw);
            codegen.GenerateCode(project);
            wxLogStatus(wxT("Code generated on \'%s\'."), path);
        } catch (wxFBException& ex) {
            wxLogError(ex.what());
        }
    }
}
