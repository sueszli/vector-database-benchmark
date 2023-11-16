#include "pch.hpp"
#include "engine_include.hpp"
#include "window_ide.h"
#include "window_view.h"
#include "window_levels.h"
#include "window_weather.h"
#include "window_weather_editor.h"

using editor::window_ide;
using editor::window_view;
using editor::window_levels;
using editor::window_weather;
using editor::window_weather_editor;

//using WeifenLuo::WinFormsUI::Docking::VS2005Theme;

void window_ide::custom_init(XRay::Editor::engine_base* engine)
{
    SuspendLayout();

    m_engine = engine;

    //Extender::SetSchema(EditorDock, Extender::Schema::FromBase);

    m_view = gcnew window_view(*this);
    m_levels = gcnew window_levels(this);
    m_weather = gcnew window_weather(this);
    m_weather_editor = gcnew window_weather_editor(this, m_engine);

    EditorDock->Theme = EditorTheme;

    load_on_create();

    ResumeLayout();
}

void window_ide::custom_finalize()
{
    delete (m_view);
    delete (m_levels);
    delete (m_weather);
    delete (m_weather_editor);
}

window_view % window_ide::view()
{
    VERIFY(m_view);
    return (*m_view);
}

window_levels % window_ide::levels()
{
    VERIFY(m_levels);
    return (*m_levels);
}

window_weather % window_ide::weather()
{
    VERIFY(m_weather);
    return (*m_weather);
}

XRay::Editor::engine_base& window_ide::engine()
{
    VERIFY(m_engine);
    return (*m_engine);
}

window_weather_editor % window_ide::weather_editor()
{
    VERIFY(m_weather_editor);
    return (*m_weather_editor);
}

System::Void window_ide::window_ide_SizeChanged(System::Object ^ sender, System::EventArgs ^ e)
{
    if (WindowState == System::Windows::Forms::FormWindowState::Maximized)
        return;

    if (WindowState == System::Windows::Forms::FormWindowState::Minimized)
        return;

    m_window_rectangle = gcnew System::Drawing::Rectangle(Location, Size);
}

System::Void window_ide::window_ide_LocationChanged(System::Object ^ sender, System::EventArgs ^ e)
{
    m_view->window_view_LocationChanged(sender, e);

    if (WindowState == System::Windows::Forms::FormWindowState::Maximized)
        return;

    if (WindowState == System::Windows::Forms::FormWindowState::Minimized)
        return;

    m_window_rectangle = gcnew System::Drawing::Rectangle(Location, Size);
}

System::Void window_ide::window_ide_Activated(System::Object ^ sender, System::EventArgs ^ e)
{
    m_view->window_view_Activated(sender, e);
}

System::Void window_ide::window_ide_Deactivate(System::Object ^ sender, System::EventArgs ^ e)
{
    m_view->window_view_Deactivate(sender, e);
}

System::Void window_ide::window_ide_FormClosing(System::Object ^ sender, System::Windows::Forms::FormClosingEventArgs ^ e)
{
    e->Cancel = true;
    save_on_exit();
    m_engine->disconnect();
}

XRay::Editor::ide_base& window_ide::ide()
{
    VERIFY(m_ide);
    return (*m_ide);
}
