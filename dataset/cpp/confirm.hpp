#pragma once

class ConfirmDialog : public Gtk::Dialog {
public:
    ConfirmDialog(Gtk::Window &parent);
    void SetConfirmText(const Glib::ustring &text);
    void SetAcceptOnly(bool accept_only);

protected:
    Gtk::Label m_label;
    Gtk::Box m_layout;
    Gtk::Button m_ok;
    Gtk::Button m_cancel;
    Gtk::ButtonBox m_bbox;
};
