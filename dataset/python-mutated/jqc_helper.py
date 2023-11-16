"""This module contains methods for opening jquery-confirm boxes.
These helper methods SHOULD NOT be called directly from tests."""
from seleniumbase.fixtures import constants
from seleniumbase.fixtures import js_utils
form_code = '\'<form align="center" action="" class="jqc_form">\' +\n    \'<div class="form-group">\' +\n    \'<input style="font-size:20px; background-color: #f8fdfd; \' +\n    \' width: 84%%; border: 1px solid blue; \' +\n    \' box-shadow:inset 0 0 2px 2px #f4fafa;"\' +\n    \' type="text" class="jqc_input" />\' +\n    \'</div>\' +\n    \'</form>\''

def jquery_confirm_button_dialog(driver, message, buttons, options=None):
    if False:
        while True:
            i = 10
    js_utils.activate_jquery_confirm(driver)
    theme = constants.JqueryConfirm.DEFAULT_THEME
    border_color = constants.JqueryConfirm.DEFAULT_COLOR
    width = constants.JqueryConfirm.DEFAULT_WIDTH
    if options:
        for option in options:
            if option[0].lower() == 'theme':
                theme = option[1]
            elif option[0].lower() == 'color':
                border_color = option[1]
            elif option[0].lower() == 'width':
                width = option[1]
            else:
                raise Exception('Unknown option: "%s"' % option[0])
    if not message:
        message = ''
    key_row = ''
    if len(buttons) == 1:
        key_row = "keys: ['enter', 'y', '1'],"
    b_html = "button_%s: {\n        btnClass: 'btn-%s',\n        text: '<b>%s</b>',\n        %s\n        action: function(){\n            jqc_status = '%s';\n            $jqc_status = jqc_status;\n            jconfirm.lastButtonText = jqc_status;\n        }\n    },"
    all_buttons = ''
    btn_count = 0
    for button in buttons:
        btn_count += 1
        text = button[0]
        text = js_utils.escape_quotes_if_needed(text)
        if len(buttons) > 1 and text.lower() == 'yes':
            key_row = "keys: ['y'],"
            if btn_count < 10:
                key_row = "keys: ['y', '%s']," % btn_count
        elif len(buttons) > 1 and text.lower() == 'no':
            key_row = "keys: ['n'],"
            if btn_count < 10:
                key_row = "keys: ['n', '%s']," % btn_count
        elif len(buttons) > 1:
            if btn_count < 10:
                key_row = "keys: ['%s']," % btn_count
        color = button[1]
        if not color:
            color = 'blue'
        new_button = b_html % (btn_count, color, text, key_row, text)
        all_buttons += new_button
    content = '<div></div><font color="#0066ee">%s</font>' % message
    content = js_utils.escape_quotes_if_needed(content)
    overlay_opacity = '0.32'
    if theme.lower() == 'supervan':
        overlay_opacity = '0.56'
    if theme.lower() == 'bootstrap':
        overlay_opacity = '0.64'
    if theme.lower() == 'modern':
        overlay_opacity = '0.5'
    if theme.lower() == 'material':
        overlay_opacity = '0.4'
    jqcd = "jconfirm({\n            boxWidth: '%s',\n            useBootstrap: false,\n            containerFluid: true,\n            bgOpacity: %s,\n            type: '%s',\n            theme: '%s',\n            animationBounce: 1,\n            typeAnimated: true,\n            animation: 'scale',\n            draggable: true,\n            dragWindowGap: 1,\n            container: 'body',\n            title: '%s',\n            content: '<div></div>',\n            buttons: {\n                %s\n            }\n        });" % (width, overlay_opacity, border_color, theme, content, all_buttons)
    driver.execute_script(jqcd)

def jquery_confirm_text_dialog(driver, message, button=None, options=None):
    if False:
        print('Hello World!')
    js_utils.activate_jquery_confirm(driver)
    theme = constants.JqueryConfirm.DEFAULT_THEME
    border_color = constants.JqueryConfirm.DEFAULT_COLOR
    width = constants.JqueryConfirm.DEFAULT_WIDTH
    if not message:
        message = ''
    if button:
        if not type(button) is list and (not type(button) is tuple):
            raise Exception('"button" should be a (text, color) tuple!')
        if len(button) != 2:
            raise Exception('"button" should be a (text, color) tuple!')
    else:
        button = ('Submit', 'blue')
    if options:
        for option in options:
            if option[0].lower() == 'theme':
                theme = option[1]
            elif option[0].lower() == 'color':
                border_color = option[1]
            elif option[0].lower() == 'width':
                width = option[1]
            else:
                raise Exception('Unknown option: "%s"' % option[0])
    btn_text = button[0]
    btn_color = button[1]
    if not btn_color:
        btn_color = 'blue'
    content = '<div></div><font color="#0066ee">%s</font>' % message
    content = js_utils.escape_quotes_if_needed(content)
    overlay_opacity = '0.32'
    if theme.lower() == 'supervan':
        overlay_opacity = '0.56'
    if theme.lower() == 'bootstrap':
        overlay_opacity = '0.64'
    if theme.lower() == 'modern':
        overlay_opacity = '0.5'
    if theme.lower() == 'material':
        overlay_opacity = '0.4'
    jqcd = 'jconfirm({\n            boxWidth: \'%s\',\n            useBootstrap: false,\n            containerFluid: true,\n            bgOpacity: %s,\n            type: \'%s\',\n            theme: \'%s\',\n            animationBounce: 1,\n            typeAnimated: true,\n            animation: \'scale\',\n            draggable: true,\n            dragWindowGap: 1,\n            container: \'body\',\n            title: \'%s\',\n            content: \'<div></div>\' +\n            %s,\n            buttons: {\n                formSubmit: {\n                btnClass: \'btn-%s\',\n                text: \'%s\',\n                action: function () {\n                    jqc_input = this.$content.find(\'.jqc_input\').val();\n                    $jqc_input = this.$content.find(\'.jqc_input\').val();\n                    jconfirm.lastInputText = jqc_input;\n                    $jqc_status = \'%s\';  // There is only one button\n                },\n            },\n            },\n            onContentReady: function () {\n            var jc = this;\n            this.$content.find(\'form.jqc_form\').on(\'submit\', function (e) {\n                // User submits the form by pressing "Enter" in the field\n                e.preventDefault();\n                jc.$$formSubmit.trigger(\'click\');  // Click the button\n            });\n            }\n        });' % (width, overlay_opacity, border_color, theme, content, form_code, btn_color, btn_text, btn_text)
    driver.execute_script(jqcd)

def jquery_confirm_full_dialog(driver, message, buttons, options=None):
    if False:
        i = 10
        return i + 15
    js_utils.activate_jquery_confirm(driver)
    theme = constants.JqueryConfirm.DEFAULT_THEME
    border_color = constants.JqueryConfirm.DEFAULT_COLOR
    width = constants.JqueryConfirm.DEFAULT_WIDTH
    if not message:
        message = ''
    btn_count = 0
    b_html = "button_%s: {\n            btnClass: 'btn-%s',\n            text: '%s',\n            action: function(){\n            jqc_input = this.$content.find('.jqc_input').val();\n            $jqc_input = this.$content.find('.jqc_input').val();\n            jconfirm.lastInputText = jqc_input;\n            $jqc_status = '%s';\n            }\n        },"
    b1_html = "formSubmit: {\n            btnClass: 'btn-%s',\n            text: '%s',\n            action: function(){\n            jqc_input = this.$content.find('.jqc_input').val();\n            $jqc_input = this.$content.find('.jqc_input').val();\n            jconfirm.lastInputText = jqc_input;\n            jqc_status = '%s';\n            $jqc_status = jqc_status;\n            jconfirm.lastButtonText = jqc_status;\n            }\n        },"
    one_button_trigger = ''
    if len(buttons) == 1:
        one_button_trigger = "jc.$$formSubmit.trigger('click');"
    all_buttons = ''
    for button in buttons:
        text = button[0]
        text = js_utils.escape_quotes_if_needed(text)
        color = button[1]
        if not color:
            color = 'blue'
        btn_count += 1
        if len(buttons) == 1:
            new_button = b1_html % (color, text, text)
        else:
            new_button = b_html % (btn_count, color, text, text)
        all_buttons += new_button
    if options:
        for option in options:
            if option[0].lower() == 'theme':
                theme = option[1]
            elif option[0].lower() == 'color':
                border_color = option[1]
            elif option[0].lower() == 'width':
                width = option[1]
            else:
                raise Exception('Unknown option: "%s"' % option[0])
    content = '<div></div><font color="#0066ee">%s</font>' % message
    content = js_utils.escape_quotes_if_needed(content)
    overlay_opacity = '0.32'
    if theme.lower() == 'supervan':
        overlay_opacity = '0.56'
    if theme.lower() == 'bootstrap':
        overlay_opacity = '0.64'
    if theme.lower() == 'modern':
        overlay_opacity = '0.5'
    if theme.lower() == 'material':
        overlay_opacity = '0.4'
    jqcd = 'jconfirm({\n            boxWidth: \'%s\',\n            useBootstrap: false,\n            containerFluid: true,\n            bgOpacity: %s,\n            type: \'%s\',\n            theme: \'%s\',\n            animationBounce: 1,\n            typeAnimated: true,\n            animation: \'scale\',\n            draggable: true,\n            dragWindowGap: 1,\n            container: \'body\',\n            title: \'%s\',\n            content: \'<div></div>\' +\n            %s,\n            buttons: {\n                %s\n            },\n            onContentReady: function () {\n            var jc = this;\n            this.$content.find(\'form.jqc_form\').on(\'submit\', function (e) {\n                // User submits the form by pressing "Enter" in the field\n                e.preventDefault();\n                %s\n            });\n            }\n        });' % (width, overlay_opacity, border_color, theme, content, form_code, all_buttons, one_button_trigger)
    driver.execute_script(jqcd)