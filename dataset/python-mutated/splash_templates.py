"""
Templates for the splash screen tcl script.
"""
from PyInstaller.compat import is_cygwin, is_darwin, is_win
ipc_script = '\nproc _ipc_server {channel clientaddr clientport} {\n    # This function is called if a new client connects to\n    # the server. This creates a channel, which calls\n    # _ipc_caller if data was send through the connection\n    set client_name [format <%s:%d> $clientaddr $clientport]\n\n    chan configure $channel \\\n        -buffering none \\\n        -encoding utf-8 \\\n        -eofchar \\x04 \\\n        -translation cr\n    chan event $channel readable [list _ipc_caller $channel $client_name]\n}\n\nproc _ipc_caller {channel client_name} {\n    # This function is called if a command was sent through\n    # the tcp connection. The current implementation supports\n    # two commands: update_text and exit, although exit\n    # is implemented to be called if the connection gets\n    # closed (from python) or the character 0x04 was received\n    chan gets $channel cmd\n\n    if {[chan eof $channel]} {\n        # This is entered if either the connection was closed\n        # or the char 0x04 was send\n        chan close $channel\n        exit\n\n    } elseif {![chan blocked $channel]} {\n        # RPC methods\n\n        # update_text command\n        if {[string match "update_text*" $cmd]} {\n            global status_text\n            set first [expr {[string first "(" $cmd] + 1}]\n            set last [expr {[string last ")" $cmd] - 1}]\n\n            set status_text [string range $cmd $first $last]\n        }\n        # Implement other procedures here\n    }\n}\n\n# By setting the port to 0 the os will assign a free port\nset server_socket [socket -server _ipc_server -myaddr localhost 0]\nset server_port [fconfigure $server_socket -sockname]\n\n# This environment variable is shared between the python and the tcl\n# interpreter and publishes the port the tcp server socket is available\nset env(_PYIBoot_SPLASH) [lindex $server_port 2]\n'
image_script = '\n# The variable $_image_data, which holds the data for the splash\n# image is created by the bootloader.\nimage create photo splash_image\nsplash_image put $_image_data\n# delete the variable, because the image now holds the data\nunset _image_data\n\nproc canvas_text_update {canvas tag _var - -}  {\n    # This function is rigged to be called if the a variable\n    # status_text gets changed. This updates the text on\n    # the canvas\n    upvar $_var var\n    $canvas itemconfigure $tag -text $var\n}\n'
splash_canvas_setup = '\npackage require Tk\n\nset image_width [image width splash_image]\nset image_height [image height splash_image]\nset display_width [winfo screenwidth .]\nset display_height [winfo screenheight .]\n\nset x_position [expr {int(0.5*($display_width - $image_width))}]\nset y_position [expr {int(0.5*($display_height - $image_height))}]\n\n# Toplevel frame in which all widgets should be positioned\nframe .root\n\n# Configure the canvas on which the splash\n# screen will be drawn\ncanvas .root.canvas \\\n    -width $image_width \\\n    -height $image_height \\\n    -borderwidth 0 \\\n    -highlightthickness 0\n\n# Draw the image into the canvas, filling it.\n.root.canvas create image \\\n    [expr {$image_width / 2}] \\\n    [expr {$image_height / 2}] \\\n    -image splash_image\n'
splash_canvas_text = '\n# Create a text on the canvas, which tracks the local\n# variable status_text. status_text is changed via C to\n# update the progress on the splash screen.\n# We cannot use the default label, because it has a\n# default background, which cannot be turned transparent\n.root.canvas create text \\\n        %(pad_x)d \\\n        %(pad_y)d \\\n        -fill %(color)s \\\n        -justify center \\\n        -font myFont \\\n        -tag vartext \\\n        -anchor sw\ntrace variable status_text w \\\n    [list canvas_text_update .root.canvas vartext]\nset status_text "%(default_text)s"\n'
splash_canvas_default_font = '\nfont create myFont {*}[font actual TkDefaultFont]\nfont configure myFont -size %(font_size)d\n'
splash_canvas_custom_font = '\nfont create myFont -family %(font)s -size %(font_size)d\n'
if is_win or is_cygwin:
    transparent_setup = '\n# If the image is transparent, the background will be filled\n# with magenta. The magenta background is later replaced with transparency.\n# Here is the limitation of this implementation, that only\n# sharp transparent image corners are possible\nwm attributes . -transparentcolor magenta\n.root.canvas configure -background magenta\n'
elif is_darwin:
    transparent_setup = '\nwm attributes . -transparent 1\n. configure -background systemTransparent\n.root.canvas configure -background systemTransparent\n'
else:
    transparent_setup = ''
pack_widgets = '\n# Position all widgets in the window\npack .root\ngrid .root.canvas   -column 0 -row 0 -columnspan 1 -rowspan 2\n'
position_window_on_top = '\n# Set position and mode of the window - always-on-top behavior\nwm overrideredirect . 1\nwm geometry         . +${x_position}+${y_position}\nwm attributes       . -topmost 1\n'
if is_win or is_cygwin or is_darwin:
    position_window = '\n# Set position and mode of the window\nwm overrideredirect . 1\nwm geometry         . +${x_position}+${y_position}\nwm attributes       . -topmost 0\n'
else:
    position_window = '\n# Set position and mode of the window\nwm geometry         . +${x_position}+${y_position}\nwm attributes       . -type splash\n'
raise_window = '\nraise .\n'

def build_script(text_options=None, always_on_top=False):
    if False:
        while True:
            i = 10
    '\n    This function builds the tcl script for the splash screen.\n    '
    script = [ipc_script, image_script, splash_canvas_setup]
    if text_options:
        if text_options['font'] == 'TkDefaultFont':
            script.append(splash_canvas_default_font % text_options)
        else:
            script.append(splash_canvas_custom_font % text_options)
        script.append(splash_canvas_text % text_options)
    script.append(transparent_setup)
    script.append(pack_widgets)
    script.append(position_window_on_top if always_on_top else position_window)
    script.append(raise_window)
    return '\n'.join(script)