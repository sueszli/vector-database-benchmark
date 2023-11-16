sample_workspace_dict = {
    "session_name": "sample workspace",
    "start_directory": "~",
    "windows": [
        {
            "window_name": "editor",
            "panes": [
                {"start_directory": "~", "shell_command": ["vim"]},
                {"shell_command": ['cowsay "hey"']},
            ],
            "layout": "main-vertical",
        },
        {
            "window_name": "logging",
            "panes": [
                {
                    "shell_command": ["tail -F /var/log/syslog"],
                    "start_directory": "/var/log",
                }
            ],
        },
        {"options": {"automatic_rename": True}, "panes": [{"shell_command": ["htop"]}]},
    ],
}
