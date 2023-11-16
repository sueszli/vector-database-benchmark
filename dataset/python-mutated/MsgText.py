from enum import Enum

class MsgText(Enum):

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.value
    WelcomeToJrnl = "\n        Welcome to jrnl {version}!\n\n        It looks like you've been using an older version of jrnl until now. That's\n        okay - jrnl will now upgrade your configuration and journal files. Afterwards\n        you can enjoy all of the great new features that come with jrnl 2:\n\n        - Support for storing your journal in multiple files\n        - Faster reading and writing for large journals\n        - New encryption back-end that makes installing jrnl much easier\n        - Tons of bug fixes\n\n        Please note that jrnl 1.x is NOT forward compatible with this version of jrnl.\n        If you choose to proceed, you will not be able to use your journals with\n        older versions of jrnl anymore.\n        "
    AllDoneUpgrade = "We're all done here and you can start enjoying jrnl 2"
    InstallComplete = '\n        jrnl configuration created at {config_path}\n        For advanced features, read the docs at https://jrnl.sh\n    '
    InstallJournalPathQuestion = '\n        Path to your journal file (leave blank for {default_journal_path}):\n        '
    DeleteEntryQuestion = "Delete entry '{entry_title}'?"
    ChangeTimeEntryQuestion = "Change time for '{entry_title}'?"
    EncryptJournalQuestion = '\n        Do you want to encrypt your journal? (You can always change this later)\n        '
    UseColorsQuestion = '\n        Do you want jrnl to use colors to display entries? (You can always change this later)\n        '
    YesOrNoPromptDefaultYes = '[Y/n]'
    YesOrNoPromptDefaultNo = '[y/N]'
    ContinueUpgrade = 'Continue upgrading jrnl?'
    OneCharacterYes = 'y'
    OneCharacterNo = 'n'
    Error = 'Error'
    UncaughtException = '\n        {name}\n        {exception}\n\n        This is probably a bug. Please file an issue at:\n        https://github.com/jrnl-org/jrnl/issues/new/choose\n        '
    ConfigDirectoryIsFile = '\n        Problem with config file!\n        The path to your jrnl configuration directory is a file, not a directory:\n\n        {config_directory_path}\n\n        Removing this file will allow jrnl to save its configuration.\n        '
    CantParseConfigFile = '\n        Unable to parse config file at:\n        {config_path}\n        '
    LineWrapTooSmallForDateFormat = '\n        The provided linewrap value of {config_linewrap} is too small by\n        {columns} columns to display the timestamps in the configured time\n        format for journal {journal}.\n\n        You can avoid this error by specifying a linewrap value that is larger\n        by at least {columns} in the configuration file or by using\n        --config-override at the command line\n        '
    CannotEncryptJournalType = "\n        The journal {journal_name} can't be encrypted because it is a\n        {journal_type} journal.\n\n        To encrypt it, create a new journal referencing a file, export\n        this journal to the new journal, then encrypt the new journal.\n        "
    ConfigEncryptedForUnencryptableJournalType = '\n        The config for journal "{journal_name}" has \'encrypt\' set to true, but this type\n        of journal can\'t be encrypted. Please fix your config file.\n        '
    DecryptionFailedGeneric = 'The decryption of journal data failed.'
    KeyboardInterruptMsg = 'Aborted by user'
    CantReadTemplate = '\n        Unable to find a template file {template_path}.\n\n        The following paths were checked:\n         * {jrnl_template_dir}{template_path}\n         * {actual_template_path}\n        '
    NoNamedJournal = "No '{journal_name}' journal configured\n{journals}"
    DoesNotExist = '{name} does not exist'
    JournalNotSaved = 'Entry NOT saved to journal'
    JournalEntryAdded = 'Entry added to {journal_name} journal'
    JournalCountAddedSingular = '{num} entry added'
    JournalCountModifiedSingular = '{num} entry modified'
    JournalCountDeletedSingular = '{num} entry deleted'
    JournalCountAddedPlural = '{num} entries added'
    JournalCountModifiedPlural = '{num} entries modified'
    JournalCountDeletedPlural = '{num} entries deleted'
    JournalCreated = "Journal '{journal_name}' created at {filename}"
    DirectoryCreated = 'Directory {directory_name} created'
    JournalEncrypted = 'Journal will be encrypted'
    JournalEncryptedTo = 'Journal encrypted to {path}'
    JournalDecryptedTo = 'Journal decrypted to {path}'
    BackupCreated = 'Created a backup at {filename}'
    WritingEntryStart = '\n        Writing Entry\n        To finish writing, press {how_to_quit} on a blank line.\n        '
    HowToQuitWindows = 'Ctrl+z and then Enter'
    HowToQuitLinux = 'Ctrl+d'
    EditorMisconfigured = "\n        No such file or directory: '{editor_key}'\n\n        Please check the 'editor' key in your config file for errors:\n            editor: '{editor_key}'\n        "
    EditorNotConfigured = '\n        There is no editor configured\n\n        To use the --edit option, please specify an editor your config file:\n            {config_file}\n\n        For examples of how to configure an external editor, see:\n            https://jrnl.sh/en/stable/external-editors/\n        '
    NoEditsReceivedJournalNotDeleted = '\n        No text received from editor. Were you trying to delete all the entries?\n\n        This seems a bit drastic, so the operation was cancelled.\n\n        To delete all entries, use the --delete option.\n        '
    NoEditsReceived = 'No edits to save, because nothing was changed'
    NoTextReceived = '\n        No entry to save, because no text was received\n        '
    NoChangesToTemplate = '\n        No entry to save, because the template was not changed\n    '
    JournalFailedUpgrade = '\n        The following journal{s} failed to upgrade:\n        {failed_journals}\n\n        Please tell us about this problem at the following URL:\n        https://github.com/jrnl-org/jrnl/issues/new?title=JournalFailedUpgrade\n        '
    UpgradeAborted = 'jrnl was NOT upgraded'
    AbortingUpgrade = 'Aborting upgrade...'
    ImportAborted = 'Entries were NOT imported'
    JournalsToUpgrade = '\n        The following journals will be upgraded to jrnl {version}:\n\n        '
    JournalsToIgnore = '\n        The following journals will not be touched:\n\n        '
    UpgradingJournal = "\n        Upgrading '{journal_name}' journal stored in {path}...\n        "
    UpgradingConfig = 'Upgrading config...'
    PaddedJournalName = '{journal_name:{pad}} -> {path}'
    AltConfigNotFound = '\n        Alternate configuration file not found at the given path:\n            {config_file}\n        '
    ConfigUpdated = '\n        Configuration updated to newest version at {config_path}\n        '
    ConfigDoubleKeys = '\n        There is at least one duplicate key in your configuration file.\n\n        Details:\n        {error_message}\n        '
    Password = 'Password:'
    PasswordFirstEntry = "Enter password for journal '{journal_name}': "
    PasswordConfirmEntry = 'Enter password again: '
    PasswordMaxTriesExceeded = 'Too many attempts with wrong password'
    PasswordCanNotBeEmpty = "Password can't be empty!"
    PasswordDidNotMatch = 'Passwords did not match, please try again'
    WrongPasswordTryAgain = 'Wrong password, try again'
    PasswordStoreInKeychain = 'Do you want to store the password in your keychain?'
    NothingToDelete = '\n        No entries to delete, because the search returned no results\n        '
    NothingToModify = '\n        No entries to modify, because the search returned no results\n        '
    NoEntriesFound = 'no entries found'
    EntryFoundCountSingular = '{num} entry found'
    EntryFoundCountPlural = '{num} entries found'
    HeadingsPastH6 = '\n        Headings increased past H6 on export - {date} {title}\n        '
    YamlMustBeDirectory = '\n        YAML export must be to a directory, not a single file\n        '
    JournalExportedTo = 'Journal exported to {path}'
    ImportSummary = '\n        {count} imported to {journal_name} journal\n        '
    InvalidColor = '{key} set to invalid color: {color}'
    KeyringBackendNotFound = '\n        Keyring backend not found.\n\n        Please install one of the supported backends by visiting:\n          https://pypi.org/project/keyring/\n        '
    KeyringRetrievalFailure = 'Failed to retrieve keyring'
    DeprecatedCommand = '\n        The command {old_cmd} is deprecated and will be removed from jrnl soon.\n        Please use {new_cmd} instead.\n        '