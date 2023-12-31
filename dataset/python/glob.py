from sentry_relay.processing import is_glob_match


def glob_match(
    value, pat, doublestar=False, ignorecase=False, path_normalize=False, allow_newline=True
):
    """A beefed up version of fnmatch.fnmatch"""
    return is_glob_match(
        value if value is not None else "",
        pat,
        double_star=doublestar,
        case_insensitive=ignorecase,
        path_normalize=path_normalize,
        allow_newline=allow_newline,
    )
