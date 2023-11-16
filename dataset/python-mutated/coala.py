import logging
import sys
from pyprint.ConsolePrinter import ConsolePrinter
from dependency_management.requirements.PipRequirement import PipRequirement
from coalib.parsing.FilterHelper import apply_filter, apply_filters, InvalidFilterException, filter_vector_to_dict
from coalib.output.Logging import configure_logging
from coalib.parsing.DefaultArgParser import default_arg_parser
from coalib.misc.Exceptions import get_exitcode

def main(debug=False):
    if False:
        for i in range(10):
            print('nop')
    configure_logging()
    args = None
    try:
        args = default_arg_parser().parse_args()
        if args.debug:
            req_ipdb = PipRequirement('ipdb')
            if not req_ipdb.is_installed():
                logging.error('--debug flag requires ipdb. You can install it with:\n%s', ' '.join(req_ipdb.install_command()))
                sys.exit(13)
        if debug or args.debug:
            args.log_level = 'DEBUG'
        from coalib.coala_modes import mode_format, mode_json, mode_non_interactive, mode_normal
        from coalib.output.ConsoleInteraction import show_bears, show_language_bears_capabilities
        console_printer = ConsolePrinter(print_colored=not args.no_color)
        configure_logging(not args.no_color)
        if args.show_bears:
            from coalib.settings.ConfigurationGathering import get_all_bears
            kwargs = {}
            if args.bears:
                kwargs['bear_globs'] = args.bears
            filtered_bears = get_all_bears(**kwargs)
            if args.filter_by_language:
                logging.warning("'--filter-by-language ...' is deprecated. Use '--filter-by language ...' instead.")
                if args.filter_by is None:
                    args.filter_by = []
                args.filter_by.append(['language'] + args.filter_by_language)
            if args.filter_by:
                try:
                    args.filter_by = filter_vector_to_dict(args.filter_by)
                    filtered_bears = apply_filters(args.filter_by, filtered_bears)
                except (InvalidFilterException, NotImplementedError) as ex:
                    console_printer.print(ex)
                    return 2
            (local_bears, global_bears) = filtered_bears
            show_bears(local_bears, global_bears, args.show_description or args.show_details, args.show_details, console_printer, args)
            return 0
        elif args.show_capabilities:
            from coalib.collecting.Collectors import filter_capabilities_by_languages
            (local_bears, _) = apply_filter('language', args.show_capabilities)
            capabilities = filter_capabilities_by_languages(local_bears, args.show_capabilities)
            show_language_bears_capabilities(capabilities, console_printer)
            return 0
        if args.json:
            return mode_json(args, debug=debug)
    except BaseException as exception:
        if not isinstance(exception, SystemExit):
            if args and args.debug:
                import ipdb
                with ipdb.launch_ipdb_on_exception():
                    raise
            if debug:
                raise
        return get_exitcode(exception)
    if args.format:
        return mode_format(args, debug=debug)
    if args.non_interactive:
        return mode_non_interactive(console_printer, args, debug=debug)
    return mode_normal(console_printer, None, args, debug=debug)
if __name__ == '__main__':
    sys.exit(main())