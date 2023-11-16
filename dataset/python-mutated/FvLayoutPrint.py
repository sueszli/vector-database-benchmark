from utils.FmmtLogger import FmmtLogger as logger

def GetFormatter(layout_format: str):
    if False:
        print('Hello World!')
    if layout_format == 'json':
        return JsonFormatter()
    elif layout_format == 'yaml':
        return YamlFormatter()
    elif layout_format == 'html':
        return HtmlFormatter()
    else:
        return TxtFormatter()

class Formatter(object):

    def dump(self, layoutdict, layoutlist, outputfile: str=None) -> None:
        if False:
            return 10
        raise NotImplemented

class JsonFormatter(Formatter):

    def dump(self, layoutdict: dict, layoutlist: list, outputfile: str=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            import json
        except:
            TxtFormatter().dump(layoutdict, layoutlist, outputfile)
            return
        print(outputfile)
        if outputfile:
            with open(outputfile, 'w') as fw:
                json.dump(layoutdict, fw, indent=2)
        else:
            print(json.dumps(layoutdict, indent=2))

class TxtFormatter(Formatter):

    def LogPrint(self, layoutlist: list) -> None:
        if False:
            print('Hello World!')
        for item in layoutlist:
            print(item)
        print('\n')

    def dump(self, layoutdict: dict, layoutlist: list, outputfile: str=None) -> None:
        if False:
            return 10
        logger.info('Binary Layout Info is saved in {} file.'.format(outputfile))
        with open(outputfile, 'w') as f:
            for item in layoutlist:
                f.writelines(item + '\n')

class YamlFormatter(Formatter):

    def dump(self, layoutdict, layoutlist, outputfile=None):
        if False:
            return 10
        TxtFormatter().dump(layoutdict, layoutlist, outputfile)

class HtmlFormatter(Formatter):

    def dump(self, layoutdict, layoutlist, outputfile=None):
        if False:
            i = 10
            return i + 15
        TxtFormatter().dump(layoutdict, layoutlist, outputfile)