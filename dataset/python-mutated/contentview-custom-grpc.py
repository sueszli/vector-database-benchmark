"""
Add a custom version of the gRPC/protobuf content view, which parses
protobuf messages based on a user defined rule set.

"""
from mitmproxy import contentviews
from mitmproxy.addonmanager import Loader
from mitmproxy.contentviews.grpc import ProtoParser
from mitmproxy.contentviews.grpc import ViewConfig
from mitmproxy.contentviews.grpc import ViewGrpcProtobuf
config: ViewConfig = ViewConfig()
config.parser_rules = [ProtoParser.ParserRuleRequest(name='Geo coordinate lookup request', filter='example\\.com.*/ReverseGeocode', field_definitions=[ProtoParser.ParserFieldDefinition(tag='1', name='position'), ProtoParser.ParserFieldDefinition(tag='1.1', name='latitude', intended_decoding=ProtoParser.DecodedTypes.double), ProtoParser.ParserFieldDefinition(tag='1.2', name='longitude', intended_decoding=ProtoParser.DecodedTypes.double), ProtoParser.ParserFieldDefinition(tag='3', name='country'), ProtoParser.ParserFieldDefinition(tag='7', name='app')]), ProtoParser.ParserRuleResponse(name='Geo coordinate lookup response', filter='example\\.com.*/ReverseGeocode', field_definitions=[ProtoParser.ParserFieldDefinition(tag='1.2', name='address'), ProtoParser.ParserFieldDefinition(tag='1.3', name='address array element'), ProtoParser.ParserFieldDefinition(tag='1.3.1', name='unknown bytes', intended_decoding=ProtoParser.DecodedTypes.bytes), ProtoParser.ParserFieldDefinition(tag='1.3.2', name='element value long'), ProtoParser.ParserFieldDefinition(tag='1.3.3', name='element value short'), ProtoParser.ParserFieldDefinition(tag='', tag_prefixes=['1.5.1', '1.5.3', '1.5.4', '1.5.5', '1.5.6'], name='position'), ProtoParser.ParserFieldDefinition(tag='.1', tag_prefixes=['1.5.1', '1.5.3', '1.5.4', '1.5.5', '1.5.6'], name='latitude', intended_decoding=ProtoParser.DecodedTypes.double), ProtoParser.ParserFieldDefinition(tag='.2', tag_prefixes=['1.5.1', '1.5.3', '1.5.4', '1.5.5', '1.5.6'], name='longitude', intended_decoding=ProtoParser.DecodedTypes.double), ProtoParser.ParserFieldDefinition(tag='7', name='app')])]

class ViewGrpcWithRules(ViewGrpcProtobuf):
    name = 'customized gRPC/protobuf'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config=config)

    def __call__(self, *args, **kwargs) -> contentviews.TViewResult:
        if False:
            print('Hello World!')
        (heading, lines) = super().__call__(*args, **kwargs)
        return (heading + ' (addon with custom rules)', lines)

    def render_priority(self, *args, **kwargs) -> float:
        if False:
            i = 10
            return i + 15
        s_prio = super().render_priority(*args, **kwargs)
        return s_prio + 1 if s_prio > 0 else s_prio
view = ViewGrpcWithRules()

def load(loader: Loader):
    if False:
        while True:
            i = 10
    contentviews.add(view)

def done():
    if False:
        for i in range(10):
            print('nop')
    contentviews.remove(view)