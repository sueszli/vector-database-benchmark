import logging
logging.basicConfig(level=logging.DEBUG)

def legacy():
    if False:
        for i in range(10):
            print('nop')
    from slack_sdk.models.blocks import SectionBlock
    from slack_sdk.models.blocks.basic_components import TextObject
    fields = []
    fields.append(TextObject(text='...', type='mrkdwn'))
    block = SectionBlock(text='', fields=fields)
    assert block is not None
from slack_sdk.models.blocks import SectionBlock, TextObject
fields = []
fields.append(TextObject(text='...', type='mrkdwn'))
block = SectionBlock(text='', fields=fields)
assert block is not None