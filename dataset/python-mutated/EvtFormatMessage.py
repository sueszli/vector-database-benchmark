import sys
import win32evtlog

def main():
    if False:
        return 10
    path = 'System'
    num_events = 5
    if len(sys.argv) > 2:
        path = sys.argv[1]
        num_events = int(sys.argv[2])
    elif len(sys.argv) > 1:
        path = sys.argv[1]
    query = win32evtlog.EvtQuery(path, win32evtlog.EvtQueryForwardDirection)
    events = win32evtlog.EvtNext(query, num_events)
    context = win32evtlog.EvtCreateRenderContext(win32evtlog.EvtRenderContextSystem)
    for (i, event) in enumerate(events, 1):
        result = win32evtlog.EvtRender(event, win32evtlog.EvtRenderEventValues, Context=context)
        print(f'Event {i}')
        (level_value, level_variant) = result[win32evtlog.EvtSystemLevel]
        if level_variant != win32evtlog.EvtVarTypeNull:
            if level_value == 1:
                print('    Level: CRITICAL')
            elif level_value == 2:
                print('    Level: ERROR')
            elif level_value == 3:
                print('    Level: WARNING')
            elif level_value == 4:
                print('    Level: INFO')
            elif level_value == 5:
                print('    Level: VERBOSE')
            else:
                print('    Level: UNKNOWN')
        (time_created_value, time_created_variant) = result[win32evtlog.EvtSystemTimeCreated]
        if time_created_variant != win32evtlog.EvtVarTypeNull:
            print(f'    Timestamp: {time_created_value.isoformat()}')
        (computer_value, computer_variant) = result[win32evtlog.EvtSystemComputer]
        if computer_variant != win32evtlog.EvtVarTypeNull:
            print(f'    FQDN: {computer_value}')
        (provider_name_value, provider_name_variant) = result[win32evtlog.EvtSystemProviderName]
        if provider_name_variant != win32evtlog.EvtVarTypeNull:
            print(f'    Provider: {provider_name_value}')
            try:
                metadata = win32evtlog.EvtOpenPublisherMetadata(provider_name_value)
            except Exception:
                pass
            else:
                try:
                    message = win32evtlog.EvtFormatMessage(metadata, event, win32evtlog.EvtFormatMessageEvent)
                except Exception:
                    pass
                else:
                    try:
                        print(f'    Message: {message}')
                    except UnicodeEncodeError:
                        print(' Failed to decode:', repr(message))
if __name__ == '__main__':
    main()