import json

def _get_json_content(obj):
    if False:
        while True:
            i = 10
    'Event mixin to have methods that are common to different Event types\n    like CloudEvent, EventGridEvent etc.\n\n    :param obj: The object to get the JSON content from.\n    :type obj: any\n    :return: The JSON content of the object.\n    :rtype: dict\n    :raises ValueError if JSON content cannot be loaded from the object\n    '
    msg = 'Failed to load JSON content from the object.'
    try:
        return json.loads(obj.content)
    except ValueError as err:
        raise ValueError(msg) from err
    except AttributeError:
        try:
            return json.loads(next(obj.body))[0]
        except KeyError:
            return json.loads(next(obj.body))
        except ValueError as err:
            raise ValueError(msg) from err
        except:
            try:
                return json.loads(obj)
            except ValueError as err:
                raise ValueError(msg) from err