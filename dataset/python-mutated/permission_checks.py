"""Permission checks for tasks."""
__all__ = ('user_id_matches', 'user_id_matches_or_superuser')

def user_id_matches(request, state, context):
    if False:
        for i in range(10):
            print('nop')
    user_id = context.get('user_id', None)
    if user_id is not None and request.user.is_authenticated:
        if request.user.id == user_id:
            return True
    return False

def user_id_matches_or_superuser(request, state, context):
    if False:
        for i in range(10):
            print('nop')
    return request.user.is_superuser or user_id_matches(request, state, context)