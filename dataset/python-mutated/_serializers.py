from rest_framework import serializers

class UsernameSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    username = serializers.CharField()
    real_name = serializers.SerializerMethodField()

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.need_real_name = kwargs.pop('need_real_name', False)
        super().__init__(*args, **kwargs)

    def get_real_name(self, obj):
        if False:
            return 10
        return obj.userprofile.real_name if self.need_real_name else None