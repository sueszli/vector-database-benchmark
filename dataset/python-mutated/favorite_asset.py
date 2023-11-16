from rest_framework_bulk import BulkModelViewSet
from common.permissions import IsValidUser
from orgs.utils import tmp_to_root_org
from ..models import FavoriteAsset
from ..serializers import FavoriteAssetSerializer
__all__ = ['FavoriteAssetViewSet']

class FavoriteAssetViewSet(BulkModelViewSet):
    serializer_class = FavoriteAssetSerializer
    permission_classes = (IsValidUser,)
    filterset_fields = ['asset']

    def dispatch(self, request, *args, **kwargs):
        if False:
            print('Hello World!')
        with tmp_to_root_org():
            return super().dispatch(request, *args, **kwargs)

    def get_queryset(self):
        if False:
            while True:
                i = 10
        queryset = FavoriteAsset.objects.filter(user=self.request.user)
        return queryset

    def allow_bulk_destroy(self, qs, filtered):
        if False:
            i = 10
            return i + 15
        return filtered.count() == 1