from django.template.loader import render_to_string
from wagtail.admin.compare import ForeignObjectComparison

class ImageFieldComparison(ForeignObjectComparison):

    def htmldiff(self):
        if False:
            for i in range(10):
                print('nop')
        (image_a, image_b) = self.get_objects()
        return render_to_string('wagtailimages/widgets/compare.html', {'image_a': image_a, 'image_b': image_b})