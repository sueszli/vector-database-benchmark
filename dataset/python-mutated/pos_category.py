from odoo import api, fields, models, tools, _

class PosCategory(models.Model):
    _name = 'pos.category'
    _description = 'Public Category'
    _order = 'sequence, name'

    @api.constrains('parent_id')
    def _check_category_recursion(self):
        if False:
            print('Hello World!')
        if not self._check_recursion():
            raise ValueError(_('Error ! You cannot create recursive categories.'))
    name = fields.Char(required=True, translate=True)
    parent_id = fields.Many2one('pos.category', string='Parent Category', index=True)
    child_id = fields.One2many('pos.category', 'parent_id', string='Children Categories')
    sequence = fields.Integer(help='Gives the sequence order when displaying a list of product categories.')
    image = fields.Binary(attachment=True, help='This field holds the image used as image for the cateogry, limited to 1024x1024px.')
    image_medium = fields.Binary(string='Medium-sized image', attachment=True, help='Medium-sized image of the category. It is automatically resized as a 128x128px image, with aspect ratio preserved. Use this field in form views or some kanban views.')
    image_small = fields.Binary(string='Small-sized image', attachment=True, help='Small-sized image of the category. It is automatically resized as a 64x64px image, with aspect ratio preserved. Use this field anywhere a small image is required.')

    @api.model
    def create(self, vals):
        if False:
            print('Hello World!')
        tools.image_resize_images(vals)
        return super(PosCategory, self).create(vals)

    @api.multi
    def write(self, vals):
        if False:
            return 10
        tools.image_resize_images(vals)
        return super(PosCategory, self).write(vals)

    @api.multi
    def name_get(self):
        if False:
            for i in range(10):
                print('nop')

        def get_names(cat):
            if False:
                i = 10
                return i + 15
            res = []
            while cat:
                res.append(cat.name)
                cat = cat.parent_id
            return res
        return [(cat.id, ' / '.join(reversed(get_names(cat)))) for cat in self]