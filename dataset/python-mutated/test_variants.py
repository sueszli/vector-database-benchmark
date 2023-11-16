from . import common

class TestVariants(common.TestProductCommon):

    def setUp(self):
        if False:
            while True:
                i = 10
        res = super(TestVariants, self).setUp()
        self.size_attr = self.env['product.attribute'].create({'name': 'Size'})
        self.size_attr_value_s = self.env['product.attribute.value'].create({'name': 'S', 'attribute_id': self.size_attr.id})
        self.size_attr_value_m = self.env['product.attribute.value'].create({'name': 'M', 'attribute_id': self.size_attr.id})
        self.size_attr_value_l = self.env['product.attribute.value'].create({'name': 'L', 'attribute_id': self.size_attr.id})
        return res

    def test_variants_creation_mono(self):
        if False:
            for i in range(10):
                print('nop')
        test_template = self.env['product.template'].create({'name': 'Sofa', 'uom_id': self.uom_unit.id, 'uom_po_id': self.uom_unit.id, 'attribute_line_ids': [(0, 0, {'attribute_id': self.size_attr.id, 'value_ids': [(4, self.size_attr_value_s.id)]})]})
        self.assertEqual(len(test_template.product_variant_ids), 1)
        self.assertEqual(test_template.product_variant_ids.attribute_value_ids, self.size_attr_value_s)

    def test_variants_creation_mono_double(self):
        if False:
            print('Hello World!')
        test_template = self.env['product.template'].create({'name': 'Sofa', 'uom_id': self.uom_unit.id, 'uom_po_id': self.uom_unit.id, 'attribute_line_ids': [(0, 0, {'attribute_id': self.prod_att_1.id, 'value_ids': [(4, self.prod_attr1_v2.id)]}), (0, 0, {'attribute_id': self.size_attr.id, 'value_ids': [(4, self.size_attr_value_s.id)]})]})
        self.assertEqual(len(test_template.product_variant_ids), 1)
        self.assertEqual(test_template.product_variant_ids.attribute_value_ids, self.size_attr_value_s + self.prod_attr1_v2)

    def test_variants_creation_mono_multi(self):
        if False:
            return 10
        test_template = self.env['product.template'].create({'name': 'Sofa', 'uom_id': self.uom_unit.id, 'uom_po_id': self.uom_unit.id, 'attribute_line_ids': [(0, 0, {'attribute_id': self.prod_att_1.id, 'value_ids': [(4, self.prod_attr1_v2.id)]}), (0, 0, {'attribute_id': self.size_attr.id, 'value_ids': [(4, self.size_attr_value_s.id), (4, self.size_attr_value_m.id)]})]})
        self.assertEqual(len(test_template.product_variant_ids), 2)
        for value in self.size_attr_value_s + self.size_attr_value_m:
            products = self.env['product.product'].search([('product_tmpl_id', '=', test_template.id), ('attribute_value_ids', 'in', value.id), ('attribute_value_ids', 'in', self.prod_attr1_v2.id)])
            self.assertEqual(len(products), 1)

    def test_variants_creation_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        test_template = self.env['product.template'].create({'name': 'Sofa', 'uom_id': self.uom_unit.id, 'uom_po_id': self.uom_unit.id, 'attribute_line_ids': [(0, 0, {'attribute_id': self.prod_att_1.id, 'value_ids': [(4, self.prod_attr1_v1.id), (4, self.prod_attr1_v2.id)]}), (0, 0, {'attribute_id': self.size_attr.id, 'value_ids': [(4, self.size_attr_value_s.id), (4, self.size_attr_value_m.id), (4, self.size_attr_value_l.id)]})]})
        self.assertEqual(len(test_template.product_variant_ids), 6)
        for value_1 in self.prod_attr1_v1 + self.prod_attr1_v2:
            for value_2 in self.size_attr_value_m + self.size_attr_value_m + self.size_attr_value_l:
                products = self.env['product.product'].search([('product_tmpl_id', '=', test_template.id), ('attribute_value_ids', 'in', value_1.id), ('attribute_value_ids', 'in', value_2.id)])
                self.assertEqual(len(products), 1)

    def test_variants_creation_multi_update(self):
        if False:
            i = 10
            return i + 15
        test_template = self.env['product.template'].create({'name': 'Sofa', 'uom_id': self.uom_unit.id, 'uom_po_id': self.uom_unit.id, 'attribute_line_ids': [(0, 0, {'attribute_id': self.prod_att_1.id, 'value_ids': [(4, self.prod_attr1_v1.id), (4, self.prod_attr1_v2.id)]}), (0, 0, {'attribute_id': self.size_attr.id, 'value_ids': [(4, self.size_attr_value_s.id), (4, self.size_attr_value_m.id)]})]})
        size_attribute_line = test_template.attribute_line_ids.filtered(lambda line: line.attribute_id == self.size_attr)
        test_template.write({'attribute_line_ids': [(1, size_attribute_line.id, {'value_ids': [(4, self.size_attr_value_l.id)]})]})