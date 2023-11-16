from odoo import api, models

class IrNeedactionMixin(models.AbstractModel):
    """Mixin class for objects using the need action feature.

    Need action feature can be used by models that have to be able to
    signal that an action is required on a particular record. If in
    the business logic an action must be performed by somebody, for
    instance validation by a manager, this mechanism allows to set a
    list of users asked to perform an action.

    Models using the 'need_action' feature should override the
    ``_needaction_domain_get`` method. This method returns a
    domain to filter records requiring an action for a specific user.

    This class also offers several global services:
    - ``_needaction_count``: returns the number of actions uid has to perform
    """
    _name = 'ir.needaction_mixin'
    _needaction = True

    @api.model
    def _needaction_domain_get(self):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the domain to filter records that require an action\n            :return: domain or False is no action\n        '
        return False

    @api.model
    def _needaction_count(self, domain=None):
        if False:
            for i in range(10):
                print('nop')
        ' Get the number of actions uid has to perform. '
        dom = self._needaction_domain_get()
        if not dom:
            return 0
        res = self.search((domain or []) + dom, limit=100, order='id DESC')
        return len(res)