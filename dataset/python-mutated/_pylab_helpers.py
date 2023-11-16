"""
Manage figures for the pyplot interface.
"""
import atexit
from collections import OrderedDict

class Gcf:
    """
    Singleton to maintain the relation between figures and their managers, and
    keep track of and "active" figure and manager.

    The canvas of a figure created through pyplot is associated with a figure
    manager, which handles the interaction between the figure and the backend.
    pyplot keeps track of figure managers using an identifier, the "figure
    number" or "manager number" (which can actually be any hashable value);
    this number is available as the :attr:`number` attribute of the manager.

    This class is never instantiated; it consists of an `OrderedDict` mapping
    figure/manager numbers to managers, and a set of class methods that
    manipulate this `OrderedDict`.

    Attributes
    ----------
    figs : OrderedDict
        `OrderedDict` mapping numbers to managers; the active manager is at the
        end.
    """
    figs = OrderedDict()

    @classmethod
    def get_fig_manager(cls, num):
        if False:
            i = 10
            return i + 15
        '\n        If manager number *num* exists, make it the active one and return it;\n        otherwise return *None*.\n        '
        manager = cls.figs.get(num, None)
        if manager is not None:
            cls.set_active(manager)
        return manager

    @classmethod
    def destroy(cls, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        Destroy manager *num* -- either a manager instance or a manager number.\n\n        In the interactive backends, this is bound to the window "destroy" and\n        "delete" events.\n\n        It is recommended to pass a manager instance, to avoid confusion when\n        two managers share the same number.\n        '
        if all((hasattr(num, attr) for attr in ['num', 'destroy'])):
            manager = num
            if cls.figs.get(manager.num) is manager:
                cls.figs.pop(manager.num)
        else:
            try:
                manager = cls.figs.pop(num)
            except KeyError:
                return
        if hasattr(manager, '_cidgcf'):
            manager.canvas.mpl_disconnect(manager._cidgcf)
        manager.destroy()
        del manager, num

    @classmethod
    def destroy_fig(cls, fig):
        if False:
            for i in range(10):
                print('nop')
        'Destroy figure *fig*.'
        num = next((manager.num for manager in cls.figs.values() if manager.canvas.figure == fig), None)
        if num is not None:
            cls.destroy(num)

    @classmethod
    def destroy_all(cls):
        if False:
            while True:
                i = 10
        'Destroy all figures.'
        for manager in list(cls.figs.values()):
            manager.canvas.mpl_disconnect(manager._cidgcf)
            manager.destroy()
        cls.figs.clear()

    @classmethod
    def has_fignum(cls, num):
        if False:
            while True:
                i = 10
        'Return whether figure number *num* exists.'
        return num in cls.figs

    @classmethod
    def get_all_fig_managers(cls):
        if False:
            i = 10
            return i + 15
        'Return a list of figure managers.'
        return list(cls.figs.values())

    @classmethod
    def get_num_fig_managers(cls):
        if False:
            while True:
                i = 10
        'Return the number of figures being managed.'
        return len(cls.figs)

    @classmethod
    def get_active(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return the active manager, or *None* if there is no manager.'
        return next(reversed(cls.figs.values())) if cls.figs else None

    @classmethod
    def _set_new_active_manager(cls, manager):
        if False:
            while True:
                i = 10
        'Adopt *manager* into pyplot and make it the active manager.'
        if not hasattr(manager, '_cidgcf'):
            manager._cidgcf = manager.canvas.mpl_connect('button_press_event', lambda event: cls.set_active(manager))
        fig = manager.canvas.figure
        fig.number = manager.num
        label = fig.get_label()
        if label:
            manager.set_window_title(label)
        cls.set_active(manager)

    @classmethod
    def set_active(cls, manager):
        if False:
            while True:
                i = 10
        'Make *manager* the active manager.'
        cls.figs[manager.num] = manager
        cls.figs.move_to_end(manager.num)

    @classmethod
    def draw_all(cls, force=False):
        if False:
            return 10
        '\n        Redraw all stale managed figures, or, if *force* is True, all managed\n        figures.\n        '
        for manager in cls.get_all_fig_managers():
            if force or manager.canvas.figure.stale:
                manager.canvas.draw_idle()
atexit.register(Gcf.destroy_all)