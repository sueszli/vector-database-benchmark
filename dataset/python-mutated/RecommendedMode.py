from PyQt6.QtCore import QObject, pyqtSlot
from cura import CuraApplication

class RecommendedMode(QObject):

    @pyqtSlot(bool)
    def setAdhesion(self, checked: bool) -> None:
        if False:
            print('Hello World!')
        application = CuraApplication.CuraApplication.getInstance()
        global_stack = application.getMachineManager().activeMachine
        if global_stack is None:
            return
        adhesion_type_key = 'adhesion_type'
        user_changes_container = global_stack.userChanges
        if adhesion_type_key in user_changes_container.getAllKeys():
            user_changes_container.removeInstance(adhesion_type_key)
        value = global_stack.getProperty(adhesion_type_key, 'value')
        if checked:
            if value in ('skirt', 'none'):
                value = 'brim'
        elif value not in ('skirt', 'none'):
            value = 'skirt'
        user_changes_container.setProperty(adhesion_type_key, 'value', value)
__all__ = ['RecommendedMode']