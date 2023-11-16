from collections import OrderedDict
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel, QGridLayout
import scipy.sparse as sp
from Orange.data import Table
from Orange.modelling import SVMLearner, NuSVMLearner
from Orange.widgets import gui
from Orange.widgets.widget import Msg
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

class OWSVM(OWBaseLearner):
    name = 'SVM'
    description = 'Support Vector Machines map inputs to higher-dimensional feature spaces.'
    icon = 'icons/SVM.svg'
    replaces = ['Orange.widgets.classify.owsvmclassification.OWSVMClassification', 'Orange.widgets.regression.owsvmregression.OWSVMRegression']
    priority = 50
    keywords = 'svm, support vector machines'
    LEARNER = SVMLearner

    class Outputs(OWBaseLearner.Outputs):
        support_vectors = Output('Support Vectors', Table, explicit=True, replaces=['Support vectors'])

    class Warning(OWBaseLearner.Warning):
        sparse_data = Msg('Input data is sparse, default preprocessing is to scale it.')
    (SVM, Nu_SVM) = range(2)
    svm_type = Setting(SVM)
    C = Setting(1.0)
    epsilon = Setting(0.1)
    nu_C = Setting(1.0)
    nu = Setting(0.5)
    (Linear, Poly, RBF, Sigmoid) = range(4)
    kernel_type = Setting(RBF)
    degree = Setting(3)
    gamma = Setting(0.0)
    coef0 = Setting(1.0)
    tol = Setting(0.001)
    limit_iter = Setting(True)
    max_iter = Setting(100)
    _default_gamma = 'auto'
    kernels = (('Linear', 'x⋅y'), ('Polynomial', '(g x⋅y + c)<sup>d</sup>'), ('RBF', 'exp(-g|x-y|²)'), ('Sigmoid', 'tanh(g x⋅y + c)'))

    def add_main_layout(self):
        if False:
            return 10
        self._add_type_box()
        self._add_kernel_box()
        self._add_optimization_box()
        self._show_right_kernel()

    def _add_type_box(self):
        if False:
            print('Hello World!')
        form = QGridLayout()
        self.type_box = box = gui.radioButtonsInBox(self.controlArea, self, 'svm_type', [], box='SVM Type', orientation=form, callback=self._update_type)
        self.epsilon_radio = gui.appendRadioButton(box, 'SVM', addToLayout=False)
        self.c_spin = gui.doubleSpin(box, self, 'C', 0.1, 512.0, 0.1, decimals=2, alignment=Qt.AlignRight, addToLayout=False, callback=self.settings_changed)
        self.epsilon_spin = gui.doubleSpin(box, self, 'epsilon', 0.1, 512.0, 0.1, decimals=2, alignment=Qt.AlignRight, addToLayout=False, callback=self.settings_changed)
        form.addWidget(self.epsilon_radio, 0, 0, Qt.AlignLeft)
        form.addWidget(QLabel('Cost (C):'), 0, 1, Qt.AlignRight)
        form.addWidget(self.c_spin, 0, 2)
        form.addWidget(QLabel('Regression loss epsilon (ε):'), 1, 1, Qt.AlignRight)
        form.addWidget(self.epsilon_spin, 1, 2)
        self.nu_radio = gui.appendRadioButton(box, 'ν-SVM', addToLayout=False)
        self.nu_C_spin = gui.doubleSpin(box, self, 'nu_C', 0.1, 512.0, 0.1, decimals=2, alignment=Qt.AlignRight, addToLayout=False, callback=self.settings_changed)
        self.nu_spin = gui.doubleSpin(box, self, 'nu', 0.05, 1.0, 0.05, decimals=2, alignment=Qt.AlignRight, addToLayout=False, callback=self.settings_changed)
        form.addWidget(self.nu_radio, 2, 0, Qt.AlignLeft)
        form.addWidget(QLabel('Regression cost (C):'), 2, 1, Qt.AlignRight)
        form.addWidget(self.nu_C_spin, 2, 2)
        form.addWidget(QLabel('Complexity bound (ν):'), 3, 1, Qt.AlignRight)
        form.addWidget(self.nu_spin, 3, 2)
        self._update_type()

    def _update_type(self):
        if False:
            for i in range(10):
                print('nop')
        if self.svm_type == self.SVM:
            self.c_spin.setEnabled(True)
            self.epsilon_spin.setEnabled(True)
            self.nu_C_spin.setEnabled(False)
            self.nu_spin.setEnabled(False)
        else:
            self.c_spin.setEnabled(False)
            self.epsilon_spin.setEnabled(False)
            self.nu_C_spin.setEnabled(True)
            self.nu_spin.setEnabled(True)
        self.settings_changed()

    def _add_kernel_box(self):
        if False:
            for i in range(10):
                print('nop')
        self.kernel_eq = self.kernels[-1][1]
        box = gui.hBox(self.controlArea, 'Kernel')
        self.kernel_box = buttonbox = gui.radioButtonsInBox(box, self, 'kernel_type', btnLabels=[k[0] for k in self.kernels], callback=self._on_kernel_changed)
        buttonbox.layout().setSpacing(10)
        gui.rubber(buttonbox)
        parambox = gui.vBox(box)
        gui.label(parambox, self, 'Kernel: %(kernel_eq)s')
        common = dict(orientation=Qt.Horizontal, callback=self.settings_changed, alignment=Qt.AlignRight, controlWidth=80)
        spbox = gui.hBox(parambox)
        gui.rubber(spbox)
        inbox = gui.vBox(spbox)
        gamma = gui.doubleSpin(inbox, self, 'gamma', 0.0, 10.0, 0.01, label=' g: ', **common)
        gamma.setSpecialValueText(self._default_gamma)
        coef0 = gui.doubleSpin(inbox, self, 'coef0', 0.0, 10.0, 0.01, label=' c: ', **common)
        degree = gui.doubleSpin(inbox, self, 'degree', 0.0, 10.0, 0.5, label=' d: ', **common)
        self._kernel_params = [gamma, coef0, degree]
        gui.rubber(parambox)
        box.layout().activate()
        box.setFixedHeight(box.sizeHint().height())
        box.setMinimumWidth(box.sizeHint().width())

    def _add_optimization_box(self):
        if False:
            while True:
                i = 10
        self.optimization_box = gui.vBox(self.controlArea, 'Optimization Parameters')
        self.tol_spin = gui.doubleSpin(self.optimization_box, self, 'tol', 0.0001, 1.0, 0.0001, label='Numerical tolerance: ', alignment=Qt.AlignRight, controlWidth=100, callback=self.settings_changed)
        self.max_iter_spin = gui.spin(self.optimization_box, self, 'max_iter', 5, 1000000, 50, label='Iteration limit: ', checked='limit_iter', alignment=Qt.AlignRight, controlWidth=100, callback=self.settings_changed, checkCallback=self.settings_changed)

    def _show_right_kernel(self):
        if False:
            i = 10
            return i + 15
        enabled = [[False, False, False], [True, True, True], [True, False, False], [True, True, False]]
        self.kernel_eq = self.kernels[self.kernel_type][1]
        mask = enabled[self.kernel_type]
        for (spin, enabled) in zip(self._kernel_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def update_model(self):
        if False:
            for i in range(10):
                print('nop')
        super().update_model()
        sv = None
        if self.model is not None:
            sv = self.data[self.model.skl_model.support_]
        self.Outputs.support_vectors.send(sv)

    def _on_kernel_changed(self):
        if False:
            print('Hello World!')
        self._show_right_kernel()
        self.settings_changed()

    @OWBaseLearner.Inputs.data
    def set_data(self, data):
        if False:
            i = 10
            return i + 15
        self.Warning.sparse_data.clear()
        super().set_data(data)
        if self.data and sp.issparse(self.data.X):
            self.Warning.sparse_data()

    def create_learner(self):
        if False:
            return 10
        kernel = ['linear', 'poly', 'rbf', 'sigmoid'][self.kernel_type]
        common_args = {'kernel': kernel, 'degree': self.degree, 'gamma': self.gamma or self._default_gamma, 'coef0': self.coef0, 'probability': True, 'tol': self.tol, 'max_iter': self.max_iter if self.limit_iter else -1, 'preprocessors': self.preprocessors}
        if self.svm_type == self.SVM:
            return SVMLearner(C=self.C, epsilon=self.epsilon, **common_args)
        else:
            return NuSVMLearner(nu=self.nu, C=self.nu_C, **common_args)

    def get_learner_parameters(self):
        if False:
            print('Hello World!')
        items = OrderedDict()
        if self.svm_type == self.SVM:
            items['SVM type'] = 'SVM, C={}, ε={}'.format(self.C, self.epsilon)
        else:
            items['SVM type'] = 'ν-SVM, ν={}, C={}'.format(self.nu, self.nu_C)
        self._report_kernel_parameters(items)
        items['Numerical tolerance'] = '{:.6}'.format(self.tol)
        items['Iteration limt'] = self.max_iter if self.limit_iter else 'unlimited'
        return items

    def _report_kernel_parameters(self, items):
        if False:
            i = 10
            return i + 15
        gamma = self.gamma or self._default_gamma
        if self.kernel_type == 0:
            items['Kernel'] = 'Linear'
        elif self.kernel_type == 1:
            items['Kernel'] = 'Polynomial, ({g:.4} x⋅y + {c:.4})<sup>{d}</sup>'.format(g=gamma, c=self.coef0, d=self.degree)
        elif self.kernel_type == 2:
            items['Kernel'] = 'RBF, exp(-{:.4}|x-y|²)'.format(gamma)
        else:
            items['Kernel'] = 'Sigmoid, tanh({g:.4} x⋅y + {c:.4})'.format(g=gamma, c=self.coef0)
if __name__ == '__main__':
    WidgetPreview(OWSVM).run(Table('iris'))