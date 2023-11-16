import h2o
from h2o.model import ModelBase
from h2o.plot import decorate_plot_result, get_matplotlib_pyplot, RAISE_ON_FIGURE_ACCESS
from h2o.utils.compatibility import *
from h2o.utils.shared_utils import can_use_pandas

class H2ODimReductionModel(ModelBase):
    """
    Dimension reduction model, such as PCA or GLRM.
    """

    def varimp(self, use_pandas=False):
        if False:
            print('Hello World!')
        '\n        Return the Importance of components associated with a PCA model.\n\n        :param bool use_pandas: If ``True``, then the variable importances will be returned as a pandas data frame. (Default: ``False``)\n        '
        model = self._model_json['output']
        if 'importance' in list(model.keys()) and model['importance']:
            vals = model['importance'].cell_values
            header = model['importance'].col_header
            if use_pandas and can_use_pandas():
                import pandas
                return pandas.DataFrame(vals, columns=header)
            else:
                return vals
        else:
            print("Warning: This model doesn't have importances of components.")

    def num_iterations(self):
        if False:
            while True:
                i = 10
        'Get the number of iterations that it took to converge or reach max iterations.'
        o = self._model_json['output']
        return o['model_summary']['number_of_iterations'][0]

    def objective(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the final value of the objective function.'
        o = self._model_json['output']
        return o['model_summary']['final_objective_value'][0]

    def final_step(self):
        if False:
            print('Hello World!')
        'Get the final step size for the model.'
        o = self._model_json['output']
        return o['model_summary']['final_step_size'][0]

    def archetypes(self):
        if False:
            i = 10
            return i + 15
        'The archetypes (Y) of the GLRM model.'
        o = self._model_json['output']
        yvals = o['archetypes'].cell_values
        archetypes = []
        for (yidx, yval) in enumerate(yvals):
            archetypes.append(list(yvals[yidx])[1:])
        return archetypes

    def reconstruct(self, test_data, reverse_transform=False):
        if False:
            print('Hello World!')
        '\n        Reconstruct the training data from the model and impute all missing values.\n\n        :param H2OFrame test_data: The dataset upon which the model was trained.\n        :param bool reverse_transform: Whether the transformation of the training data during model-building\n            should be reversed on the reconstructed frame.\n\n        :returns: the approximate reconstruction of the training data.\n        '
        if test_data is None or test_data.nrow == 0:
            raise ValueError('Must specify test data')
        j = h2o.api('POST /3/Predictions/models/%s/frames/%s' % (self.model_id, test_data.frame_id), data={'reconstruct_train': True, 'reverse_transform': reverse_transform})
        return h2o.get_frame(j['model_metrics'][0]['predictions']['frame_id']['name'])

    def proj_archetypes(self, test_data, reverse_transform=False):
        if False:
            while True:
                i = 10
        "\n        Convert archetypes of the model into original feature space.\n\n        :param H2OFrame test_data: The dataset upon which the model was trained.\n        :param bool reverse_transform: Whether the transformation of the training data during model-building\n            should be reversed on the projected archetypes.\n\n        :returns: model archetypes projected back into the original training data's feature space.\n        "
        if test_data is None or test_data.nrow == 0:
            raise ValueError('Must specify test data')
        j = h2o.api('POST /3/Predictions/models/%s/frames/%s' % (self.model_id, test_data.frame_id), data={'project_archetypes': True, 'reverse_transform': reverse_transform})
        return h2o.get_frame(j['model_metrics'][0]['predictions']['frame_id']['name'])

    def screeplot(self, type='barplot', server=False, save_plot_path=None):
        if False:
            while True:
                i = 10
        '\n        Produce the scree plot.\n\n        Library ``matplotlib`` is required for this function.\n\n        :param str type: either ``"barplot"`` or ``"lines"``.\n        :param bool server: if ``True``, set ``server`` settings to matplotlib and do not show the graph.\n        :param save_plot_path: a path to save the plot via using matplotlib function savefig.\n        \n        :returns: Object that contains the resulting scree plot (can be accessed like ``result.figure()``).\n        '
        plt = get_matplotlib_pyplot(server)
        if plt is None:
            return decorate_plot_result(figure=RAISE_ON_FIGURE_ACCESS)
        fig = plt.figure()
        variances = [s ** 2 for s in self._model_json['output']['importance'].cell_values[0][1:]]
        plt.xlabel('Components')
        plt.ylabel('Variances')
        plt.title('Scree Plot')
        plt.xticks(list(range(1, len(variances) + 1)))
        if type == 'barplot':
            plt.bar(list(range(1, len(variances) + 1)), variances)
        elif type == 'lines':
            plt.plot(list(range(1, len(variances) + 1)), variances, 'b--')
        if save_plot_path is not None:
            plt.savefig(fname=save_plot_path)
        if not server:
            plt.show()
        return decorate_plot_result(figure=fig)