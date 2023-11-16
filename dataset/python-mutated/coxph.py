from h2o.model import ModelBase

class H2OCoxPHModel(ModelBase):

    def formula(self):
        if False:
            i = 10
            return i + 15
        'Survival formula.'
        return self._model_json['output']['formula']

    def concordance(self):
        if False:
            return 10
        'Concordance'
        return self._model_json['output']['concordance']

    def coefficients_table(self):
        if False:
            i = 10
            return i + 15
        'Coefficients table.'
        return self._model_json['output']['coefficients_table']

    def summary(self):
        if False:
            return 10
        'legacy behaviour as for some reason, CoxPH is formatting summary differently than other models'
        return self._summary()

    def get_summary(self):
        if False:
            return 10
        output = self._model_json['output']
        return 'Call:\n{formula}\n{coefs}\nLikelihood ratio test={lrt:f}\nConcordance={concordance:f}\nn={n:d}, number of events={tot_events:d}\n'.format(formula=self.formula(), coefs=self.coefficients_table(), lrt=output['loglik_test'], concordance=self.concordance(), n=output['n'], tot_events=output['total_event'])

class H2OCoxPHMojoModel(ModelBase):
    pass