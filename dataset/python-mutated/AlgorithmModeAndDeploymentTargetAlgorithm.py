from AlgorithmImports import *

class AlgorithmModeAndDeploymentTargetAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.Debug(f'Algorithm Mode: {self.AlgorithmMode}. Is Live Mode: {self.LiveMode}. Deployment Target: {self.DeploymentTarget}.')
        if self.AlgorithmMode != AlgorithmMode.Backtesting:
            raise Exception(f'Algorithm mode is not backtesting. Actual: {self.AlgorithmMode}')
        if self.LiveMode:
            raise Exception('Algorithm should not be live')
        if self.DeploymentTarget != DeploymentTarget.LocalPlatform:
            raise Exception(f'Algorithm deployment target is not local. Actual{self.DeploymentTarget}')
        self.Quit()