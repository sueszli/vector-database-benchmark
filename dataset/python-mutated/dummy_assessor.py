from nni.assessor import Assessor, AssessResult

class DummyAssessor(Assessor):

    def assess_trial(self, trial_job_id, trial_history):
        if False:
            print('Hello World!')
        return AssessResult.Good