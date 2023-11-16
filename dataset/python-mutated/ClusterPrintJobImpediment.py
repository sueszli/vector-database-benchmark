from ..BaseModel import BaseModel

class ClusterPrintJobImpediment(BaseModel):
    """Class representing the reasons that prevent this job from being printed on the associated printer"""

    def __init__(self, translation_key: str, severity: int, **kwargs) -> None:
        if False:
            return 10
        "Creates a new print job constraint.\n\n        :param translation_key: A string indicating a reason the print cannot be printed,\n        such as 'does_not_fit_in_build_volume'\n        :param severity: A number indicating the severity of the problem, with higher being more severe\n        "
        self.translation_key = translation_key
        self.severity = severity
        super().__init__(**kwargs)