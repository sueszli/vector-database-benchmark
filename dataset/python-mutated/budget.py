from sqlalchemy import Column, Integer, String, Float
from superagi.models.base_model import DBBaseModel

class Budget(DBBaseModel):
    """
    Model representing a budget.

    Attributes:
        id (Integer): The primary key of the budget.
        budget (Float): The budget value.
        cycle (String): The cycle of the budget.
    """
    __tablename__ = 'budgets'
    id = Column(Integer, primary_key=True)
    budget = Column(Float)
    cycle = Column(String)

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Returns a string representation of the Budget object.\n\n        Returns:\n            str: String representation of the Budget object.\n        '
        return f"Budget(id={self.id}, budget={self.budget}, cycle='{self.cycle}')"