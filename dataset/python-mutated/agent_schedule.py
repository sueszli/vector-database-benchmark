from sqlalchemy import Column, Integer, String, Date, DateTime
from superagi.models.base_model import DBBaseModel
from superagi.controllers.types.agent_schedule import AgentScheduleInput

class AgentSchedule(DBBaseModel):
    """
    Represents an Agent Scheduler record in the database.

    Attributes:
        id (Integer): The primary key of the agent scheduler record.
        agent_id (Integer): The ID of the agent being scheduled.
        start_time (DateTime): The date and time from which the agent is scheduled.
        recurrence_interval (String): Stores "none" if not recurring,
            or a time interval like '2 Weeks', '1 Month', '2 Minutes' based on input.
        expiry_date (DateTime): The date and time when the agent is scheduled to stop runs.
        expiry_runs (Integer): The number of runs before the agent expires.
        current_runs (Integer): Number of runs executed in that schedule.
        status: state in which the schedule is, "SCHEDULED" or "STOPPED" or "COMPLETED" or "TERMINATED"

    Methods:
        __repr__: Returns a string representation of the AgentSchedule instance.

    """
    __tablename__ = 'agent_schedule'
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer)
    start_time = Column(DateTime)
    next_scheduled_time = Column(DateTime)
    recurrence_interval = Column(String)
    expiry_date = Column(DateTime)
    expiry_runs = Column(Integer)
    current_runs = Column(Integer)
    status = Column(String)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string representation of the AgentSchedule instance.\n        '
        return f'AgentSchedule(id={self.id}, agent_id={self.agent_id}, start_time={self.start_time}, next_scheduled_time={self.next_scheduled_time}, recurrence_interval={self.recurrence_interval}, expiry_date={self.expiry_date}, expiry_runs={self.expiry_runs}), current_runs={self.expiry_runs}), status={self.status}), '

    @classmethod
    def save_schedule_from_config(cls, session, db_agent, schedule_config: AgentScheduleInput):
        if False:
            return 10
        agent_schedule = AgentSchedule(agent_id=db_agent.id, start_time=schedule_config.start_time, next_scheduled_time=schedule_config.start_time, recurrence_interval=schedule_config.recurrence_interval, expiry_date=schedule_config.expiry_date, expiry_runs=schedule_config.expiry_runs, current_runs=0, status='SCHEDULED')
        agent_schedule.agent_id = db_agent.id
        session.add(agent_schedule)
        session.commit()
        return agent_schedule

    @classmethod
    def find_by_agent_id(cls, session, agent_id: int):
        if False:
            i = 10
            return i + 15
        db_schedule = session.query(AgentSchedule).filter(AgentSchedule.agent_id == agent_id).first()
        return db_schedule