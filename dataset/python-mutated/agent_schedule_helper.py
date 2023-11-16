from superagi.models.db import connect_db
from sqlalchemy.orm import sessionmaker
from superagi.models.agent_config import AgentConfiguration
from superagi.models.agent_schedule import AgentSchedule
from datetime import datetime, timedelta
from superagi.helper.time_helper import parse_interval_to_seconds
import pytz
engine = connect_db()
Session = sessionmaker(bind=engine)

class AgentScheduleHelper:
    AGENT_SCHEDULE_TIME_INTERVAL = 300

    def run_scheduled_agents(self):
        if False:
            while True:
                i = 10
        '\n        Execute all eligible scheduled agent tasks since last five minutes.\n        '
        now = datetime.now()
        last_five_minutes = now - timedelta(minutes=5)
        session = Session()
        scheduled_agents = session.query(AgentSchedule).filter(AgentSchedule.next_scheduled_time.between(last_five_minutes, now), AgentSchedule.status == 'SCHEDULED').all()
        for agent in scheduled_agents:
            interval = agent.recurrence_interval
            interval_in_seconds = 0
            if interval is not None:
                interval_in_seconds = parse_interval_to_seconds(interval)
            agent_id = agent.agent_id
            agent_execution_name = self.__create_execution_name_for_scheduling(agent_id)
            should_execute_agent = self.__should_execute_agent(agent, interval)
            self.__execute_schedule(should_execute_agent, interval_in_seconds, session, agent, agent_execution_name)
        for agent in scheduled_agents:
            if self.__can_remove_agent(agent, interval):
                agent.status = 'COMPLETED'
                session.commit()
        session.close()

    def update_next_scheduled_time(self):
        if False:
            i = 10
            return i + 15
        '\n        Update the next scheduled time of each agent and terminate those who have finished their schedule, in case of any miss.\n        '
        now = datetime.now()
        session = Session()
        scheduled_agents = session.query(AgentSchedule).filter(AgentSchedule.start_time <= now, AgentSchedule.next_scheduled_time <= now, AgentSchedule.status == 'SCHEDULED').all()
        for agent in scheduled_agents:
            if (now - agent.next_scheduled_time).total_seconds() < AgentScheduleHelper.AGENT_SCHEDULE_TIME_INTERVAL:
                continue
            if agent.recurrence_interval is not None:
                interval_in_seconds = parse_interval_to_seconds(agent.recurrence_interval)
                time_diff = now - agent.start_time
                num_intervals_passed = time_diff.total_seconds() // interval_in_seconds
                updated_next_scheduled_time = agent.start_time + timedelta(seconds=interval_in_seconds * (num_intervals_passed + 1))
                agent.next_scheduled_time = updated_next_scheduled_time
            else:
                agent.status = 'TERMINATED'
            session.commit()
        session.close()

    def __create_execution_name_for_scheduling(self, agent_id) -> str:
        if False:
            while True:
                i = 10
        '\n        Create name for an agent execution based on current time.\n\n        Args:\n            agent_id (str): The id of the agent job to be scheduled.\n\n        Returns:\n            str: Execution name of the agent in the format "Run <timestamp>"\n        '
        session = Session()
        user_timezone = session.query(AgentConfiguration).filter(AgentConfiguration.key == 'user_timezone', AgentConfiguration.agent_id == agent_id).first()
        if user_timezone and user_timezone.value != 'None':
            current_time = datetime.now().astimezone(pytz.timezone(user_timezone.value))
        else:
            current_time = datetime.now().astimezone(pytz.timezone('GMT'))
        timestamp = current_time.strftime(' %d %B %Y %H:%M')
        return f'Run{timestamp}'

    def __should_execute_agent(self, agent, interval):
        if False:
            return 10
        '\n        Determine if an agent should be executed based on its scheduling.\n\n        Args:\n            agent (object): The agent job to evaluate.\n            interval (int): Recurrence interval of the scheduled agent in seconds.\n\n        Returns:\n            bool: True if the agent should be executed, False otherwise.\n        '
        expiry_date = agent.expiry_date
        expiry_runs = agent.expiry_runs
        current_runs = agent.current_runs
        if not interval or (expiry_date is None and expiry_runs == -1):
            return True
        if expiry_date and datetime.now() < expiry_date:
            return True
        if expiry_runs != -1 and current_runs < expiry_runs:
            return True
        return False

    def __can_remove_agent(self, agent, interval):
        if False:
            return 10
        '\n        Determine if an agent can be removed based on its scheduled expiry.\n\n        Args:\n            agent (object): The agent job to evaluate.\n            interval (int): Recurrence interval of the scheduled agent in seconds.\n\n        Returns:\n            bool: True if the agent can be removed, False otherwise.\n        '
        expiry_date = agent.expiry_date
        expiry_runs = agent.expiry_runs
        current_runs = agent.current_runs
        next_scheduled = agent.next_scheduled_time + timedelta(seconds=parse_interval_to_seconds(interval)) if interval else None
        if not interval:
            return True
        if expiry_date and datetime.now() < expiry_date and (next_scheduled is None or next_scheduled <= expiry_date):
            return False
        if expiry_runs != -1 and current_runs < expiry_runs:
            return False
        if expiry_date is None and expiry_runs == -1:
            return False
        return True

    def __execute_schedule(self, should_execute_agent, interval_in_seconds, session, agent, agent_execution_name):
        if False:
            return 10
        '\n        Executes a scheduled job, if it should be executed.\n        Args:\n            should_execute_agent (bool): Whether agent should be executed.\n            interval_in_seconds (int): The interval in seconds for the schedule.\n            session (Session): The database session.\n            agent (object): The agent to be scheduled.\n            agent_execution_name (str): The name for the execution.\n        '
        from superagi.jobs.scheduling_executor import ScheduledAgentExecutor
        if should_execute_agent:
            executor = ScheduledAgentExecutor()
            executor.execute_scheduled_agent(agent.agent_id, agent_execution_name)
            agent.current_runs = agent.current_runs + 1
            if agent.recurrence_interval:
                next_scheduled_time = agent.next_scheduled_time + timedelta(seconds=interval_in_seconds)
                agent.next_scheduled_time = next_scheduled_time
            session.commit()