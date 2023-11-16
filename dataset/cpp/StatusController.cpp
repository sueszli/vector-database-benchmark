#include "StatusController.h"

#include "ApplicationSettings.h"
#include "StatusView.h"
#include "utility.h"

StatusController::StatusController()
{
	m_statusFilter = ApplicationSettings::getInstance()->getStatusFilter();
}

StatusController::~StatusController() {}

StatusView* StatusController::getView() const
{
	return Controller::getView<StatusView>();
}

void StatusController::clear() {}

void StatusController::handleMessage(MessageClearStatusView* message)
{
	m_status.clear();
	getView()->clear();
}

void StatusController::handleMessage(MessageShowStatus* message)
{
	getView()->showDockWidget();
}

void StatusController::handleMessage(MessageStatus* message)
{
	if (message->status().empty())
	{
		return;
	}

	std::vector<Status> stati;

	for (const std::wstring& status: message->stati())
	{
		stati.push_back(Status(status, message->isError));
	}

	utility::append(m_status, stati);

	addStatus(stati);
}

void StatusController::handleMessage(MessageStatusFilterChanged* message)
{
	m_statusFilter = message->statusFilter;

	getView()->clear();
	addStatus(m_status);

	ApplicationSettings* settings = ApplicationSettings::getInstance().get();
	settings->setStatusFilter(m_statusFilter);
	settings->save();
}

void StatusController::addStatus(const std::vector<Status> status)
{
	std::vector<Status> filteredStatus;

	for (const Status& s: status)
	{
		if (s.type & m_statusFilter)
		{
			filteredStatus.push_back(s);
		}
	}

	getView()->addStatus(filteredStatus);
}
