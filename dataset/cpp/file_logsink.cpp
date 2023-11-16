// Copyright 2015-2023 the openage authors. See copying.md for legal info.

#include "file_logsink.h"

#include <iomanip>
#include <string>

#include "log/level.h"
#include "log/logsource.h"
#include "log/message.h"
#include "util/enum.h"

namespace openage::log {


FileSink::FileSink(const char *filename, bool append) :
	outfile{filename, std::ios_base::out | (append ? std::ios_base::app : std::ios_base::trunc)} {}


void FileSink::output_log_message(const message &msg, LogSource *source) {
	this->outfile << msg.lvl->name << "|";
	this->outfile << source->logsource_name() << "|";
	this->outfile << msg.filename << ":" << msg.lineno << "|";
	this->outfile << msg.functionname << "|";
	this->outfile << msg.thread_id << "|";
	this->outfile << std::setprecision(7) << std::fixed << msg.timestamp / 1e9 << "|";
	this->outfile << msg.text << std::endl;
}

} // namespace openage::log
