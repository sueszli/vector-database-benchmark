
#include <linux/netlink.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <iostream>
#include <map>

#include <async/recurring-event.hpp>
#include <helix/ipc.hpp>
#include <protocols/fs/common.hpp>
#include "nl-socket.hpp"
#include "process.hpp"

namespace nl_socket {

struct OpenFile;
struct Group;

bool logSockets = false;

std::map<std::pair<int, int>, std::unique_ptr<Group>> globalGroupMap;

int nextPort = -1;
std::map<int, OpenFile *> globalPortMap;

struct Packet {
	// Sender netlink socket information.
	int senderPort;
	int group;

	// Sender process information.
	int senderPid;

	// The actual octet data that the packet consists of.
	std::vector<char> buffer;
};

struct OpenFile : File {
public:
	static void serve(smarter::shared_ptr<OpenFile> file) {
//TODO:		assert(!file->_passthrough);

		helix::UniqueLane lane;
		std::tie(lane, file->_passthrough) = helix::createStream();
		async::detach(protocols::fs::servePassthrough(std::move(lane),
				file, &File::fileOperations, file->_cancelServe));
	}

	OpenFile(int protocol, bool nonBlock = false)
	: File{StructName::get("nl-socket"), File::defaultPipeLikeSeek}, _protocol{protocol},
			_currentSeq{1}, _inSeq{0}, _socketPort{0}, _passCreds{false}, nonBlock_{nonBlock} { }

	void deliver(Packet packet) {
		_recvQueue.push_back(std::move(packet));
		_inSeq = ++_currentSeq;
		_statusBell.raise();
	}

	void handleClose() override {
		_isClosed = true;
		_statusBell.raise();
		_cancelServe.cancel();
	}

public:
	async::result<frg::expected<Error, size_t>>
	readSome(Process *, void *data, size_t max_length) override {
		if(logSockets)
			std::cout << "posix: Read from socket " << this << std::endl;

		if(_recvQueue.empty() && nonBlock_) {
			if(logSockets)
				std::cout << "posix: netlink socket would block" << std::endl;
			co_return Error::wouldBlock;
		}

		while(_recvQueue.empty())
			co_await _statusBell.async_wait();

		// TODO: Truncate packets (for SOCK_DGRAM) here.
		auto packet = &_recvQueue.front();
		auto size = packet->buffer.size();
		assert(max_length >= size);
		memcpy(data, packet->buffer.data(), size);
		_recvQueue.pop_front();
		co_return size;
	}

	async::result<frg::expected<Error, size_t>>
	writeAll(Process *, const void *data, size_t length) override {
		throw std::runtime_error("posix: Fix netlink send()");
/*
		if(logSockets)
			std::cout << "posix: Write to socket " << this << std::endl;

		Packet packet;
		packet.buffer.resize(length);
		memcpy(packet.buffer.data(), data, length);

		_remote->deliver(std::move(packet));
*/

		co_return {};
	}

	async::result<protocols::fs::RecvResult>
	recvMsg(Process *process, uint32_t flags, void *data, size_t max_length,
			void *addr_ptr, size_t max_addr_length, size_t max_ctrl_length) override {
		using namespace protocols::fs;
		if(logSockets)
			std::cout << "posix: Recv from socket \e[1;34m" << structName() << "\e[0m" << std::endl;
		if(!(flags & ~(MSG_DONTWAIT | MSG_CMSG_CLOEXEC))) {
			std::cout << "posix: Unsupported flag in recvMsg" << std::endl;
		}

		if(_recvQueue.empty() && ((flags & MSG_DONTWAIT) || nonBlock_)) {
			if(logSockets)
				std::cout << "posix: netlink socket would block" << std::endl;
			co_return RecvResult { protocols::fs::Error::wouldBlock };
		}

		while(_recvQueue.empty())
			co_await _statusBell.async_wait();

		// TODO: Truncate packets (for SOCK_DGRAM) here.
		auto packet = &_recvQueue.front();

		auto size = packet->buffer.size();
		assert(max_length >= size);
		memcpy(data, packet->buffer.data(), size);

		struct sockaddr_nl sa;
		memset(&sa, 0, sizeof(struct sockaddr_nl));
		sa.nl_family = AF_NETLINK;
		sa.nl_pid = packet->senderPort;
		sa.nl_groups = packet->group ? (1 << (packet->group - 1)) : 0;
		memcpy(addr_ptr, &sa, sizeof(struct sockaddr_nl));

		protocols::fs::CtrlBuilder ctrl{max_ctrl_length};

		if(_passCreds) {
			struct ucred creds;
			memset(&creds, 0, sizeof(struct ucred));
			creds.pid = packet->senderPid;

			if(!ctrl.message(SOL_SOCKET, SCM_CREDENTIALS, sizeof(struct ucred)))
				throw std::runtime_error("posix: Implement CMSG truncation");
			ctrl.write<struct ucred>(creds);
		}

		_recvQueue.pop_front();
		co_return RecvData{ctrl.buffer(), size, sizeof(struct sockaddr_nl), 0};
	}

	async::result<frg::expected<protocols::fs::Error, size_t>>
	sendMsg(Process *process, uint32_t flags,
			const void *data, size_t max_length,
			const void *addr_ptr, size_t addr_length,
			std::vector<smarter::shared_ptr<File, FileHandle>> files) override;

	async::result<void> setOption(int option, int value) override {
		assert(option == SO_PASSCRED);
		_passCreds = value;
		co_return;
	};

	async::result<frg::expected<Error, PollWaitResult>>
	pollWait(Process *, uint64_t past_seq, int mask,
			async::cancellation_token cancellation) override {
		(void)mask; // TODO: utilize mask.
		if(_isClosed)
			co_return Error::fileClosed;

		assert(past_seq <= _currentSeq);
		while(past_seq == _currentSeq && !cancellation.is_cancellation_requested())
			co_await _statusBell.async_wait(cancellation);

		if(_isClosed)
			co_return Error::fileClosed;

		// For now making sockets always writable is sufficient.
		int edges = EPOLLOUT;
		if(_inSeq > past_seq)
			edges |= EPOLLIN;

//		std::cout << "posix: pollWait(" << past_seq << ") on \e[1;34m" << structName() << "\e[0m"
//				<< " returns (" << _currentSeq
//				<< ", " << edges << ")" << std::endl;

		co_return PollWaitResult(_currentSeq, edges);
	}

	async::result<frg::expected<Error, PollStatusResult>>
	pollStatus(Process *) override {
		int events = EPOLLOUT;
		if(!_recvQueue.empty())
			events |= EPOLLIN;

		co_return PollStatusResult(_currentSeq, events);
	}

	async::result<protocols::fs::Error>
	bind(Process *, const void *, size_t) override;

	async::result<size_t> sockname(void *, size_t) override;

	helix::BorrowedDescriptor getPassthroughLane() override {
		return _passthrough;
	}

	async::result<void> setFileFlags(int flags) override {
		if (flags & ~O_NONBLOCK) {
			std::cout << "posix: setFileFlags on netlink socket \e[1;34m" << structName() << "\e[0m called with unknown flags" << std::endl;
			co_return;
		}
		if (flags & O_NONBLOCK)
			nonBlock_ = true;
		else
			nonBlock_ = false;
		co_return;
	}

	async::result<int> getFileFlags() override {
		if(nonBlock_)
			co_return O_NONBLOCK;
		co_return 0;
	}

private:
	void _associatePort() {
		assert(!_socketPort);
		_socketPort = nextPort--;
		auto res = globalPortMap.insert({_socketPort, this});
		assert(res.second);
	}

	int _protocol;
	helix::UniqueLane _passthrough;
	async::cancellation_event _cancelServe;

	// Status management for poll().
	async::recurring_event _statusBell;
	bool _isClosed = false;
	uint64_t _currentSeq;
	uint64_t _inSeq;

	int _socketPort;

	// The actual receive queue of the socket.
	std::deque<Packet> _recvQueue;

	// Socket options.
	bool _passCreds;
	bool nonBlock_;
};

struct Group {
	friend struct OpenFile;

	// Sends a copy of the given message to this group.
	void carbonCopy(const Packet &packet);

private:
	std::vector<OpenFile *> _subscriptions;
};

// ----------------------------------------------------------------------------
// OpenFile implementation.
// ----------------------------------------------------------------------------


async::result<frg::expected<protocols::fs::Error, size_t>>
OpenFile::sendMsg(Process *process, uint32_t flags, const void *data, size_t max_length,
		const void *addr_ptr, size_t addr_length,
		std::vector<smarter::shared_ptr<File, FileHandle>> files) {
	if(logSockets)
		std::cout << "posix: Send to socket \e[1;34m" << structName() << "\e[0m" << std::endl;
	assert(!flags);
	assert(addr_length >= sizeof(struct sockaddr_nl));
	assert(files.empty());

	struct sockaddr_nl sa;
	memcpy(&sa, addr_ptr, sizeof(struct sockaddr_nl));

	int grp_idx = 0;
	if(sa.nl_groups) {
		// Linux allows multicast only to a single group at a time.
		grp_idx = __builtin_ffs(sa.nl_groups);
	}

	// TODO: Associate port otherwise.
	assert(_socketPort);

	Packet packet;
	packet.senderPid = process->pid();
	packet.senderPort = _socketPort;
	packet.group = grp_idx;
	packet.buffer.resize(max_length);
	memcpy(packet.buffer.data(), data, max_length);

	// Carbon-copy to the message to a group.
	if(grp_idx) {
		auto it = globalGroupMap.find({_protocol, grp_idx});
		assert(it != globalGroupMap.end());
		auto group = it->second.get();
		group->carbonCopy(packet);
	}

	// Netlink delivers the message per unicast.
	// This is done even if the target address includes multicast groups.
	if(sa.nl_pid) {
		// Deliver to a user-mode socket.
		auto it = globalPortMap.find(sa.nl_pid);
		assert(it != globalPortMap.end());

		it->second->deliver(std::move(packet));
	}else{
		// TODO: Deliver the message a listener function.
	}

	co_return max_length;
}

async::result<protocols::fs::Error> OpenFile::bind(Process *,
		const void *addr_ptr, size_t addr_length) {
	struct sockaddr_nl sa;
	assert(addr_length >= sizeof(struct sockaddr_nl));
	memcpy(&sa, addr_ptr, addr_length);

	assert(!sa.nl_pid);
	_associatePort();

	if(sa.nl_groups) {
		for(int i = 0; i < 32; i++) {
			if(!(sa.nl_groups & (1 << i)))
				continue;
			std::cout << "posix: Join netlink group "
					<< _protocol << "." << (i + 1) << std::endl;

			auto it = globalGroupMap.find({_protocol, i + 1});
			assert(it != globalGroupMap.end());
			auto group = it->second.get();
			group->_subscriptions.push_back(this);
		}
	}

	// Do nothing for now.
	co_return protocols::fs::Error::none;
}

async::result<size_t> OpenFile::sockname(void *addr_ptr, size_t max_addr_length) {
	assert(_socketPort);

	// TODO: Fill in nl_groups.
	struct sockaddr_nl sa;
	memset(&sa, 0, sizeof(struct sockaddr_nl));
	sa.nl_family = AF_NETLINK;
	sa.nl_pid = _socketPort;
	memcpy(addr_ptr, &sa, std::min(sizeof(struct sockaddr_nl), max_addr_length));

	co_return sizeof(struct sockaddr_nl);
}

// ----------------------------------------------------------------------------
// Group implementation.
// ----------------------------------------------------------------------------

void Group::carbonCopy(const Packet &packet) {
	for(auto socket : _subscriptions)
		socket->deliver(packet);
}

// ----------------------------------------------------------------------------
// Free functions.
// ----------------------------------------------------------------------------

void configure(int protocol, int num_groups) {
	for(int i = 0; i < num_groups; i++) {
		std::pair<int, int> idx{protocol, i + 1};
		auto res = globalGroupMap.insert(std::make_pair(idx, std::make_unique<Group>()));
		assert(res.second);
	}
}

void broadcast(int proto_idx, int grp_idx, std::string buffer) {
	Packet packet;
	packet.senderPid = 0;
	packet.senderPort = 0;
	packet.group = grp_idx;
	packet.buffer.resize(buffer.size());
	memcpy(packet.buffer.data(), buffer.data(), buffer.size());

	auto it = globalGroupMap.find({proto_idx, grp_idx});
	assert(it != globalGroupMap.end());
	auto group = it->second.get();
	group->carbonCopy(packet);
}

smarter::shared_ptr<File, FileHandle> createSocketFile(int protocol, bool nonBlock) {
	auto file = smarter::make_shared<OpenFile>(protocol, nonBlock);
	file->setupWeakFile(file);
	OpenFile::serve(file);
	return File::constructHandle(std::move(file));
}

} // namespace nl_socket

