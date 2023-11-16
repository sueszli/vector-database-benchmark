
#include <linux/netlink.h>
#include <sstream>

#include "drvcore.hpp"
#include "nl-socket.hpp"

namespace drvcore {

std::shared_ptr<sysfs::Object> globalDevicesObject;
std::shared_ptr<sysfs::Object> globalBusObject;
std::shared_ptr<sysfs::Object> globalClassObject;
std::shared_ptr<sysfs::Object> globalCharObject;
std::shared_ptr<sysfs::Object> globalBlockDevObject;
std::shared_ptr<sysfs::Object> globalBlockObject;

sysfs::Object *devicesObject() {
	assert(globalDevicesObject);
	return globalDevicesObject.get();
}

sysfs::Object *busObject() {
	assert(globalBusObject);
	return globalBusObject.get();
}

sysfs::Object *classObject() {
	assert(globalClassObject);
	return globalClassObject.get();
}

struct UeventAttribute : sysfs::Attribute {
	static auto singleton() {
		static UeventAttribute attr;
		return &attr;
	}

private:
	UeventAttribute()
	: sysfs::Attribute("uevent", true) { }

public:
	virtual async::result<frg::expected<Error, std::string>> show(sysfs::Object *object) override {
		auto device = static_cast<Device *>(object);

		UeventProperties ue;
		device->composeStandardUevent(ue);
		device->composeUevent(ue);

		std::stringstream ss;
		for(const auto &[name, value] : ue)
			ss << name << '=' << value << '\n';

		co_return ss.str();
	}

	virtual async::result<Error> store(sysfs::Object *object, std::string data) override {
		auto device = static_cast<Device *>(object);

		UeventProperties ue;
		device->composeStandardUevent(ue);
		device->composeUevent(ue);

		std::string sysfs_path = device->getSysfsPath();
		std::stringstream ss;
		ss << "add@/" << sysfs_path << '\0';
		ss << "ACTION=add" << '\0';
		ss << "DEVPATH=/" << sysfs_path << '\0';
		ss << "SEQNUM=" << drvcore::makeHotplugSeqnum() << '\0';
		for(const auto &[name, value] : ue)
			ss << name << '=' << value << '\0';
		drvcore::emitHotplug(ss.str());
		co_return Error::success;
	}
};

//-----------------------------------------------------------------------------
// Device implementation.
//-----------------------------------------------------------------------------

Device::Device(std::shared_ptr<Device> parent, std::string name, UnixDevice *unix_device)
: sysfs::Object{parent ? parent : globalDevicesObject, std::move(name)},
		_unixDevice{unix_device}, _parentDevice{std::move(parent)} { }

std::string Device::getSysfsPath() {
	std::string path = name();
	auto parent = directoryNode()->treeLink()->getOwner();
	auto link = parent->treeLink().get();
	while(true) {
		auto owner = link->getOwner();
		if(!owner)
			break;
		path = link->getName() + '/' + path;
		link = owner->treeLink().get();
	}

	return path;
}

void Device::composeStandardUevent(UeventProperties &ue) {
	if(auto unix_dev = unixDevice(); unix_dev) {
		auto node_path = unix_dev->nodePath();
		if(!node_path.empty())
			ue.set("DEVNAME", node_path);
		ue.set("MAJOR", std::to_string(unix_dev->getId().first));
		ue.set("MINOR", std::to_string(unix_dev->getId().second));
	}
}

void Device::linkToSubsystem() {
	// Nothing to do for devices outside of a subsystem.
}

//-----------------------------------------------------------------------------
// BusSubsystem and BusDevice implementation.
//-----------------------------------------------------------------------------

BusSubsystem::BusSubsystem(std::string name)
: _object{std::make_shared<sysfs::Object>(globalBusObject, std::move(name))} {
	_object->addObject();
	_devicesObject = std::make_shared<sysfs::Object>(_object, "devices");
	_devicesObject->addObject();
	_driversObject = std::make_shared<sysfs::Object>(_object, "drivers");
	_driversObject->addObject();
}

BusDevice::BusDevice(BusSubsystem *subsystem, std::string name,
		UnixDevice *unix_device, std::shared_ptr<Device> parent)
: Device{parent, std::move(name), unix_device},
		_subsystem{subsystem} { }

void BusDevice::linkToSubsystem() {
	auto devices_object = _subsystem->devicesObject();
	devices_object->createSymlink(name(), devicePtr());
	createSymlink("subsystem", _subsystem->object());
}

//-----------------------------------------------------------------------------
// ClassSubsystem and ClassDevice implementation.
//-----------------------------------------------------------------------------

ClassSubsystem::ClassSubsystem(std::string name)
: _object{std::make_shared<sysfs::Object>(globalClassObject, std::move(name))} {
	_object->addObject();
}

ClassDevice::ClassDevice(ClassSubsystem *subsystem, std::shared_ptr<Device> parent,
		std::string name, UnixDevice *unix_device)
: Device{std::move(parent), std::move(name), unix_device},
		_subsystem{subsystem} { }

void ClassDevice::linkToSubsystem() {
	auto subsystem_object = _subsystem->object();
	subsystem_object->createSymlink(name(), devicePtr());
	if(auto parent = parentDevice(); parent)
		createSymlink("device", std::move(parent));
	createSymlink("subsystem", subsystem_object);
}

//-----------------------------------------------------------------------------
// BlockDevice implementation.
//-----------------------------------------------------------------------------

BlockDevice::BlockDevice(ClassSubsystem *subsystem, std::shared_ptr<Device> parent,
		std::string name, UnixDevice *unix_device)
: Device{std::move(parent), std::move(name), unix_device},
	_subsystem{subsystem} { }

void BlockDevice::linkToSubsystem() {
	auto subsystem_object = _subsystem->object();
	globalBlockObject->createSymlink(name(), devicePtr());
	subsystem_object->createSymlink(name(), devicePtr());
	if (auto parent = parentDevice(); parent)
		createSymlink("device", std::move(parent));
	createSymlink("subsystem", subsystem_object);
}

//-----------------------------------------------------------------------------
// Free functions.
//-----------------------------------------------------------------------------

void initialize() {
	nl_socket::configure(NETLINK_KOBJECT_UEVENT, 32);
	nl_socket::configure(NETLINK_ROUTE, 32);

	// Create the /sys/dev/{char,block} directories.
	auto dev_object = std::make_shared<sysfs::Object>(nullptr, "dev");
	globalCharObject = std::make_shared<sysfs::Object>(dev_object, "char");
	globalBlockDevObject = std::make_shared<sysfs::Object>(dev_object, "block");

	// Create the global /sys/{devices,class,dev} directories.
	globalDevicesObject = std::make_shared<sysfs::Object>(nullptr, "devices");
	globalBusObject = std::make_shared<sysfs::Object>(nullptr, "bus");
	globalClassObject = std::make_shared<sysfs::Object>(nullptr, "class");
	globalBlockObject = std::make_shared<sysfs::Object>(nullptr, "block");
	globalDevicesObject->addObject();
	globalBusObject->addObject();
	globalClassObject->addObject();
	globalBlockObject->addObject();
	dev_object->addObject();
	globalCharObject->addObject(); // TODO: Do this before dev_object is visible.
	globalBlockDevObject->addObject();
}

void installDevice(std::shared_ptr<Device> device) {
	device->setupDevicePtr(device);
	device->addObject();

	// TODO: Do this before the object becomes visible in sysfs.
	device->linkToSubsystem();
	device->realizeAttribute(UeventAttribute::singleton());

	if(auto unix_dev = device->unixDevice(); unix_dev) {
		std::stringstream id_ss;
		id_ss << unix_dev->getId().first << ":" << unix_dev->getId().second;
		if (unix_dev->type() == VfsType::charDevice)
			globalCharObject->createSymlink(id_ss.str(), device);
		else if (unix_dev->type() == VfsType::blockDevice)
			globalBlockDevObject->createSymlink(id_ss.str(), device);
		else
			assert(!"Unsupported unix device trying to be added!");
	}

	UeventProperties ue;
	device->composeStandardUevent(ue);
	device->composeUevent(ue);

	std::string sysfs_path = device->getSysfsPath();
	std::stringstream ss;
	ss << "add@/" << sysfs_path << '\0';
	ss << "ACTION=add" << '\0';
	ss << "DEVPATH=/" << sysfs_path << '\0';
	ss << "SEQNUM=" << drvcore::makeHotplugSeqnum() << '\0';
	for(const auto &[name, value] : ue)
		ss << name << '=' << value << '\0';
	drvcore::emitHotplug(ss.str());
}

// TODO: There could be a race between makeHotplugSeqnum() and emitHotplug().
//       Is it required that seqnums always appear in the correct order?
uint32_t makeHotplugSeqnum() {
	static uint32_t seqnum = 1;
	return seqnum++;
}

void emitHotplug(std::string buffer) {
	nl_socket::broadcast(NETLINK_KOBJECT_UEVENT, 1, std::move(buffer));
}

} // namespace drvcore

