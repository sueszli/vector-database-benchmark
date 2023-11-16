﻿
#include "util/tc_docker.h"
#include "util/tc_http.h"
#include "util/tc_http_async.h"
#include "util/tc_base64.h"
#include "util/tc_json.h"

namespace tars
{

void TC_Docker::setDockerUnixLocal(const string& unixSocket)
{
	_dockerUrl = TC_Common::replace(unixSocket, FILE_SEP, "$") + ":0";
}

bool TC_Docker::createHttpRequest(TC_HttpRequest& request)
{
	request.setHeader("X-Registry-Auth", this->_authenticationStr);
	if (!request.hasHeader("Content-Type"))
	{
		request.setHeader("Content-Type", "application/tar");
	}

	request.setHost("localhost");

	TC_HttpResponse response;
	int ret = request.doRequest(response, this->_requestTimeout);
	if(ret != 0)
	{
		_errMessage = "http request docker error, ret:" + TC_Common::tostr(ret);
		return false;
	}

	_responseMessage = TC_Common::trimright(response.getContent());

	if ( response.getStatus() >= 200 && response.getStatus() < 300)
	{
		return true;
	}
	else
	{
		if (!response.getContent().empty())
		{
			_errMessage = TC_Common::trimright(response.getContent());
		}
		else
		{
			_errMessage = "{ \"message\": \"" + response.getAbout() + "\"}";
		}

		return false;
	}
}

void TC_Docker::setAuthentication(const string& useName, const string& password, const string& serverAddress)
{
	_authentication.useName = useName;
	_authentication.password = password;
	_authentication.serverAddress = serverAddress;

	JsonValueObjPtr obj = new JsonValueObj();
	obj->value["username"] = new JsonValueString(useName);
	obj->value["password"] = new JsonValueString(password);
	obj->value["serveraddress"] = new JsonValueString(serverAddress);

	_authenticationStr = TC_Base64::encode(TC_Json::writeValue(obj));
}

bool TC_Docker::login(const string& useName, const string& password, const string& serverAddress)
{
	JsonValueObjPtr obj = new JsonValueObj();
	obj->value["username"] = new JsonValueString(useName);
	obj->value["password"] = new JsonValueString(password);
	obj->value["serveraddress"] = new JsonValueString(serverAddress);

	TC_HttpRequest request;

	request.setPostRequest(_dockerUrl + "/" + _dockerApiVersion + "/auth", TC_Json::writeValue(obj));

	bool flag = createHttpRequest(request);

	if (flag)
	{
		JsonValueObjPtr result = JsonValueObjPtr::dynamicCast(TC_Json::getValue(getResponseMessage()));

		auto it = result->value.find("Status");
		if (it != result->value.end())
		{
			JsonValueStringPtr v = JsonValueStringPtr::dynamicCast(it->second);

			if (v && v->value == "Login Succeeded")
			{
				return true;
			}
		}
		else
		{
			this->_errMessage = JsonValueStringPtr::dynamicCast(result->value["message"])->value;
		}
	}

	return false;
}

bool TC_Docker::pull(const string& image)
{
	TC_HttpRequest request;

	request.setPostRequest(_dockerUrl + "/" + _dockerApiVersion + "/images/create?fromImage=" + image,
			_authenticationStr);

	return createHttpRequest(request);
}

bool TC_Docker::create(const string& name,
		const string& image,
		const vector<string>& entrypoint,
		const vector<string>& commands,
		const vector<string>& envs,
		const map<string, string>& mounts,
		const map<string, pair<string, int>>& ports,
		const string& restartPolicy,
		int maximumRetryCount,
		const string& networkMode,
		const string& ipcMode,
		bool autoRemove,
		bool privileged)
{
	TC_HttpRequest request;

	JsonValueObjPtr obj = new JsonValueObj();

	obj->value["Image"] = new JsonValueString(image);

	if (!entrypoint.empty())
	{
		JsonValueArrayPtr ePtr = new JsonValueArray();
		obj->value["Entrypoint"] = ePtr;

		for (auto& e: entrypoint)
		{
			ePtr->value.push_back(new JsonValueString(e));
		}
	}

	if (!commands.empty())
	{
		JsonValueArrayPtr cmd = new JsonValueArray();
		obj->value["Cmd"] = cmd;

		for (auto& c: commands)
		{
			cmd->value.push_back(new JsonValueString(c));
		}
	}

	if (!envs.empty())
	{
		JsonValueArrayPtr env = new JsonValueArray();
		obj->value["Env"] = env;

		for (auto& e: envs)
		{
			env->value.push_back(new JsonValueString(e));
		}
	}

	JsonValueObjPtr hostConfig = new JsonValueObj();
	obj->value["HostConfig"] = hostConfig;

	hostConfig->value["NetworkMode"] = new JsonValueString(networkMode);

	if (!ports.empty())
	{
		JsonValueObjPtr espec = new JsonValueObj();
		obj->value["ExposedPorts"] = espec;

		JsonValueObjPtr portBindings = new JsonValueObj();
		hostConfig->value["PortBindings"] = portBindings;

		for (auto e: ports)
		{
			espec->value[e.first] = new JsonValueObj();

			JsonValueArrayPtr hostPort = new JsonValueArray();

			JsonValueObjPtr hObj = new JsonValueObj();
			hObj->value["HostIp"] = new JsonValueString(e.second.first);
			hObj->value["HostPort"] = new JsonValueString(TC_Common::tostr(e.second.second));

			hostPort->push_back(hObj);

			portBindings->value[e.first] = hostPort;
		}
	}

	if (!mounts.empty())
	{
		JsonValueArrayPtr Mounts = new JsonValueArray();
		hostConfig->value["Mounts"] = Mounts;

		for (auto e: mounts)
		{
			JsonValueObjPtr mObj = new JsonValueObj();
			mObj->value["Source"] = new JsonValueString(e.first);
			mObj->value["Target"] = new JsonValueString(e.second);
			mObj->value["Type"] = new JsonValueString("bind");

			Mounts->push_back(mObj);
		}
	}

	if (!restartPolicy.empty())
	{
		JsonValueObjPtr rObj = new JsonValueObj();
		rObj->value["RestartPolicy"] = new JsonValueString(restartPolicy);
		rObj->value["maximumRetryCount"] = new JsonValueNum((int64_t)maximumRetryCount, true);

		hostConfig->value["RestartPolicy"] = rObj;
	}

//	JsonValueArrayPtr shell = new JsonValueArray();
//	shell->value.push_back(new JsonValueString("bash"));
//	shell->value.push_back(new JsonValueString("bash"));
//	shell->value.push_back(new JsonValueString("bash"));
//	hostConfig->value["Shell"] = shell;

	hostConfig->value["WorkingDir"] = new JsonValueString("/");
	hostConfig->value["AutoRemove"] = new JsonValueBoolean(autoRemove);
	hostConfig->value["Privileged"] = new JsonValueBoolean(privileged);

	string json = TC_Json::writeValue(obj);

	request.setPostRequest(_dockerUrl + "/" + _dockerApiVersion + "/containers/create?name=" + name, json);

	request.setHeader("Content-Type", "application/json");

	return createHttpRequest(request);
}

bool TC_Docker::exec(const string& name, const vector<string>& commands, const vector<string>& envs, bool privileged)
{

	TC_HttpRequest request;

	JsonValueObjPtr obj = new JsonValueObj();

	if (!commands.empty())
	{
		JsonValueArrayPtr cmd = new JsonValueArray();
		obj->value["Cmd"] = cmd;

		for (auto& c: commands)
		{
			cmd->value.push_back(new JsonValueString(c));
		}
	}

	if (!envs.empty())
	{
		JsonValueArrayPtr env = new JsonValueArray();
		obj->value["Env"] = env;

		for (auto& e: envs)
		{
			env->value.push_back(new JsonValueString(e));
		}
	}

	obj->value["WorkingDir"] = new JsonValueString("/");
	obj->value["Privileged"] = new JsonValueBoolean(privileged);

	string json = TC_Json::writeValue(obj);

	request.setPostRequest(_dockerUrl + "/" + _dockerApiVersion + "/containers/" + name + "/exec", json);

	request.setHeader("Content-Type", "application/json");

	return createHttpRequest(request);
}

bool TC_Docker::start(const string& containerId)
{
	TC_HttpRequest request;

	request.setPostRequest(_dockerUrl + "/" + _dockerApiVersion + "/containers/" + containerId + "/start", "");

	return createHttpRequest(request);
}

bool TC_Docker::stop(const string& containerId, size_t waitSeconds)
{
	TC_HttpRequest request;

	request.setPostRequest(_dockerUrl + "/" + _dockerApiVersion + "/containers/" + containerId + "/stop?t=" +
						   TC_Common::tostr(waitSeconds), "");

	return createHttpRequest(request);
}

bool TC_Docker::remove(const string& containerId, bool force)
{
	TC_HttpRequest request;

	request.setDeleteRequest(_dockerUrl + "/" + _dockerApiVersion + "/containers/" + containerId + "?force=" +
							 string(force ? "true" : "false"), "");

	return createHttpRequest(request);
}

bool TC_Docker::inspectContainer(const string& containerId)
{
	TC_HttpRequest request;

	request.setGetRequest(_dockerUrl + "/" + _dockerApiVersion + "/containers/" + containerId + "/json");

	return createHttpRequest(request);
}

bool TC_Docker::inspectImage(const string& imageId)
{
	TC_HttpRequest request;

	request.setGetRequest(_dockerUrl + "/" + _dockerApiVersion + "/images/" + imageId + "/json");

	return createHttpRequest(request);
}
}



