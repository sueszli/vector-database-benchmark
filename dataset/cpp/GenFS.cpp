#include "GenFS.h"

#include "ModelKDTree.h"
#include "Model.h"
#include "Unit.h"

#include <Basic/File.h>
#include <Basic/Math.h>
#include <Basic/Error.h>

#include <3rdParty/csv/CSV.hpp>

#include <sstream>
#include <map>
#include <functional>

using namespace App;

using namespace jay::util;
using namespace rapidjson;
using namespace std;

const string GenFS::Call(
	const int ID,
	const std::string dir,
	const std::string templateFSPath,
	const std::vector<Connection> & connections,
	const std::vector<Activation> & activations)
{
	static auto const & ERROR = ErrorRetVal(Call);

	constexpr int inputDim = 17;
	constexpr int outputDim = 3;

	auto modelKDTree = LoadModelKDTree(ID, dir, connections, activations);
	if (IsError(modelKDTree))
		return ERROR;

	File templateFS(templateFSPath, File::READ);
	if (!templateFS.IsValid()) {
		printf("ERROR: template FS open error\n");
		return ERROR;
	}

	auto minValAndExtent = LoadMatrix(dir + to_string(ID) + "_minVal_and_extent.csv");
	if (minValAndExtent.size() != 2 || minValAndExtent[0].size() != inputDim + outputDim) {
		printf("ERROR: minValAndExtent.size() != 2 || minValAndExtent[0].size() != %d", inputDim + outputDim);
		return ERROR;
	}

	string fs = templateFS.ReadAll();

	const string funcname = "$funcname$";
	while (fs.find(funcname) != string::npos) {
		auto pos = fs.find(funcname);
		fs.erase(fs.begin() + pos, fs.begin() + pos + funcname.length());
		fs.insert(pos, "Model");
	}

	stringstream shader;

	shader << fs;
	
	shader << modelKDTree->GenFunc(minValAndExtent[0], minValAndExtent[1]) << endl;

	return shader.str();
}

const vector<vector<float>> GenFS::LoadCSV(const string & path) {
	static auto const & ERROR = ErrorRetVal(LoadCSV);
	
	CSVread reader(path);
	if (reader.error) {
		printf("ERROR: %s\n", reader.error_msg.c_str());
		return ERROR;
	}

	vector<vector<float>> rst;

	while (reader.ReadRecord()) {
		rst.push_back(vector<float>());
		for (auto const & valStr : reader.fields) {
			float val = Math::CastTo<float>(valStr);
			rst.back().push_back(val);
		}
	}

	return rst;
}

const GenFS::Matf GenFS::LoadMatrix(const string & path) {
	static auto const & ERROR = ErrorRetVal(LoadMatrix);

	auto rst = LoadCSV(path);
	if (IsError(rst))
		return ERROR;

	for (auto const & row : rst) {
		if (row.size() != rst.front().size()) {
			printf("ERROR: not matrix [%s]\n", path.c_str());
			return ERROR;
		}
	}

	return rst;
}

const Ptr<Layer> GenFS::LoadLayer(const string & path, const Connection & connection, const Activation & activation) {
	static auto const & ERROR = ErrorRetVal(LoadLayer);

	auto rst = LoadMatrix(path);
	if (IsError(rst))
		return ERROR;

	if (rst.size() == 0)
		return ERROR;

	size_t w = rst[0].size();
	size_t h = rst.size();

	// transpose
	vector<vector<float>> rstT(w);
	for (int row = 0; row < w; row++) {
		rstT[row] = vector<float>(h);
		for (int col = 0; col < h; col++)
			rstT[row][col] = rst[col][row];
	}

	auto layer = Layer::New(nullptr, static_cast<int>(h) - 1, connection, activation);

	for (auto const & weights : rstT)
		auto unit = Unit::New(layer, weights);

	return layer;
}

const Ptr<Model> GenFS::LoadModel(
	const int ID,
	const int secID,
	const std::string & dir,
	const vector<Connection> & connections,
	const vector<Activation> & activations)
{
	static auto const & ERROR = ErrorRetVal(LoadModel);

	if (connections.size() != 3 || activations.size() != 3) {
		printf("ERROR: connections.size() != 3 || activations.size() != 3\n");
		return ERROR;
	}

	constexpr int inputDim = 17;
	constexpr int outputDim = 3;

	const string wholeID = to_string(ID) + "_" + to_string(secID);
	const string modelName = "FNN_" + wholeID;

	const string dense0Path = dir + wholeID + "_dense0.csv";
	const string dense1Path = dir + wholeID + "_dense1.csv";
	const string dense2Path = dir + wholeID + "_dense2.csv";

	auto model = Model::New(modelName, inputDim, outputDim);

	auto dense0 = LoadLayer(dense0Path, connections[0], activations[0]);
	if (IsError(dense0))
		return ERROR;
	dense0->SetModel(model);

	auto dense1 = LoadLayer(dense1Path, connections[1], activations[1]);
	if (IsError(dense1))
		return ERROR;
	dense1->SetModel(model);

	auto dense2 = LoadLayer(dense2Path, connections[2], activations[2]);
	if (IsError(dense2))
		return ERROR;
	dense2->SetModel(model);

	return model;
}

const Ptr<ModelKDTree> GenFS::LoadModelKDTree(
	const int id,
	const string & dir,
	const vector<Connection> & connections,
	const vector<Activation> & activations)
{
	static auto const & ERROR = ErrorRetVal(LoadModelKDTree);

	string jsonFilePath = dir + to_string(id) + "_KDTree.json";
	File jsonFile(jsonFilePath.c_str(), File::READ);
	if (!jsonFile.IsValid()) {
		printf("ERROR: !jsonFile.IsValid()\n");
		return ERROR;
	}
	string jsonContent = jsonFile.ReadAll();

	Document doc;
	doc.Parse(jsonContent.c_str());
	if (doc.GetParseError() != ParseErrorCode::kParseErrorNone) {
		printf("ERROR: parse json[%s] error\n", jsonFilePath.c_str());
		return ERROR;
	}
	
	return LoadModelKDTreeFromJson(id, dir, doc.MemberBegin(), doc.MemberEnd(), connections, activations);
}

const Ptr<ModelKDTree> GenFS::LoadModelKDTreeFromJson(
	const int id,
	const string & dir,
	Value::MemberIterator begin,
	Value::MemberIterator end,
	const vector<Connection> & connections,
	const vector<Activation> & activations)
{
	constexpr int inputDim = 17;
	constexpr int outputDim = 3;

	Ptr<ModelKDTree> leftChild = nullptr;
	Ptr<ModelKDTree> rightChild = nullptr;

	for (auto it = begin; it != end; ++it) {
		string name = it->name.GetString();
		if (name == KEY::_names()[KEY::left] && !it->value.IsNull()) {
			auto jsonTree = it->value.GetObject();
			leftChild = LoadModelKDTreeFromJson(id, dir, jsonTree.MemberBegin(), jsonTree.MemberEnd(), connections, activations);
		}
		else if (name == KEY::_names()[KEY::right] && !it->value.IsNull()) {
			auto jsonTree = it->value.GetObject();
			rightChild = LoadModelKDTreeFromJson(id, dir, jsonTree.MemberBegin(), jsonTree.MemberEnd(), connections, activations);
		}
		else
			continue;
	}

	bool needModel = leftChild == nullptr || rightChild == nullptr;

	int axis = -1;
	float spiltVal = 0;
	int nodeID = -1;
	Ptr<Model> model = nullptr;
	for (auto it = begin; it != end; ++it) {
		string name = it->name.GetString();
		if (name == KEY::_names()[KEY::spiltAxis])
			axis = it->value.GetInt();
		else if (name == KEY::_names()[KEY::spiltVal])
			spiltVal = it->value.GetFloat();
		else if (name == KEY::_names()[KEY::nodeID]) {
			nodeID = it->value.GetInt();
			if (needModel) {
				string modelName = to_string(id) + "_" + to_string(nodeID);
				model = LoadModel(id, nodeID, dir, connections, activations);
			}
		}else
			continue;
	}

	auto modelKDTree = ModelKDTree::New(nodeID, nullptr, inputDim, outputDim, axis, spiltVal, model);

	modelKDTree->SetLeft(leftChild);
	modelKDTree->SetRight(rightChild);

	return modelKDTree;
}
