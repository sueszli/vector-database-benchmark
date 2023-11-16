#include "ObjLoader.h"
#include <fstream>

using namespace std;

int ObjLoader::GetVertexCount()
{
	return vertexCount;
}

int ObjLoader::GetTextureCount()
{
	return textureCount;
}

int ObjLoader::GetNormalCount()
{
	return normalCount;
}

int ObjLoader::GetFaceCount()
{
	return faceCount;
}

void ObjLoader::Reset()
{
	vertexCount = 0;
	textureCount = 0;
	normalCount = 0;
	faceCount = 0;

	vertices.clear();
	texcoords.clear();
	normals.clear();
}

bool ObjLoader::ReadFileCounts(const char* fileName)
{
	ifstream fin;
	char input;

	int tempVertexCount = 0;
	int tempTextureCount = 0;
	int tempNormalCount = 0;
	int tempFaceCount = 0;

	fin.open(fileName);

	if (fin.fail())
	{
		return false;
	}

	fin.get(input);
	while (!fin.eof())
	{
		switch (input)
		{
		case 'v':
			fin.get(input);
			if (input == ' ')		// 버텍스 좌표
				tempVertexCount++;
			else if (input == 't')	// 텍스처 좌표
				tempTextureCount++;
			else if (input == 'n')	// 법선 좌표
				tempNormalCount++;
			break;
		case 'f':
			tempFaceCount++;
			break;
		}

		while (input != '\n')
			fin.get(input);

		fin.get(input);
	}

	vertexCount = tempVertexCount;
	textureCount = tempTextureCount;
	normalCount = tempNormalCount;
	faceCount = tempFaceCount;

	fin.close();

	return true;
}

bool ObjLoader::LoadObjVertexData(const char* fileName, VertexType * vertexType, WORD* indices)
{
	ifstream fin;
	XMFLOAT3 vertex;
	XMFLOAT2 texcoord;
	XMFLOAT3 normal;
	FaceType face;

	char input, input2;

	fin.open(fileName);

	if (fin.fail())
		return false;

	vertices.clear();
	texcoords.clear();
	normals.clear();

	fin.get(input);
	while (!fin.eof())
	{
		switch (input)
		{
		case 'v':
			fin.get(input);
			if (input == ' ')		// 버텍스 좌표
			{
				fin >> vertex.x >> vertex.y >> vertex.z;
				vertex.z = vertex.z * -1.0f;		// 오른손 좌표계(obj) -> 왼손 좌표계 변환(DX11)

				vertices.emplace_back(vertex);
			}
			else if (input == 't')	// 텍스처 좌표
			{
				fin >> texcoord.x >> texcoord.y;
				texcoord.y = 1.0f - texcoord.y;		// 오른손 좌표계(obj) -> 왼손 좌표계 변환(DX11)

				texcoords.emplace_back(texcoord);
			}
			else if (input == 'n')	// 법선 좌표
			{
				fin >> normal.x >> normal.y >> normal.z;
				normal.z = normal.z * -1.0f;		// 오른손 좌표계(obj) -> 왼손 좌표계 변환(DX11)

				normals.emplace_back(normal);
			}
			break;
		case 'f':
			fin.get(input);
			if (input == ' ')
			{
				fin >> face.vIndex3 >> input2 >> face.tIndex3 >> input2 >> face.nIndex3;
				fin >> face.vIndex2 >> input2 >> face.tIndex2 >> input2 >> face.nIndex2;
				fin >> face.vIndex1 >> input2 >> face.tIndex1 >> input2 >> face.nIndex1;

				faces.emplace_back(face);
			}
			break;
		}

		while (input != '\n')
			fin.get(input);

		fin.get(input);
	}

	fin.close();

	// 버텍스, 인덱스 정보 저장
	int count = 0;
	for (int i = 0; i < faceCount; i++)
	{
		// .obj에서는 인덱스의 번호가 1부터 시작해서 cpp에 맞도록 0부터 시작할 수 있도록 -1 처리
		vertexType[count].pos = vertices[faces[i].vIndex1 - 1];
		vertexType[count].texture = texcoords[faces[i].tIndex1 - 1];
		vertexType[count].normal = normals[faces[i].nIndex1 - 1];

		indices[count] = count;

		count++;

		vertexType[count].pos = vertices[faces[i].vIndex2 - 1];
		vertexType[count].texture = texcoords[faces[i].tIndex2 - 1];
		vertexType[count].normal = normals[faces[i].nIndex2 - 1];

		indices[count] = count;

		count++;

		vertexType[count].pos = vertices[faces[i].vIndex3 - 1];
		vertexType[count].texture = texcoords[faces[i].tIndex3 - 1];
		vertexType[count].normal = normals[faces[i].nIndex3 - 1];

		indices[count] = count;

		count++;
	}

	return true;
}
