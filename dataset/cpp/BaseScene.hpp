#pragma once

#include "Graphics/RendererTypes.hpp"
#include "Scene/GameObject.hpp"

IGNORE_WARNINGS_PUSH
#include <glm/gtx/norm.hpp> // for distance2
IGNORE_WARNINGS_POP

namespace flex
{
	class PhysicsWorld;
	class ReflectionProbe;
	class Player;
	class PointLight;
	class DirectionalLight;
	struct JSONObject;
	struct JSONField;
	struct Material;
	class ICallbackGameObject;

	class BaseScene final
	{
	public:
		// fileName e.g. "scene_01.json"
		explicit BaseScene(const std::string& fileName);
		~BaseScene() = default;

		BaseScene(const BaseScene&) = delete;
		BaseScene& operator=(const BaseScene&) = delete;

		void Initialize();
		void PostInitialize();
		void Destroy();
		void FixedUpdate();
		void Update();
		void LateUpdate();
		void Render();

		bool IsInitialized() const;

		void OnPrefabChanged(const PrefabID& prefabID);

		bool LoadFromFile(const std::string& filePath);
		void CreateBlank(const std::string& filePath);

		void DrawImGuiObjects();
		void DoSceneContextMenu();

		void BindOnGameObjectDestroyedCallback(ICallbackGameObject* callback);
		void UnbindOnGameObjectDestroyedCallback(ICallbackGameObject* callback);

		void SetName(const std::string& name);
		std::string GetName() const;
		std::string GetDefaultRelativeFilePath() const;
		std::string GetRelativeFilePath() const;
		std::string GetShortRelativeFilePath() const;
		// Creates a new file at the specified location and copies this scene's data in to it
		// Always copies saved file if exists, old files can optionally be deleted
		bool SetFileName(const std::string& fileName, bool bDeletePreviousFiles);
		std::string GetFileName() const;

		bool IsUsingSaveFile() const;

		PhysicsWorld* GetPhysicsWorld();

		/*
		* Serializes all data from scene into JSON scene file.
		* Only writes data that has non-default values (e.g. an identity
		* transform is not saved)
		*/
		void SerializeToFile(bool bSaveOverDefault = false) const;

		void DeleteSaveFiles();

		std::vector<GameObject*>& GetRootObjects();
		void GetInteractableObjects(std::vector<GameObject*>& interactableObjects);
		void GetNearbyInteractableObjects(std::list<Pair<GameObject*, real>>& sortedInteractableObjects,
			const glm::vec3& posWS,
			real sqDistThreshold,
			GameObjectID excludeGameObjectID);

		GameObject* AddRootObject(GameObject* gameObject);
		GameObject* AddRootObjectImmediate(GameObject* gameObject);
		GameObject* AddChildObject(GameObject* parent, GameObject* child);
		GameObject* AddChildObjectImmediate(GameObject* parent, GameObject* child);
		GameObject* AddSiblingObjectImmediate(GameObject* gameObject, GameObject* newSibling);

		void SetRootObjectIndex(GameObject* rootObject, i32 newIndex);

		// Editor objects are objects not normally shown in the scene hierarchy, or searched by standard functions
		EditorObject* AddEditorObjectImmediate(EditorObject* editorObject);
		void RemoveEditorObjectImmediate(EditorObject* editorObject);

		void RemoveAllObjects(); // Removes and destroys all objects in scene at end of frame
		void RemoveAllObjectsImmediate();  // Removes and destroys all objects in scene
		void RemoveAllEditorObjectsImmediate();
		void RemoveObject(const GameObjectID& gameObjectID, bool bDestroy);
		void RemoveObject(GameObject* gameObject, bool bDestroy);
		void RemoveObjectImmediate(const GameObjectID& gameObjectID, bool bDestroy);
		void RemoveObjectImmediate(GameObject* gameObject, bool bDestroy);
		void RemoveObjects(const std::vector<GameObjectID>& gameObjects, bool bDestroy);
		void RemoveObjects(const std::vector<GameObject*>& gameObjects, bool bDestroy);
		void RemoveObjectsImmediate(const std::vector<GameObjectID>& gameObjects, bool bDestroy);
		void RemoveObjectsImmediate(const std::vector<GameObject*>& gameObjects, bool bDestroy);

		GameObject* InstantiatePrefab(const PrefabID& prefabID, GameObject* parent = nullptr);
		GameObject* ReinstantiateFromPrefab(const PrefabID& prefabID, GameObject* previousInstance);

		u32 NumObjectsLoadedFromPrefabID(const PrefabID& prefabID) const;
		void DeleteInstancesOfPrefab(const PrefabID& prefabID);

		GameObjectID FirstObjectWithTag(const std::string& tag);

		Player* GetPlayer(i32 index);

		bool IsLoaded() const;

		template<class T>
		T* GetFirstObjectOfType(StringID typeID)
		{
			for (GameObject* rootObject : m_RootObjects)
			{
				GameObject* result = rootObject->FilterFirst([&typeID](GameObject* gameObject)
				{
					return (gameObject->GetTypeID() == typeID);
				});
				if (result != nullptr)
				{
					return (T*)result;
				}
			}

			return nullptr;
		}

		template<class T>
		std::vector<T*> GetObjectsOfType(StringID typeID)
		{
			std::vector<T*> result;

			for (GameObject* rootObject : m_RootObjects)
			{
				rootObject->FilterType([&typeID](GameObject* gameObject)
				{
					return (gameObject->GetTypeID() == typeID);
				}, result);
			}

			return result;
		}

		// Returns a unique name beginning with existingName that no other children of parent have
		// If root object, specify nullptr as parent
		std::string GetUniqueObjectName(const std::string& existingName, GameObject* parent = nullptr);
		// Returns 'prefix' with a number appended representing
		// how many other objects with that prefix are in the scene
		std::string GetUniqueObjectName(const std::string& prefix, i16 digits, GameObject* parent = nullptr);

		i32 GetSceneFileVersion() const;

		bool HasPlayers() const;

		const SkyboxData& GetSkyboxData() const;

		void DrawImGuiForSelectedObjectsAndSceneHierarchy();

		// If the object gets deleted this frame *gameObjectRef gets set to nullptr
		void DoCreateGameObjectButton(const char* buttonName, const char* popupName);
		bool DrawImGuiGameObjectNameAndChildren(GameObject* gameObject, bool bDrawingEditorObjects);
		// Returns true if the parent-child tree changed during this call
		bool DrawImGuiGameObjectNameAndChildrenInternal(GameObject* gameObject, bool bDrawingEditorObjects);

		bool DoNewGameObjectTypeList();
		bool DoGameObjectTypeList(const char* currentlySelectedTypeCStr, StringID& selectedTypeStringID, std::string& selectedTypeStr);

		GameObject* GetGameObject(const GameObjectID& gameObjectID) const;
		GameObject* GetEditorObject(const EditorObjectID& editorObjectID) const;

		bool DrawImGuiGameObjectIDField(const char* label, GameObjectID& ID, bool bReadOnly = false);

		void SetTimeOfDay(real time);
		real GetTimeOfDay() const;
		void SetSecondsPerDay(real secPerDay);
		real GetSecondsPerDay() const;
		void SetTimeOfDayPaused(bool bPaused);
		bool GetTimeOfDayPaused() const;

		real GetPlayerMinHeight() const;
		glm::vec3 GetPlayerSpawnPoint() const;

		void RegenerateTerrain();

		void OnExternalMeshChange(const std::string& meshFilePath);

		// Fills out a sorted list of objects with the given typeID & their distance to the given point
		// Returns true when there are nearby objects
		template<class T>
		bool GetObjectsInRadius(const glm::vec3& pos, real radius, StringID typeID, std::vector<Pair<T*, real>>& objects)
		{
			real radiusSq = radius * radius;
			std::vector<T*> allObjects = GetObjectsOfType<T>(typeID);
			for (T* object : allObjects)
			{
				real dist2 = glm::distance2(pos, object->GetTransform()->GetWorldPosition());
				if (dist2 < radiusSq)
				{
					objects.push_back({ object, glm::sqrt(dist2) });
				}
			}

			std::sort(objects.begin(), objects.end(), [](const Pair<MineralDeposit*, real>& a, const Pair<MineralDeposit*, real>& b)
			{
				return a.second > b.second;
			});

			return !objects.empty();
		}

		// Returns true when there are nearby items
		bool GetDroppedItemsInRadius(const glm::vec3& pos, real radius, std::vector<DroppedItem*>& items);

		void CreateDroppedItem(const PrefabID& prefabID, i32 stackSize, const glm::vec3& dropPosWS, const glm::vec3& initialVel);
		void OnDroppedItemDestroyed(DroppedItem* item);

		static const char* GameObjectTypeIDToString(StringID typeID);

		static const i32 LATEST_SCENE_FILE_VERSION = 7;
		static const i32 LATEST_MATERIALS_FILE_VERSION = 2;
		static const i32 LATEST_MESHES_FILE_VERSION = 1;
		static const i32 LATETST_PREFAB_FILE_VERSION = 3;

		static AudioSourceID s_PickupAudioID;

	protected:
		friend GameObject;
		friend EditorObject;
		friend SceneManager;

		void FindNextAvailableUniqueName(GameObject* gameObject, i32& highestNoNameObj, i16& maxNumChars, const char* defaultNewNameBase);
		void DeleteInstancesOfPrefabRecursive(const PrefabID& prefabID, GameObject* gameObject);

		void RemoveObjectImmediateRecursive(const GameObjectID& gameObjectID, bool bDestroy);

		void UpdateRootObjectSiblingIndices();
		void RegisterGameObject(GameObject* gameObject);
		void UnregisterGameObject(const GameObjectID& gameObjectID, bool bAssertSuccess = false);
		void UnregisterGameObjectRecursive(const GameObjectID& gameObjectID, bool bAssertSuccess = false);

		void RegisterEditorObject(EditorObject* editorObject);
		void UnregisterEditorObject(EditorObjectID* editorObjectID);
		void UnregisterEditorObjectRecursive(EditorObjectID* editorObjectID);

		void CreateNewGameObject(const std::string& newObjectName, GameObject* parent = nullptr);

		bool FindConflictingObjectsWithName(GameObject* parent, const std::string& name, const std::vector<GameObject*>& objects);

		i32 m_SceneFileVersion = 1;
		i32 m_MaterialsFileVersion = 1;
		i32 m_MeshesFileVersion = 1;

		std::string m_Name;
		std::string m_FileName;

		PhysicsWorld* m_PhysicsWorld = nullptr;

		std::map<GameObjectID, GameObject*> m_GameObjectLUT;
		std::vector<GameObject*> m_RootObjects;
		std::map<EditorObjectID, EditorObject*> m_EditorObjectLUT;
		std::vector<EditorObject*> m_EditorObjects;

		bool m_bInitialized = false;
		bool m_bLoaded = false;
		bool m_bSpawnPlayer = false;
		GameObjectID m_PlayerGUIDs[2];

		ReflectionProbe* m_ReflectionProbe = nullptr;

		bool m_bPauseTimeOfDay = false;
		real m_TimeOfDay = 0.0f; // [0, 1) - 0 = noon, 0.5 = midnight
		real m_SecondsPerDay = 6000.0f;

		SkyboxData m_SkyboxDatas[4];
		SkyboxData m_SkyboxData;

		glm::vec3 m_DirLightColours[4];

		const real m_DroppedItemScale = 0.3f;

		// Kill zone for player
		real m_PlayerMinHeight = -500.0f;
		glm::vec3 m_PlayerSpawnPoint;

		Player* m_Player0 = nullptr;
		Player* m_Player1 = nullptr;

		std::vector<ICallbackGameObject*> m_OnGameObjectDestroyedCallbacks;

		std::vector<GameObject*> m_PendingAddObjects; // Objects to add as root objects at LateUpdate
		std::vector<Pair<GameObject*, GameObject*>> m_PendingAddChildObjects; // Objects to add to parents at LateUpdate
		std::vector<GameObjectID> m_PendingRemoveObjects; // Objects to remove but not destroy at LateUpdate this frame
		std::vector<GameObjectID> m_PendingDestroyObjects; // Objects to destroy at LateUpdate this frame

		std::vector<DroppedItem*> m_DroppedItems;

	private:
		/*
		* Recursively searches through all game objects and returns first
		* one containing given tag, or nullptr if none exist
		*/
		GameObjectID FindObjectWithTag(const std::string& tag, GameObject* gameObject);

		void OnPrefabChangedInternal(const PrefabID& prefabID, GameObject* prefabTemplate, GameObject* rootObject);

		Pair<StringID, std::string> m_NewObjectTypeIDPair;

	};
} // namespace flex
