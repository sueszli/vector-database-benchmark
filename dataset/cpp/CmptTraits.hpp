#pragma once

#include "AccessTypeID.hpp"

#include <memory_resource>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <string>

namespace Ubpa::UECS {
	class EntityMngr;

	// run-time dynamic component traits
	// size (> 0) is neccessary
	// optional
	// - alignment: alignof(std::max_align_t) as default, 8 / 16 in most cases
	// - default constructor: do nothing as default
	// - copy constructor: memcpy as default
	// - move constructor: memcpy as default
	// - move assignment: memcpy as default
	// - destructor: do nothing as default
	// - name
	class CmptTraits {
	public:
		static constexpr std::size_t default_alignment = alignof(std::max_align_t);

		CmptTraits& Clear();

		CmptTraits& RegisterName(Type);
		CmptTraits& RegisterTrivial(TypeID);
		CmptTraits& RegisterSize(TypeID, std::size_t size);
		CmptTraits& RegisterAlignment(TypeID, std::size_t alignment);
		CmptTraits& RegisterDefaultConstructor(TypeID, std::function<void(void*, std::pmr::memory_resource*)>);
		CmptTraits& RegisterCopyConstructor(TypeID, std::function<void(void*, const void*, std::pmr::memory_resource*)>);
		CmptTraits& RegisterMoveConstructor(TypeID, std::function<void(void*,void*, std::pmr::memory_resource*)>);
		CmptTraits& RegisterMoveAssignment(TypeID, std::function<void(void*,void*)>);
		CmptTraits& RegisterDestructor(TypeID, std::function<void(void*)>);

		const std::pmr::unordered_set<TypeID>& GetTrivials() const noexcept;
		const std::pmr::unordered_map<TypeID, std::string_view>& GetNames() const noexcept;
		const std::pmr::unordered_map<TypeID, std::size_t>& GetSizeofs() const noexcept;
		const std::pmr::unordered_map<TypeID, std::size_t>& GetAlignments() const noexcept;
		const std::pmr::unordered_map<TypeID, std::function<void(void*, std::pmr::memory_resource*)>>& GetDefaultConstructors() const noexcept;
		const std::pmr::unordered_map<TypeID, std::function<void(void*, const void*, std::pmr::memory_resource*)>>& GetCopyConstructors() const noexcept;
		const std::pmr::unordered_map<TypeID, std::function<void(void*, void*, std::pmr::memory_resource*)>>& GetMoveConstructors() const noexcept;
		const std::pmr::unordered_map<TypeID, std::function<void(void*, void*)>>& GetMoveAssignments() const noexcept;
		const std::pmr::unordered_map<TypeID, std::function<void(void*)>>& GetDestructors() const noexcept;

		bool IsTrivial(TypeID) const;
		std::size_t Sizeof(TypeID) const;
		std::size_t Alignof(TypeID) const;
		std::string_view Nameof(TypeID) const;

		CmptTraits& Deregister(TypeID) noexcept;

		template<typename... Cmpts>
		void Register();

		template<typename... Cmpts>
		void UnsafeRegister();

		template<typename Cmpt>
		void Deregister();

	private:
		friend class EntityMngr;
		CmptTraits(std::pmr::unsynchronized_pool_resource* rsrc);
		CmptTraits(const CmptTraits& other, std::pmr::unsynchronized_pool_resource* rsrc);
		CmptTraits(CmptTraits&& other) noexcept;
		~CmptTraits();

		// register all for Cmpt
		// static_assert
		// - is_default_constructible_v
		// - is_copy_constructible_v || std::is_constructible_v<Cmpt, Cmpt&>
		// - is_move_constructible_v
		// - is_move_assignable_v
		// - is_destructible_v
		template<typename Cmpt>
		void RegisterOne();

		template<typename Cmpt>
		void UnsafeRegisterOne();

		struct Impl;
		std::unique_ptr<Impl> impl;
	};
}

#include "details/CmptTraits.inl"
