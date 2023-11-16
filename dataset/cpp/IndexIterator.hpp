// Copyright (C) 2023 Jérôme "Lynix" Leclercq (lynix680@gmail.com)
// This file is part of the "Nazara Engine - Utility module"
// For conditions of distribution and use, see copyright notice in Config.hpp

#pragma once

#ifndef NAZARA_UTILITY_INDEXITERATOR_HPP
#define NAZARA_UTILITY_INDEXITERATOR_HPP

#include <NazaraUtils/Prerequisites.hpp>

namespace Nz
{
	class IndexMapper;

	class IndexIterator
	{
		friend IndexMapper;

		public:
			class Reference;

			IndexIterator();
			IndexIterator(const IndexIterator& iterator);
			~IndexIterator() = default;

			Reference operator*() const;

			Reference operator[](UInt32 index) const;

			IndexIterator& operator=(const IndexIterator& iterator);

			IndexIterator operator+(UInt32 indexCount) const;
			IndexIterator operator-(UInt32 indexCount) const;

			IndexIterator& operator+=(UInt32 indexCount);
			IndexIterator& operator-=(UInt32 indexCount);

			IndexIterator& operator++();
			IndexIterator operator++(int);

			IndexIterator& operator--();
			IndexIterator operator--(int);

			friend bool operator==(const IndexIterator& lhs, const IndexIterator& rhs);
			friend bool operator!=(const IndexIterator& lhs, const IndexIterator& rhs);
			friend bool operator<(const IndexIterator& lhs, const IndexIterator& rhs);
			friend bool operator<=(const IndexIterator& lhs, const IndexIterator& rhs);
			friend bool operator>(const IndexIterator& lhs, const IndexIterator& rhs);
			friend bool operator>=(const IndexIterator& lhs, const IndexIterator& rhs);

		private:
			IndexIterator(IndexMapper* mapper, UInt32 index);

			IndexMapper* m_mapper;
			UInt32 m_index;
	};

	class IndexIterator::Reference
	{
		friend IndexIterator;

		public:
			Reference(const Reference& reference) = default;
			~Reference() = default;

			Reference& operator=(UInt32 value);
			Reference& operator=(const Reference& reference);

			operator UInt32() const;

		private:
			Reference(IndexMapper* mapper, UInt32 index);

			IndexMapper* m_mapper;
			UInt32 m_index;
	};
}

#include <Nazara/Utility/IndexIterator.inl>

#endif // NAZARA_UTILITY_INDEXITERATOR_HPP
