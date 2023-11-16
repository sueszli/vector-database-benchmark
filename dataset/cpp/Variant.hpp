#pragma once

namespace flex
{
	namespace IR
	{
		struct Value;
	}

	class VirtualMachine;

	static const char* VariantTypeNames[] =
	{
		"int",
		"uint",
		"long",
		"ulong",
		"float",
		"bool",
		"string",
		"char",
		"void",

		"NONE"
	};

	struct Variant
	{
		enum class Type
		{
			INT,
			UINT,
			LONG,
			ULONG,
			FLOAT,
			BOOL,
			STRING,
			CHAR,
			VOID_, // Avoid Windows' define of VOID

			_NONE
		};

		static_assert((u32)Type::_NONE == ((u32)ARRAY_LENGTH(VariantTypeNames) - 1), "Variant::VariantTypeNames doesn't contain a matching number of elements to Variant::Type enum");

		static const char* TypeToString(Type type);

		Variant() :
			type(Type::_NONE),
			valInt(0)
		{}

		explicit Variant(i32 val) :
			type(Type::INT),
			valInt(val)
		{}

		explicit Variant(u32 val) :
			type(Type::UINT),
			valUInt(val)
		{}

		explicit Variant(i64 val) :
			type(Type::LONG),
			valLong(val)
		{}

		explicit Variant(u64 val) :
			type(Type::ULONG),
			valULong(val)
		{}

		explicit Variant(real val) :
			type(Type::FLOAT),
			valFloat(val)
		{}

		explicit Variant(bool val) :
			type(Type::BOOL),
			valBool(val ? 1 : 0)
		{}

		explicit Variant(const char* val) :
			type(Type::STRING),
			valStr(val)
		{}

		explicit Variant(char val) :
			type(Type::CHAR),
			valChar(val)
		{}

		Variant(const Variant& other);
		Variant(const Variant&& other);

		explicit Variant(const IR::Value& other);

		bool IsValid() const;

		std::string ToString() const;

		static bool IsIntegral(Type type);

		i32 AsInt() const;
		u32 AsUInt() const;
		i64 AsLong() const;
		u64 AsULong() const;
		real AsFloat() const;
		i32 AsBool() const;
		const char* AsString() const;
		char AsChar() const;

		bool IsZero() const;
		bool IsPositive() const;

		// Updates this variant's value (does not allow changing value type)
		// Returns true when successful
		bool SetValueFromString(const char* str);

		Variant& operator=(const Variant& other);
		Variant& operator=(const Variant&& other);
		bool operator<(const Variant& other);
		bool operator<=(const Variant& other);
		bool operator>(const Variant& other);
		bool operator>=(const Variant& other);
		bool operator==(const Variant& other);
		bool operator!=(const Variant& other);

		Type type = Type::_NONE;
		union
		{
			i32 valInt;
			u32 valUInt;
			i64 valLong;
			u64 valULong;
			real valFloat;
			i32 valBool;
			const char* valStr;
			char valChar;

			u64 _largestField;
		};

		static Type CheckAssignmentType(Type lhsType, Type rhsType);

	private:
		void CheckAssignmentType(Type otherType);

	};

	Variant operator+(const Variant& lhs, const Variant& rhs);
	Variant operator-(const Variant& lhs, const Variant& rhs);
	Variant operator*(const Variant& lhs, const Variant& rhs);
	Variant operator/(const Variant& lhs, const Variant& rhs);
	Variant operator%(const Variant& lhs, const Variant& rhs);
	Variant operator^(const Variant& lhs, const Variant& rhs);
	Variant operator&(const Variant& lhs, const Variant& rhs);
	Variant operator&&(const Variant& lhs, const Variant& rhs);
	Variant operator|(const Variant& lhs, const Variant& rhs);
	Variant operator||(const Variant& lhs, const Variant& rhs);

	extern Variant g_EmptyVariant;
} // namespace flex
