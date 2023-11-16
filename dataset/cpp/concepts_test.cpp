// The MIT License (MIT)
//
// Copyright (c) 2018 Mateusz Pusz
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <mp-units/chrono.h>
#include <mp-units/concepts.h>
#include <mp-units/quantity_point.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/natural/natural.h>
#include <mp-units/systems/si/si.h>
#include <chrono>
#include <complex>
#include <optional>
#include <string>

template<typename T>
inline constexpr bool mp_units::is_scalar<std::complex<T>> = true;

namespace {

using namespace mp_units;

inline constexpr struct my_origin : absolute_point_origin<isq::length> {
} my_origin;
inline constexpr struct my_relative_origin : relative_point_origin<my_origin + isq::length(42 * si::metre)> {
} my_relative_origin;

struct dim_speed : decltype(isq::dim_length / isq::dim_time) {};

// BaseDimension
static_assert(detail::BaseDimension<struct isq::dim_length>);
static_assert(!detail::BaseDimension<std::remove_const_t<decltype(isq::dim_length / isq::dim_time)>>);
static_assert(!detail::BaseDimension<std::remove_const_t<decltype(inverse(isq::dim_time))>>);
static_assert(!detail::BaseDimension<std::remove_const_t<decltype(pow<2>(isq::dim_length))>>);
static_assert(!detail::BaseDimension<derived_dimension<struct isq::dim_length, per<struct isq::dim_time>>>);
static_assert(!detail::BaseDimension<dim_speed>);
static_assert(!detail::BaseDimension<base_dimension<"L">>);
static_assert(!detail::BaseDimension<struct si::metre>);
static_assert(!detail::BaseDimension<int>);

// DerivedDimension
static_assert(detail::DerivedDimension<std::remove_const_t<decltype(isq::dim_length / isq::dim_time)>>);
static_assert(detail::DerivedDimension<std::remove_const_t<decltype(inverse(isq::dim_time))>>);
static_assert(detail::DerivedDimension<std::remove_const_t<decltype(pow<2>(isq::dim_length))>>);
static_assert(detail::DerivedDimension<derived_dimension<struct isq::dim_length, per<struct isq::dim_time>>>);
static_assert(detail::DerivedDimension<struct dimension_one>);
static_assert(!detail::DerivedDimension<dim_speed>);
static_assert(!detail::DerivedDimension<struct isq::dim_length>);
static_assert(!detail::DerivedDimension<struct si::metre>);
static_assert(!detail::DerivedDimension<int>);

// Dimension
static_assert(Dimension<struct isq::dim_length>);
static_assert(Dimension<std::remove_const_t<decltype(isq::dim_length / isq::dim_time)>>);
static_assert(Dimension<std::remove_const_t<decltype(inverse(isq::dim_time))>>);
static_assert(Dimension<std::remove_const_t<decltype(pow<2>(isq::dim_length))>>);
static_assert(Dimension<derived_dimension<struct isq::dim_length, per<struct isq::dim_time>>>);
static_assert(Dimension<struct dimension_one>);
static_assert(!Dimension<dim_speed>);
static_assert(!Dimension<base_dimension<"L">>);
static_assert(!Dimension<struct si::metre>);
static_assert(!Dimension<int>);

// QuantitySpec
struct speed : decltype(isq::length / isq::time) {};  // this is not recommended

static_assert(QuantitySpec<struct isq::length>);
static_assert(QuantitySpec<struct isq::radius>);
static_assert(QuantitySpec<struct isq::speed>);
static_assert(QuantitySpec<std::remove_const_t<decltype(kind_of<isq::length>)>>);
static_assert(QuantitySpec<std::remove_const_t<decltype(isq::length / isq::time)>>);
static_assert(QuantitySpec<std::remove_const_t<decltype(pow<2>(isq::length))>>);
static_assert(QuantitySpec<struct dimensionless>);
static_assert(!QuantitySpec<speed>);
static_assert(!QuantitySpec<struct isq::dim_length>);
static_assert(!QuantitySpec<int>);

// NamedQuantitySpec
static_assert(detail::NamedQuantitySpec<struct isq::length>);
static_assert(detail::NamedQuantitySpec<struct isq::radius>);
static_assert(detail::NamedQuantitySpec<struct isq::speed>);
static_assert(!detail::NamedQuantitySpec<std::remove_const_t<decltype(kind_of<isq::length>)>>);
static_assert(!detail::NamedQuantitySpec<std::remove_const_t<decltype(isq::length / isq::time)>>);
static_assert(!detail::NamedQuantitySpec<std::remove_const_t<decltype(pow<2>(isq::length))>>);
static_assert(detail::NamedQuantitySpec<struct dimensionless>);
static_assert(!detail::NamedQuantitySpec<speed>);
static_assert(!detail::NamedQuantitySpec<struct isq::dim_length>);
static_assert(!detail::NamedQuantitySpec<int>);

// IntermediateDerivedQuantitySpec
static_assert(!detail::IntermediateDerivedQuantitySpec<struct isq::length>);
static_assert(!detail::IntermediateDerivedQuantitySpec<struct isq::radius>);
static_assert(!detail::IntermediateDerivedQuantitySpec<std::remove_const_t<decltype(kind_of<isq::length>)>>);
static_assert(!detail::IntermediateDerivedQuantitySpec<struct isq::speed>);
static_assert(detail::IntermediateDerivedQuantitySpec<std::remove_const_t<decltype(isq::length / isq::time)>>);
static_assert(detail::IntermediateDerivedQuantitySpec<std::remove_const_t<decltype(pow<2>(isq::length))>>);
static_assert(!detail::IntermediateDerivedQuantitySpec<struct dimensionless>);
static_assert(!detail::IntermediateDerivedQuantitySpec<speed>);
static_assert(!detail::IntermediateDerivedQuantitySpec<struct isq::dim_length>);
static_assert(!detail::IntermediateDerivedQuantitySpec<int>);

// QuantityKindSpec
static_assert(!detail::QuantityKindSpec<struct isq::length>);
static_assert(!detail::QuantityKindSpec<struct isq::radius>);
static_assert(detail::QuantityKindSpec<std::remove_const_t<decltype(kind_of<isq::length>)>>);
static_assert(!detail::QuantityKindSpec<struct isq::speed>);
static_assert(!detail::QuantityKindSpec<std::remove_const_t<decltype(isq::length / isq::time)>>);
static_assert(!detail::QuantityKindSpec<std::remove_const_t<decltype(pow<2>(isq::length))>>);
static_assert(!detail::QuantityKindSpec<struct dimensionless>);
static_assert(!detail::QuantityKindSpec<speed>);
static_assert(!detail::QuantityKindSpec<struct isq::dim_length>);
static_assert(!detail::QuantityKindSpec<int>);

// Unit
struct metre_per_second : decltype(si::metre / si::second) {};

static_assert(Unit<struct si::metre>);
static_assert(Unit<struct si::kilogram>);
static_assert(Unit<std::remove_const_t<decltype(si::kilo<si::gram>)>>);
static_assert(Unit<struct natural::electronvolt>);
static_assert(Unit<std::remove_const_t<decltype(si::metre / si::second)>>);
static_assert(Unit<std::remove_const_t<decltype(inverse(si::second))>>);
static_assert(Unit<std::remove_const_t<decltype(mag<10> * si::second)>>);
static_assert(Unit<std::remove_const_t<decltype(square(si::metre))>>);
static_assert(Unit<std::remove_const_t<decltype(pow<2>(si::metre))>>);
static_assert(Unit<struct si::standard_gravity>);
static_assert(Unit<scaled_unit<mag<10>, struct si::second>>);
static_assert(Unit<metre_per_second>);
static_assert(Unit<derived_unit<struct si::metre, per<struct si::second>>>);
static_assert(Unit<struct one>);
static_assert(!Unit<named_unit<"?", isq::length>>);
static_assert(!Unit<named_unit<"?">>);
static_assert(!Unit<named_unit<"?", si::metre / si::second>>);
static_assert(!Unit<named_unit<"?", si::metre, isq::length>>);
static_assert(!Unit<prefixed_unit<"?", mag<10>, si::second>>);
static_assert(!Unit<struct isq::dim_length>);
static_assert(!Unit<int>);
static_assert(!Unit<std::chrono::seconds>);

// NamedUnit
static_assert(detail::NamedUnit<struct si::metre>);
static_assert(detail::NamedUnit<struct natural::electronvolt>);
static_assert(!detail::NamedUnit<struct si::kilogram>);
static_assert(!detail::NamedUnit<std::remove_const_t<decltype(si::kilo<si::gram>)>>);
static_assert(!detail::NamedUnit<std::remove_const_t<decltype(si::metre / si::second)>>);
static_assert(!detail::NamedUnit<std::remove_const_t<decltype(inverse(si::second))>>);
static_assert(!detail::NamedUnit<std::remove_const_t<decltype(mag<10> * si::second)>>);
static_assert(!detail::NamedUnit<std::remove_const_t<decltype(square(si::metre))>>);
static_assert(!detail::NamedUnit<std::remove_const_t<decltype(pow<2>(si::metre))>>);
static_assert(detail::NamedUnit<struct si::standard_gravity>);
static_assert(!detail::NamedUnit<scaled_unit<mag<10>, struct si::second>>);
static_assert(!detail::NamedUnit<metre_per_second>);
static_assert(!detail::NamedUnit<derived_unit<struct si::metre, per<struct si::second>>>);
static_assert(!detail::NamedUnit<struct one>);
static_assert(!detail::NamedUnit<named_unit<"?", isq::length>>);
static_assert(!detail::NamedUnit<named_unit<"?">>);
static_assert(!detail::NamedUnit<named_unit<"?", si::metre / si::second>>);
static_assert(!detail::NamedUnit<named_unit<"?", si::metre, isq::length>>);
static_assert(!detail::NamedUnit<prefixed_unit<"?", mag<10>, si::second>>);
static_assert(!detail::NamedUnit<struct isq::dim_length>);
static_assert(!detail::NamedUnit<int>);
static_assert(!detail::NamedUnit<std::chrono::seconds>);

// PrefixableUnit
static_assert(PrefixableUnit<struct si::metre>);
static_assert(PrefixableUnit<struct natural::electronvolt>);
static_assert(!PrefixableUnit<struct si::kilogram>);
static_assert(!PrefixableUnit<std::remove_const_t<decltype(si::kilo<si::gram>)>>);
static_assert(!PrefixableUnit<std::remove_const_t<decltype(si::metre / si::second)>>);
static_assert(!PrefixableUnit<std::remove_const_t<decltype(inverse(si::second))>>);
static_assert(!PrefixableUnit<std::remove_const_t<decltype(mag<10> * si::second)>>);
static_assert(!PrefixableUnit<std::remove_const_t<decltype(square(si::metre))>>);
static_assert(!PrefixableUnit<std::remove_const_t<decltype(pow<2>(si::metre))>>);
static_assert(PrefixableUnit<struct si::standard_gravity>);
static_assert(!PrefixableUnit<scaled_unit<mag<10>, struct si::second>>);
static_assert(!PrefixableUnit<metre_per_second>);
static_assert(!PrefixableUnit<derived_unit<struct si::metre, per<struct si::second>>>);
static_assert(!PrefixableUnit<struct one>);
static_assert(!PrefixableUnit<named_unit<"?", isq::length>>);
static_assert(!PrefixableUnit<named_unit<"?">>);
static_assert(!PrefixableUnit<named_unit<"?", si::metre / si::second>>);
static_assert(!PrefixableUnit<named_unit<"?", si::metre, isq::length>>);
static_assert(!PrefixableUnit<prefixed_unit<"?", mag<10>, si::second>>);
static_assert(!PrefixableUnit<struct isq::dim_length>);
static_assert(!PrefixableUnit<int>);
static_assert(!PrefixableUnit<std::chrono::seconds>);

// AssociatedUnit
static_assert(AssociatedUnit<struct si::metre>);
static_assert(!AssociatedUnit<struct natural::electronvolt>);
static_assert(AssociatedUnit<struct si::kilogram>);
static_assert(AssociatedUnit<std::remove_const_t<decltype(si::kilo<si::gram>)>>);
static_assert(AssociatedUnit<std::remove_const_t<decltype(si::metre / si::second)>>);
static_assert(AssociatedUnit<std::remove_const_t<decltype(inverse(si::second))>>);
static_assert(AssociatedUnit<std::remove_const_t<decltype(mag<10> * si::second)>>);
static_assert(AssociatedUnit<std::remove_const_t<decltype(square(si::metre))>>);
static_assert(AssociatedUnit<std::remove_const_t<decltype(pow<2>(si::metre))>>);
static_assert(AssociatedUnit<struct si::standard_gravity>);
static_assert(AssociatedUnit<scaled_unit<mag<10>, struct si::second>>);
static_assert(AssociatedUnit<metre_per_second>);
static_assert(AssociatedUnit<derived_unit<struct si::metre, per<struct si::second>>>);
static_assert(AssociatedUnit<struct one>);
static_assert(!AssociatedUnit<named_unit<"?", isq::length>>);
static_assert(!AssociatedUnit<named_unit<"?">>);
static_assert(!AssociatedUnit<named_unit<"?", si::metre / si::second>>);
static_assert(!AssociatedUnit<named_unit<"?", si::metre, isq::length>>);
static_assert(!AssociatedUnit<prefixed_unit<"?", mag<10>, si::second>>);
static_assert(!AssociatedUnit<struct isq::dim_length>);
static_assert(!AssociatedUnit<int>);
static_assert(!AssociatedUnit<std::chrono::seconds>);

// UnitOf
static_assert(UnitOf<struct si::metre, isq::length>);
static_assert(UnitOf<struct si::metre, isq::radius>);
static_assert(UnitOf<struct si::kilogram, isq::mass>);
static_assert(UnitOf<struct si::hertz, isq::frequency>);
static_assert(UnitOf<struct si::hertz, inverse(isq::time)>);
static_assert(UnitOf<struct one, dimensionless>);
static_assert(UnitOf<struct percent, dimensionless>);
static_assert(UnitOf<struct si::radian, isq::angular_measure>);
static_assert(UnitOf<struct one, isq::angular_measure>);
static_assert(UnitOf<struct percent, isq::angular_measure>);
static_assert(!UnitOf<struct si::radian, dimensionless>);
static_assert(!UnitOf<struct si::metre, isq::time>);
static_assert(!UnitOf<struct natural::electronvolt, isq::energy>);

// Reference
static_assert(Reference<struct si::metre>);
static_assert(Reference<std::remove_const_t<decltype(si::metre / si::second)>>);
static_assert(Reference<std::remove_const_t<decltype(isq::length[si::metre])>>);
static_assert(Reference<std::remove_const_t<decltype(isq::radius[si::metre])>>);
static_assert(Reference<std::remove_const_t<decltype(isq::radius[si::metre] / isq::time[si::second])>>);
static_assert(!Reference<struct natural::electronvolt>);
static_assert(!Reference<struct isq::length>);
static_assert(!Reference<std::remove_const_t<decltype(kind_of<isq::length>)>>);
static_assert(!Reference<struct isq::dim_length>);
static_assert(!Reference<int>);

// ReferenceOf
static_assert(ReferenceOf<struct si::metre, isq::length>);
static_assert(ReferenceOf<struct si::metre, isq::radius>);
static_assert(!ReferenceOf<struct si::second, isq::length>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::length[si::metre])>, isq::length>);
static_assert(!ReferenceOf<std::remove_const_t<decltype(isq::length[si::metre])>, isq::radius>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::radius[si::metre])>, isq::length>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::radius[si::metre])>, isq::radius>);
static_assert(ReferenceOf<struct si::metre, isq::dim_length>);
static_assert(!ReferenceOf<struct si::second, isq::dim_length>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::length[si::metre])>, isq::dim_length>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::radius[si::metre])>, isq::dim_length>);

static_assert(ReferenceOf<struct one, dimensionless>);
static_assert(ReferenceOf<struct one, dimension_one>);
static_assert(ReferenceOf<std::remove_const_t<decltype(dimensionless[one])>, dimensionless>);
static_assert(ReferenceOf<std::remove_const_t<decltype(dimensionless[one])>, dimension_one>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::rotation[one])>, isq::rotation>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::rotation[one])>, dimensionless>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::rotation[one])>, dimension_one>);
static_assert(ReferenceOf<struct si::radian, isq::angular_measure>);
static_assert(!ReferenceOf<struct si::radian, dimensionless>);
static_assert(ReferenceOf<struct si::radian, dimension_one>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::angular_measure[si::radian])>, isq::angular_measure>);
static_assert(!ReferenceOf<std::remove_const_t<decltype(isq::angular_measure[si::radian])>, dimensionless>);
static_assert(ReferenceOf<std::remove_const_t<decltype(isq::angular_measure[si::radian])>, dimension_one>);
static_assert(ReferenceOf<struct one, isq::rotation>);
static_assert(ReferenceOf<struct one, isq::angular_measure>);
static_assert(!ReferenceOf<std::remove_const_t<decltype(dimensionless[one])>, isq::rotation>);
static_assert(!ReferenceOf<std::remove_const_t<decltype(dimensionless[one])>, isq::angular_measure>);

// Representation
static_assert(Representation<int>);
static_assert(Representation<double>);
static_assert(Representation<std::complex<double>>);
static_assert(!Representation<bool>);
static_assert(!Representation<std::optional<int>>);
static_assert(!Representation<std::chrono::seconds>);
static_assert(!Representation<std::string>);

// RepresentationOf
static_assert(RepresentationOf<int, quantity_character::scalar>);
static_assert(RepresentationOf<double, quantity_character::scalar>);
static_assert(RepresentationOf<std::complex<double>, quantity_character::scalar>);
static_assert(!RepresentationOf<bool, quantity_character::scalar>);
static_assert(!RepresentationOf<std::optional<int>, quantity_character::scalar>);
static_assert(!RepresentationOf<std::chrono::seconds, quantity_character::scalar>);
static_assert(!RepresentationOf<std::string, quantity_character::scalar>);

// Quantity
static_assert(Quantity<quantity<si::metre>>);
static_assert(Quantity<quantity<isq::length[si::metre]>>);
static_assert(Quantity<quantity<si::metre, int>>);
static_assert(Quantity<quantity<isq::length[si::metre], int>>);
static_assert(!Quantity<std::chrono::seconds>);
static_assert(!Quantity<quantity_point<si::metre, my_origin>>);
static_assert(!Quantity<std::remove_const_t<decltype(isq::length[si::metre])>>);

// QuantityOf
static_assert(QuantityOf<quantity<si::metre>, isq::length>);
static_assert(QuantityOf<quantity<si::metre>, isq::radius>);
static_assert(!QuantityOf<quantity<si::second>, isq::length>);
static_assert(QuantityOf<quantity<isq::length[si::metre]>, isq::length>);
static_assert(!QuantityOf<quantity<isq::length[si::metre]>, isq::radius>);
static_assert(QuantityOf<quantity<isq::radius[si::metre]>, isq::length>);
static_assert(QuantityOf<quantity<isq::radius[si::metre]>, isq::radius>);
static_assert(QuantityOf<quantity<si::metre>, isq::dim_length>);
static_assert(!QuantityOf<quantity<si::second>, isq::dim_length>);
static_assert(QuantityOf<quantity<isq::length[si::metre]>, isq::dim_length>);
static_assert(QuantityOf<quantity<isq::radius[si::metre]>, isq::dim_length>);

static_assert(QuantityOf<quantity<one>, dimensionless>);
static_assert(QuantityOf<quantity<one>, dimension_one>);
static_assert(QuantityOf<quantity<dimensionless[one]>, dimensionless>);
static_assert(QuantityOf<quantity<dimensionless[one]>, dimension_one>);
static_assert(QuantityOf<quantity<isq::rotation[one]>, isq::rotation>);
static_assert(QuantityOf<quantity<isq::rotation[one]>, dimensionless>);
static_assert(QuantityOf<quantity<isq::rotation[one]>, dimension_one>);
static_assert(QuantityOf<quantity<si::radian>, isq::angular_measure>);
static_assert(!QuantityOf<quantity<si::radian>, dimensionless>);
static_assert(QuantityOf<quantity<si::radian>, dimension_one>);
static_assert(QuantityOf<quantity<isq::angular_measure[si::radian]>, isq::angular_measure>);
static_assert(!QuantityOf<quantity<isq::angular_measure[si::radian]>, dimensionless>);
static_assert(QuantityOf<quantity<isq::angular_measure[si::radian]>, dimension_one>);
static_assert(QuantityOf<quantity<one>, isq::rotation>);
static_assert(QuantityOf<quantity<one>, isq::angular_measure>);
static_assert(!QuantityOf<quantity<dimensionless[one]>, isq::rotation>);
static_assert(!QuantityOf<quantity<dimensionless[one]>, isq::angular_measure>);

// QuantityLike
static_assert(QuantityLike<std::chrono::seconds>);
static_assert(QuantityLike<std::chrono::hours>);
static_assert(!QuantityLike<quantity<isq::time[si::second]>>);
static_assert(!QuantityLike<quantity_point<isq::length[si::metre], my_origin>>);
static_assert(!QuantityLike<int>);

// QuantityPoint
static_assert(QuantityPoint<quantity_point<si::metre, my_origin>>);
static_assert(QuantityPoint<quantity_point<si::metre, my_relative_origin>>);
static_assert(QuantityPoint<quantity_point<isq::length[si::metre], my_origin>>);
static_assert(QuantityPoint<quantity_point<isq::length[si::metre], my_relative_origin, int>>);
static_assert(QuantityPoint<quantity_point<isq::radius[si::metre], my_origin>>);
static_assert(QuantityPoint<quantity_point<isq::radius[si::metre], my_relative_origin>>);
static_assert(!QuantityPoint<std::remove_const_t<decltype(isq::length[si::metre])>>);
static_assert(!QuantityPoint<absolute_point_origin<isq::length>>);
static_assert(!QuantityPoint<struct my_origin>);
static_assert(!QuantityPoint<struct my_relative_origin>);
static_assert(!QuantityPoint<std::chrono::seconds>);
static_assert(!QuantityPoint<std::chrono::time_point<std::chrono::system_clock>>);
static_assert(!QuantityPoint<int>);

// QuantityPointOf
static_assert(QuantityPointOf<quantity_point<si::metre, my_origin>, isq::length>);
static_assert(QuantityPointOf<quantity_point<si::metre, my_origin>, isq::radius>);
static_assert(QuantityPointOf<quantity_point<isq::length[si::metre], my_origin>, isq::length>);
static_assert(!QuantityPointOf<quantity_point<isq::length[si::metre], my_origin>, isq::radius>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_origin>, isq::length>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_origin>, isq::radius>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_relative_origin>, isq::length>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_relative_origin>, isq::radius>);
static_assert(QuantityPointOf<quantity_point<si::metre, my_origin>, my_origin>);
static_assert(QuantityPointOf<quantity_point<si::metre, my_origin>, my_relative_origin>);
static_assert(QuantityPointOf<quantity_point<si::metre, my_relative_origin>, my_relative_origin>);
static_assert(QuantityPointOf<quantity_point<si::metre, my_relative_origin>, my_origin>);
static_assert(QuantityPointOf<quantity_point<isq::length[si::metre], my_origin>, my_origin>);
static_assert(QuantityPointOf<quantity_point<isq::length[si::metre], my_origin>, my_relative_origin>);
static_assert(QuantityPointOf<quantity_point<isq::length[si::metre], my_relative_origin>, my_relative_origin>);
static_assert(QuantityPointOf<quantity_point<isq::length[si::metre], my_relative_origin>, my_origin>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_origin>, my_origin>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_origin>, my_relative_origin>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_relative_origin>, my_relative_origin>);
static_assert(QuantityPointOf<quantity_point<isq::radius[si::metre], my_relative_origin>, my_origin>);

// PointOrigin
static_assert(PointOrigin<struct my_origin>);
static_assert(PointOrigin<struct my_relative_origin>);
static_assert(!PointOrigin<absolute_point_origin<isq::length>>);
static_assert(!PointOrigin<relative_point_origin<my_origin + 42 * si::metre>>);
static_assert(!PointOrigin<quantity_point<si::metre, my_origin>>);
static_assert(!PointOrigin<quantity_point<isq::length[si::metre], my_origin>>);
static_assert(!PointOrigin<quantity_point<isq::radius[si::metre], my_origin>>);
static_assert(!PointOrigin<std::remove_const_t<decltype(isq::length[si::metre])>>);
static_assert(!PointOrigin<std::chrono::seconds>);
static_assert(!PointOrigin<std::chrono::time_point<std::chrono::system_clock>>);
static_assert(!PointOrigin<int>);

// PointOriginFor
static_assert(PointOriginFor<struct my_origin, isq::length>);
static_assert(PointOriginFor<struct my_origin, isq::radius>);
static_assert(!PointOriginFor<struct my_origin, isq::time>);
static_assert(PointOriginFor<struct my_relative_origin, isq::length>);
static_assert(PointOriginFor<struct my_relative_origin, isq::radius>);
static_assert(!PointOriginFor<struct my_relative_origin, isq::time>);
static_assert(!PointOriginFor<quantity_point<si::metre, my_origin>, isq::length>);
static_assert(!PointOriginFor<quantity_point<si::metre, my_origin>, isq::radius>);
static_assert(!PointOriginFor<quantity_point<si::metre, my_origin>, isq::time>);
static_assert(!PointOriginFor<quantity_point<isq::length[si::metre], my_origin>, isq::length>);
static_assert(!PointOriginFor<quantity_point<isq::length[si::metre], my_origin>, isq::radius>);
static_assert(!PointOriginFor<quantity_point<isq::length[si::metre], my_origin>, isq::time>);
static_assert(!PointOriginFor<quantity_point<isq::radius[si::metre], my_origin>, isq::length>);
static_assert(!PointOriginFor<quantity_point<isq::radius[si::metre], my_origin>, isq::radius>);
static_assert(!PointOriginFor<quantity_point<isq::radius[si::metre], my_origin>, isq::time>);
static_assert(!PointOriginFor<quantity_point<isq::radius[si::metre], my_relative_origin>, isq::length>);
static_assert(!PointOriginFor<quantity_point<isq::radius[si::metre], my_relative_origin>, isq::radius>);
static_assert(!PointOriginFor<quantity_point<isq::radius[si::metre], my_relative_origin>, isq::time>);
static_assert(!PointOriginFor<std::remove_const_t<decltype(isq::length[si::metre])>, isq::length>);
static_assert(!PointOriginFor<std::chrono::seconds, isq::length>);
static_assert(!PointOriginFor<std::chrono::time_point<std::chrono::system_clock>, isq::length>);
static_assert(!PointOriginFor<int, isq::length>);

// QuantityPointLike
static_assert(QuantityPointLike<std::chrono::time_point<std::chrono::system_clock>>);
static_assert(!QuantityPointLike<std::chrono::seconds>);
static_assert(!QuantityPointLike<quantity<isq::time[si::second]>>);
static_assert(!QuantityPointLike<quantity_point<si::metre, my_origin>>);
static_assert(!QuantityPointLike<int>);

}  // namespace
