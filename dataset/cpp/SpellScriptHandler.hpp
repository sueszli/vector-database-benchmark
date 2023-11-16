/*
 * ArcEmu MMORPG Server
 * Copyright (C) 2008-2023 <http://www.ArcEmu.org/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef APE_SPELLSCRIPTHANDLER_HPP_
#define APE_SPELLSCRIPTHANDLER_HPP_

class Spell;
class Aura;

class SpellScriptHandler
{
public:
	/// Call the dummy spell handler Python function
	static bool handleDummySpell( uint32 spellEffectIndex, Spell *spell );

	/// Call the scripted effect handler Python function
	static bool handleScriptedEffect( uint32 spellEffectIndex, Spell *spell );

	/// Call the dummy aura handler Python function
	static bool handleDummyAura( uint32 i, Aura *aura, bool apply );
};

#endif
