
// -----------------------------------------------------------------------------
// SLADE - It's a Doom Editor
// Copyright(C) 2008 - 2022 Simon Judd
//
// Email:       sirjuddington@gmail.com
// Web:         http://slade.mancubus.net
// Filename:    InfoOverlay3d.cpp
// Description: InfoOverlay3d class - a map editor overlay that displays
//              information about the currently highlighted wall/floor/thing in
//              3d mode
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA  02110 - 1301, USA.
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
//
// Includes
//
// -----------------------------------------------------------------------------
#include "Main.h"
#include "InfoOverlay3d.h"
#include "App.h"
#include "Game/Configuration.h"
#include "General/ColourConfiguration.h"
#include "MapEditor/MapEditContext.h"
#include "MapEditor/MapEditor.h"
#include "MapEditor/MapTextureManager.h"
#include "MapEditor/UI/MapEditorWindow.h"
#include "OpenGL/Drawing.h"
#include "OpenGL/OpenGL.h"
#include "SLADEMap/SLADEMap.h"
#include "Utility/StringUtils.h"

using namespace slade;


// -----------------------------------------------------------------------------
//
// External Variables
//
// -----------------------------------------------------------------------------
EXTERN_CVAR(Bool, use_zeth_icons)


// -----------------------------------------------------------------------------
//
// InfoOverlay3D Class Functions
//
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Updates the info text for the object of [item_type] at [item_index] in [map]
// -----------------------------------------------------------------------------
void InfoOverlay3D::update(int item_index, mapeditor::ItemType item_type, SLADEMap* map)
{
	using game::Feature;
	using game::UDMFFeature;

	// Clear current info
	info_.clear();
	info2_.clear();

	// Setup variables
	current_type_   = item_type;
	texname_        = "";
	texture_        = 0;
	thing_icon_     = false;
	auto map_format = mapeditor::editContext().mapDesc().format;

	// Wall
	if (item_type == mapeditor::ItemType::WallBottom || item_type == mapeditor::ItemType::WallMiddle
		|| item_type == mapeditor::ItemType::WallTop)
	{
		// Get line and side
		auto side = map->side(item_index);
		if (!side)
			return;
		auto line = side->parentLine();
		if (!line)
			return;
		object_ = side;

		// --- Line/side info ---
		info_.push_back(fmt::format("Line #{}", line->index()));
		if (side == line->s1())
			info_.push_back(fmt::format("Front Side #{}", side->index()));
		else
			info_.push_back(fmt::format("Back Side #{}", side->index()));

		// Relevant flags
		string flags;
		if (game::configuration().lineBasicFlagSet("dontpegtop", line, map_format))
			flags += "Upper Unpegged, ";
		if (game::configuration().lineBasicFlagSet("dontpegbottom", line, map_format))
			flags += "Lower Unpegged, ";
		if (game::configuration().lineBasicFlagSet("blocking", line, map_format))
			flags += "Blocking, ";
		if (!flags.empty())
			strutil::removeLast(flags, 2);
		info_.push_back(flags);

		info_.push_back(fmt::format("Length: {}", (int)line->length()));

		// Other potential info: special, sector#


		// --- Wall part info ---

		// Part
		if (item_type == mapeditor::ItemType::WallBottom)
			info2_.emplace_back("Lower Texture");
		else if (item_type == mapeditor::ItemType::WallMiddle)
			info2_.emplace_back("Middle Texture");
		else
			info2_.emplace_back("Upper Texture");

		// Offsets
		if (map->currentFormat() == MapFormat::UDMF
			&& game::configuration().featureSupported(UDMFFeature::TextureOffsets))
		{
			// Get x offset info
			int    xoff      = side->texOffsetX();
			double xoff_part = 0;
			if (item_type == mapeditor::ItemType::WallBottom)
				xoff_part = side->floatProperty("offsetx_bottom");
			else if (item_type == mapeditor::ItemType::WallMiddle)
				xoff_part = side->floatProperty("offsetx_mid");
			else
				xoff_part = side->floatProperty("offsetx_top");

			// Add x offset string
			string xoff_info;
			if (xoff_part == 0)
				xoff_info = fmt::format("{}", xoff);
			else if (xoff_part > 0)
				xoff_info = fmt::format("{:1.2f} ({}+{:1.2f})", (double)xoff + xoff_part, xoff, xoff_part);
			else
				xoff_info = fmt::format("{:1.2f} ({}-{:1.2f})", (double)xoff + xoff_part, xoff, -xoff_part);

			// Get y offset info
			int    yoff      = side->texOffsetY();
			double yoff_part = 0;
			if (item_type == mapeditor::ItemType::WallBottom)
				yoff_part = side->floatProperty("offsety_bottom");
			else if (item_type == mapeditor::ItemType::WallMiddle)
				yoff_part = side->floatProperty("offsety_mid");
			else
				yoff_part = side->floatProperty("offsety_top");

			// Add y offset string
			string yoff_info;
			if (yoff_part == 0)
				yoff_info = fmt::format("{}", yoff);
			else if (yoff_part > 0)
				yoff_info = fmt::format("{:1.2f} ({}+{:1.2f})", (double)yoff + yoff_part, yoff, yoff_part);
			else
				yoff_info = fmt::format("{:1.2f} ({}-{:1.2f})", (double)yoff + yoff_part, yoff, -yoff_part);

			info2_.push_back(fmt::format("Offsets: {}, {}", xoff_info, yoff_info));
		}
		else
		{
			// Basic offsets
			info2_.push_back(fmt::format("Offsets: {}, {}", side->texOffsetX(), side->texOffsetY()));
		}

		// UDMF extras
		if (map->currentFormat() == MapFormat::UDMF
			&& game::configuration().featureSupported(UDMFFeature::TextureScaling))
		{
			// Scale
			double xscale, yscale;
			if (item_type == mapeditor::ItemType::WallBottom)
			{
				xscale = side->floatProperty("scalex_bottom");
				yscale = side->floatProperty("scaley_bottom");
			}
			else if (item_type == mapeditor::ItemType::WallMiddle)
			{
				xscale = side->floatProperty("scalex_mid");
				yscale = side->floatProperty("scaley_mid");
			}
			else
			{
				xscale = side->floatProperty("scalex_top");
				yscale = side->floatProperty("scaley_top");
			}
			info2_.push_back(fmt::format("Scale: {:1.2f}x, {:1.2f}x", xscale, yscale));
		}
		else
		{
			info2_.emplace_back("");
		}

		// Height of this section of the wall
		// TODO this is wrong in the case of slopes, but slope support only
		// exists in the 3.1.1 branch
		Vec2d    left_point, right_point;
		MapSide* other_side;
		if (side == line->s1())
		{
			left_point  = line->v1()->position();
			right_point = line->v2()->position();
			other_side  = line->s2();
		}
		else
		{
			left_point  = line->v2()->position();
			right_point = line->v1()->position();
			other_side  = line->s1();
		}

		auto       this_sector  = side->sector();
		MapSector* other_sector = nullptr;
		if (other_side)
			other_sector = other_side->sector();

		double left_height, right_height;
		if (item_type == mapeditor::ItemType::WallMiddle && other_sector)
		{
			// A two-sided line's middle area is the smallest distance between
			// both sides' floors and ceilings, which is more complicated with
			// slopes.
			auto floor1   = this_sector->floor().plane;
			auto floor2   = other_sector->floor().plane;
			auto ceiling1 = this_sector->ceiling().plane;
			auto ceiling2 = other_sector->ceiling().plane;
			left_height   = min(ceiling1.heightAt(left_point), ceiling2.heightAt(left_point))
						  - max(floor1.heightAt(left_point), floor2.heightAt(left_point));
			right_height = min(ceiling1.heightAt(right_point), ceiling2.heightAt(right_point))
						   - max(floor1.heightAt(right_point), floor2.heightAt(right_point));
		}
		else
		{
			Plane top_plane, bottom_plane;
			if (item_type == mapeditor::ItemType::WallMiddle)
			{
				top_plane    = this_sector->ceiling().plane;
				bottom_plane = this_sector->floor().plane;
			}
			else
			{
				if (!other_sector)
					return;
				if (item_type == mapeditor::ItemType::WallTop)
				{
					top_plane    = this_sector->ceiling().plane;
					bottom_plane = other_sector->ceiling().plane;
				}
				else
				{
					top_plane    = other_sector->floor().plane;
					bottom_plane = this_sector->floor().plane;
				}
			}

			left_height  = top_plane.heightAt(left_point) - bottom_plane.heightAt(left_point);
			right_height = top_plane.heightAt(right_point) - bottom_plane.heightAt(right_point);
		}
		if (fabs(left_height - right_height) < 0.001)
			info2_.push_back(fmt::format("Height: {}", (int)left_height));
		else
			info2_.push_back(fmt::format("Height: {} ~ {}", (int)left_height, (int)right_height));

		// Texture
		if (item_type == mapeditor::ItemType::WallBottom)
			texname_ = side->texLower();
		else if (item_type == mapeditor::ItemType::WallMiddle)
			texname_ = side->texMiddle();
		else
			texname_ = side->texUpper();
		texture_ = mapeditor::textureManager()
					   .texture(texname_, game::configuration().featureSupported(Feature::MixTexFlats))
					   .gl_id;
	}


	// Floor
	else if (item_type == mapeditor::ItemType::Floor || item_type == mapeditor::ItemType::Ceiling)
	{
		// Get sector
		auto sector = map->sector(item_index);
		if (!sector)
			return;
		object_ = sector;

		// Get basic info
		int fheight = sector->floor().height;
		int cheight = sector->ceiling().height;

		// --- Sector info ---

		// Sector index
		info_.push_back(fmt::format("Sector #{}", item_index));

		// Sector height
		info_.push_back(fmt::format("Total Height: {}", cheight - fheight));

		// ZDoom UDMF extras
		/*
		if (game::configuration().udmfNamespace() == "zdoom") {
			// Sector colour
			ColRGBA col = sector->getColour(0, true);
			info.push_back(fmt::format("Colour: R{}, G{}, B{}", col.r, col.g, col.b));
		}
		*/


		// --- Flat info ---

		// Height
		if (item_type == mapeditor::ItemType::Floor)
			info2_.push_back(fmt::format("Floor Height: {}", fheight));
		else
			info2_.push_back(fmt::format("Ceiling Height: {}", cheight));

		// Light
		int light = sector->lightLevel();
		if (game::configuration().featureSupported(UDMFFeature::FlatLighting))
		{
			// Get extra light info
			int  fl  = 0;
			bool abs = false;
			if (item_type == mapeditor::ItemType::Floor)
			{
				fl  = sector->intProperty("lightfloor");
				abs = sector->boolProperty("lightfloorabsolute");
			}
			else
			{
				fl  = sector->intProperty("lightceiling");
				abs = sector->boolProperty("lightceilingabsolute");
			}

			// Set if absolute
			if (abs)
			{
				light = fl;
				fl    = 0;
			}

			// Add info string
			if (fl == 0)
				info2_.push_back(fmt::format("Light: {}", light));
			else if (fl > 0)
				info2_.push_back(fmt::format("Light: {} ({}+{})", light + fl, light, fl));
			else
				info2_.push_back(fmt::format("Light: {} ({}-{})", light + fl, light, -fl));
		}
		else
			info2_.push_back(fmt::format("Light: {}", light));

		// UDMF extras
		if (mapeditor::editContext().mapDesc().format == MapFormat::UDMF)
		{
			// Offsets
			double xoff, yoff;
			xoff = yoff = 0.0;
			if (game::configuration().featureSupported(UDMFFeature::FlatPanning))
			{
				if (item_type == mapeditor::ItemType::Floor)
				{
					xoff = sector->floatProperty("xpanningfloor");
					yoff = sector->floatProperty("ypanningfloor");
				}
				else
				{
					xoff = sector->floatProperty("xpanningceiling");
					yoff = sector->floatProperty("ypanningceiling");
				}
			}
			info2_.push_back(fmt::format("Offsets: {:1.2f}, {:1.2f}", xoff, yoff));

			// Scaling
			double xscale, yscale;
			xscale = yscale = 1.0;
			if (game::configuration().featureSupported(UDMFFeature::FlatScaling))
			{
				if (item_type == mapeditor::ItemType::Floor)
				{
					xscale = sector->floatProperty("xscalefloor");
					yscale = sector->floatProperty("yscalefloor");
				}
				else
				{
					xscale = sector->floatProperty("xscaleceiling");
					yscale = sector->floatProperty("yscaleceiling");
				}
			}
			info2_.push_back(fmt::format("Scale: {:1.2f}x, {:1.2f}x", xscale, yscale));
		}

		// Texture
		if (item_type == mapeditor::ItemType::Floor)
			texname_ = sector->floor().texture;
		else
			texname_ = sector->ceiling().texture;
		texture_ = mapeditor::textureManager()
					   .flat(texname_, game::configuration().featureSupported(Feature::MixTexFlats))
					   .gl_id;
	}

	// Thing
	else if (item_type == mapeditor::ItemType::Thing)
	{
		// index, type, position, sector, zpos, height?, radius?

		// Get thing
		auto thing = map->thing(item_index);
		if (!thing)
			return;
		object_ = thing;

		// Index
		info_.push_back(fmt::format("Thing #{}", item_index));

		// Position
		if (mapeditor::editContext().mapDesc().format == MapFormat::Hexen
			|| mapeditor::editContext().mapDesc().format == MapFormat::UDMF)
			info_.push_back(
				fmt::format("Position: {}, {}, {}", (int)thing->xPos(), (int)thing->yPos(), (int)thing->zPos()));
		else
			info_.push_back(fmt::format("Position: {}, {}", (int)thing->xPos(), (int)thing->yPos()));


		// Type
		auto& tt = game::configuration().thingType(thing->type());
		if (!tt.defined())
			info2_.push_back(fmt::format("Type: {}", thing->type()));
		else
			info2_.push_back(fmt::format("Type: {}", tt.name()));

		// Args
		if (mapeditor::editContext().mapDesc().format == MapFormat::Hexen
			|| (mapeditor::editContext().mapDesc().format == MapFormat::UDMF
				&& game::configuration().getUDMFProperty("arg0", MapObject::Type::Thing)))
		{
			// Get thing args
			string argxstr[2];
			argxstr[0]  = thing->stringProperty("arg0str");
			argxstr[1]  = thing->stringProperty("arg1str");
			auto argstr = tt.argSpec().stringDesc(thing->args().data(), argxstr);

			if (argstr.empty())
				info2_.emplace_back("No Args");
			else
				info2_.emplace_back(argstr);
		}

		// Sector
		auto sector = map->sectors().atPos(thing->position());
		if (sector)
			info2_.emplace_back(fmt::format("In Sector #{}", sector->index()));
		else
			info2_.emplace_back("No Sector");


		// Texture
		texture_ = mapeditor::textureManager().sprite(tt.sprite(), tt.translation(), tt.palette()).gl_id;
		if (!texture_)
		{
			if (use_zeth_icons && tt.zethIcon() >= 0)
				texture_ = mapeditor::textureManager()
							   .editorImage(fmt::format("zethicons/zeth{:02d}", tt.zethIcon()))
							   .gl_id;
			if (!texture_)
				texture_ = mapeditor::textureManager().editorImage(fmt::format("thing/{}", tt.icon())).gl_id;
			thing_icon_ = true;
		}
		texname_ = "";
	}

	last_update_ = app::runTimer();
}

// -----------------------------------------------------------------------------
// Draws the overlay
// -----------------------------------------------------------------------------
void InfoOverlay3D::draw(int bottom, int right, int middle, float alpha)
{
	// Don't bother if invisible
	if (alpha <= 0.0f)
		return;

	// Don't bother if no info
	if (info_.empty())
		return;

	// Update if needed
	if (object_
		&& (object_->modifiedTime() > last_update_ || // object updated
			(object_->objType() == MapObject::Type::Side
			 && (dynamic_cast<MapSide*>(object_)->parentLine()->modifiedTime() > last_update_ || // parent line updated
				 dynamic_cast<MapSide*>(object_)->sector()->modifiedTime() > last_update_)))) // parent sector updated
		update(object_->index(), current_type_, object_->parentMap());

	// Init GL stuff
	glLineWidth(1.0f);
	glDisable(GL_LINE_SMOOTH);

	// Determine overlay height
	int nlines = std::max<int>(info_.size(), info2_.size());
	if (nlines < 4)
		nlines = 4;
	double scale       = (drawing::fontSize() / 12.0);
	int    line_height = 16 * scale;
	int    height      = nlines * line_height + 4;

	// Get colours
	ColRGBA col_bg = colourconfig::colour("map_3d_overlay_background");
	ColRGBA col_fg = colourconfig::colour("map_3d_overlay_foreground");
	col_fg.a       = col_fg.a * alpha;
	col_bg.a       = col_bg.a * alpha;
	ColRGBA col_border(0, 0, 0, 140);

	// Slide in/out animation
	float alpha_inv = 1.0f - alpha;
	int   bottom2   = bottom;
	bottom += height * alpha_inv * alpha_inv;

	// Draw overlay background
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	drawing::drawBorderedRect(0, bottom - height - 4, right, bottom + 2, col_bg, col_border);

	// Draw info text lines (left)
	int y = height;
	for (const auto& text : info_)
	{
		drawing::drawText(
			text, middle - (40 * scale) - 4, bottom - y, col_fg, drawing::Font::Condensed, drawing::Align::Right);
		y -= line_height;
	}

	// Draw info text lines (right)
	y = height;
	for (const auto& text : info2_)
	{
		drawing::drawText(text, middle + (40 * scale) + 4, bottom - y, col_fg, drawing::Font::Condensed);
		y -= line_height;
	}

	// Draw texture if any
	drawTexture(alpha, middle - (40 * scale), bottom);

	// Done
	glEnable(GL_LINE_SMOOTH);
}

// -----------------------------------------------------------------------------
// Draws the item texture/graphic box (if any)
// -----------------------------------------------------------------------------
void InfoOverlay3D::drawTexture(float alpha, int x, int y) const
{
	double scale        = (drawing::fontSize() / 12.0);
	int    tex_box_size = 80 * scale;
	int    line_height  = 16 * scale;

	// Get colours
	ColRGBA col_bg = colourconfig::colour("map_3d_overlay_background");
	ColRGBA col_fg = colourconfig::colour("map_3d_overlay_foreground");
	col_fg.a       = col_fg.a * alpha;

	// Check texture exists
	if (texture_)
	{
		// Draw background
		glEnable(GL_TEXTURE_2D);
		gl::setColour(255, 255, 255, 255 * alpha, gl::Blend::Normal);
		glPushMatrix();
		glTranslated(x, y - tex_box_size - line_height, 0);
		drawing::drawTextureTiled(gl::Texture::backgroundTexture(), tex_box_size, tex_box_size);
		glPopMatrix();

		// Draw texture
		if (texture_ && texture_ != gl::Texture::missingTexture())
		{
			gl::setColour(255, 255, 255, 255 * alpha, gl::Blend::Normal);
			drawing::drawTextureWithin(
				texture_, x, y - tex_box_size - line_height, x + tex_box_size, y - line_height, 0);
		}
		else if (texname_ == "-")
		{
			// Draw missing icon
			auto icon = mapeditor::textureManager().editorImage("thing/minus").gl_id;
			glEnable(GL_TEXTURE_2D);
			gl::setColour(180, 0, 0, 255 * alpha, gl::Blend::Normal);
			drawing::drawTextureWithin(
				icon, x, y - tex_box_size - line_height, x + tex_box_size, y - line_height, 0, 0.2);
		}
		else if (texname_ != "-" && texture_ == gl::Texture::missingTexture())
		{
			// Draw unknown icon
			auto icon = mapeditor::textureManager().editorImage("thing/unknown").gl_id;
			glEnable(GL_TEXTURE_2D);
			gl::setColour(180, 0, 0, 255 * alpha, gl::Blend::Normal);
			drawing::drawTextureWithin(
				icon, x, y - tex_box_size - line_height, x + tex_box_size, y - line_height, 0, 0.2);
		}

		glDisable(GL_TEXTURE_2D);

		// Draw outline
		gl::setColour(col_fg.r, col_fg.g, col_fg.b, 255 * alpha, gl::Blend::Normal);
		glLineWidth(1.0f);
		glDisable(GL_LINE_SMOOTH);
		drawing::drawRect(x, y - tex_box_size - line_height, x + tex_box_size, y - line_height);
	}

	// Draw texture name (even if texture is blank)
	auto tn_truncated = texname_;
	if (tn_truncated.size() > 8)
	{
		strutil::truncateIP(tn_truncated, 8);
		tn_truncated.append("...");
	}
	drawing::drawText(
		tn_truncated,
		x + (tex_box_size * 0.5),
		y - line_height,
		col_fg,
		drawing::Font::Condensed,
		drawing::Align::Center);
}
