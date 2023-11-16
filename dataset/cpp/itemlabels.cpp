#include "itemlabels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "control.h"
#include "cursor.h"
#include "engine/point.hpp"
#include "engine/render/clx_render.hpp"
#include "gmenu.h"
#include "inv.h"
#include "options.h"
#include "qol/stash.h"
#include "stores.h"
#include "utils/algorithm/container.hpp"
#include "utils/format_int.hpp"
#include "utils/language.h"

namespace devilution {

namespace {

struct ItemLabel {
	int id, width;
	Point pos;
	StringOrView text;
};

std::vector<ItemLabel> labelQueue;

bool highlightKeyPressed = false;
bool isLabelHighlighted = false;
std::array<std::optional<int>, ITEMTYPES> labelCenterOffsets;

const int BorderX = 4;               // minimal horizontal space between labels
const int BorderY = 2;               // minimal vertical space between labels
const int MarginX = 2;               // horizontal margins between text and edges of the label
const int MarginY = 1;               // vertical margins between text and edges of the label
const int Height = 11 + MarginY * 2; // going above 13 scatters labels of items that are next to each other

/**
 * @brief The set of used X coordinates for a certain Y coordinate.
 */
class UsedX {
public:
	[[nodiscard]] bool contains(int val) const
	{
		return c_find(data_, val) != data_.end();
	}

	void insert(int val)
	{
		if (!contains(val))
			data_.push_back(val);
	}

	void clear()
	{
		data_.clear();
	}

private:
	std::vector<int> data_;
};

} // namespace

void ToggleItemLabelHighlight()
{
	sgOptions.Gameplay.showItemLabels.SetValue(!*sgOptions.Gameplay.showItemLabels);
}

void HighlightKeyPressed(bool pressed)
{
	highlightKeyPressed = pressed;
}

bool IsItemLabelHighlighted()
{
	return isLabelHighlighted;
}

void ResetItemlabelHighlighted()
{
	isLabelHighlighted = false;
}

bool IsHighlightingLabelsEnabled()
{
	return stextflag == TalkID::None && highlightKeyPressed != *sgOptions.Gameplay.showItemLabels;
}

void AddItemToLabelQueue(int id, Point position)
{
	if (!IsHighlightingLabelsEnabled())
		return;
	Item &item = Items[id];

	StringOrView textOnGround;
	if (item._itype == ItemType::Gold) {
		textOnGround = fmt::format(fmt::runtime(_("{:s} gold")), FormatInteger(item._ivalue));
	} else {
		textOnGround = item.getName();
	}

	int nameWidth = GetLineWidth(textOnGround);
	nameWidth += MarginX * 2;
	int index = ItemCAnimTbl[item._iCurs];
	if (!labelCenterOffsets[index]) {
		const auto [xBegin, xEnd] = ClxMeasureSolidHorizontalBounds((*item.AnimInfo.sprites)[item.AnimInfo.currentFrame]);
		labelCenterOffsets[index].emplace((xBegin + xEnd) / 2);
	}

	position.x += *labelCenterOffsets[index];
	position.y -= TILE_HEIGHT;
	if (*sgOptions.Graphics.zoom) {
		position *= 2;
	}
	position.x -= nameWidth / 2;
	position.y -= Height;
	labelQueue.push_back(ItemLabel { id, nameWidth, position, std::move(textOnGround) });
}

bool IsMouseOverGameArea()
{
	if ((IsRightPanelOpen()) && GetRightPanel().contains(MousePosition))
		return false;
	if ((IsLeftPanelOpen()) && GetLeftPanel().contains(MousePosition))
		return false;
	if (GetMainPanel().contains(MousePosition))
		return false;

	return true;
}

void DrawItemNameLabels(const Surface &out)
{
	const Surface clippedOut = out.subregionY(0, gnViewportHeight);
	isLabelHighlighted = false;
	if (labelQueue.empty())
		return;
	UsedX usedX;

	for (unsigned i = 0; i < labelQueue.size(); ++i) {
		usedX.clear();

		bool canShow;
		do {
			canShow = true;
			for (unsigned j = 0; j < i; ++j) {
				ItemLabel &a = labelQueue[i];
				ItemLabel &b = labelQueue[j];
				if (std::abs(b.pos.y - a.pos.y) < Height + BorderY) {
					const int widthA = a.width + BorderX + MarginX * 2;
					const int widthB = b.width + BorderX + MarginX * 2;
					int newpos = b.pos.x;
					if (b.pos.x >= a.pos.x && b.pos.x - a.pos.x < widthA) {
						newpos -= widthA;
						if (usedX.contains(newpos))
							newpos = b.pos.x + widthB;
					} else if (b.pos.x < a.pos.x && a.pos.x - b.pos.x < widthB) {
						newpos += widthB;
						if (usedX.contains(newpos))
							newpos = b.pos.x - widthA;
					} else
						continue;
					canShow = false;
					a.pos.x = newpos;
					usedX.insert(newpos);
				}
			}
		} while (!canShow);
	}

	for (const ItemLabel &label : labelQueue) {
		Item &item = Items[label.id];

		if (MousePosition.x >= label.pos.x && MousePosition.x < label.pos.x + label.width && MousePosition.y >= label.pos.y + MarginY && MousePosition.y < label.pos.y + MarginY + Height) {
			if (!gmenu_is_active()
			    && PauseMode == 0
			    && !MyPlayerIsDead
			    && stextflag == TalkID::None
			    && IsMouseOverGameArea()
			    && LastMouseButtonAction == MouseActionType::None) {
				isLabelHighlighted = true;
				cursPosition = item.position;
				pcursitem = label.id;
			}
		}
		if (pcursitem == label.id && stextflag == TalkID::None)
			FillRect(clippedOut, label.pos.x, label.pos.y + MarginY, label.width, Height, PAL8_BLUE + 6);
		else
			DrawHalfTransparentRectTo(clippedOut, label.pos.x, label.pos.y + MarginY, label.width, Height);
		DrawString(clippedOut, label.text, { { label.pos.x + MarginX, label.pos.y }, { label.width, Height } },
		    { .flags = item.getTextColor() });
	}
	labelQueue.clear();
}

} // namespace devilution
