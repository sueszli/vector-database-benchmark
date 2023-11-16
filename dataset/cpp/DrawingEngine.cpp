/*
 * Copyright 2001-2018, Haiku, Inc.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Stephan Aßmus <superstippi@gmx.de>
 *		Julian Harnath <julian.harnath@rwth-aachen.de>
 */


#include "DrawingEngine.h"

#include <Bitmap.h>
#include <StackOrHeapArray.h>

#include <stdio.h>

#include <algorithm>
#include <stack>

#include "DrawState.h"
#include "GlyphLayoutEngine.h"
#include "Painter.h"
#include "ServerBitmap.h"
#include "ServerCursor.h"
#include "RenderingBuffer.h"

#include "drawing_support.h"


#if DEBUG
#	define ASSERT_PARALLEL_LOCKED() \
	{ if (!IsParallelAccessLocked()) debugger("not parallel locked!"); }
#	define ASSERT_EXCLUSIVE_LOCKED() \
	{ if (!IsExclusiveAccessLocked()) debugger("not exclusive locked!"); }
#else
#	define ASSERT_PARALLEL_LOCKED()
#	define ASSERT_EXCLUSIVE_LOCKED()
#endif


static inline void
make_rect_valid(BRect& rect)
{
	if (rect.left > rect.right) {
		float temp = rect.left;
		rect.left = rect.right;
		rect.right = temp;
	}
	if (rect.top > rect.bottom) {
		float temp = rect.top;
		rect.top = rect.bottom;
		rect.bottom = temp;
	}
}


static inline void
extend_by_stroke_width(BRect& rect, float penSize)
{
	// "- 0.5" because if stroke width == 1, we don't need to extend
	float inset = -ceilf(penSize / 2.0 - 0.5);
	rect.InsetBy(inset, inset);
}


class AutoFloatingOverlaysHider {
	public:
		AutoFloatingOverlaysHider(HWInterface* interface, const BRect& area)
			:
			fInterface(interface),
			fHidden(interface->HideFloatingOverlays(area))
		{
		}

		AutoFloatingOverlaysHider(HWInterface* interface)
			:
			fInterface(interface),
			fHidden(fInterface->HideFloatingOverlays())
		{
		}

		~AutoFloatingOverlaysHider()
		{
			if (fHidden)
				fInterface->ShowFloatingOverlays();
		}

		bool WasHidden() const
		{
			return fHidden;
		}

	private:
		HWInterface*	fInterface;
		bool			fHidden;

};

class DrawTransaction {
public:
	DrawTransaction(DrawingEngine *engine, const BRect &bounds)
		:
		fEngine(engine),
		fOverlaysHidden(false)
	{
		fDirty.Set(bounds);
		fDirty.IntersectWith(fEngine->fPainter->ClippingRegion());
		if (fDirty.CountRects() == 0)
			return;
		fOverlaysHidden
			= fEngine->fGraphicsCard->HideFloatingOverlays(fDirty.Frame());
	}

	DrawTransaction(DrawingEngine *engine)
		:
		fEngine(engine),
		fOverlaysHidden(false)
	{
		fDirty = *fEngine->fPainter->ClippingRegion();
		if (fDirty.CountRects() == 0)
			return;
		fOverlaysHidden
			= fEngine->fGraphicsCard->HideFloatingOverlays(fDirty.Frame());
	}

	DrawTransaction(DrawingEngine *engine, const BRegion &region)
		:
		fEngine(engine),
		fOverlaysHidden(false)
	{
		// region is already clipped
		fDirty = region;
		if (fDirty.CountRects() == 0)
			return;
		fOverlaysHidden
			= fEngine->fGraphicsCard->HideFloatingOverlays(fDirty.Frame());
	}

	~DrawTransaction()
	{
		if (fEngine->fCopyToFront)
			fEngine->fGraphicsCard->InvalidateRegion(fDirty);
		if (fOverlaysHidden)
			fEngine->fGraphicsCard->ShowFloatingOverlays();
	}

	bool IsDirty() const
	{
		return fDirty.CountRects() > 0;
	}

	void SetDirty(const BRect &rect)
	{
		fDirty.Set(rect);
		fDirty.IntersectWith(fEngine->fPainter->ClippingRegion());
	}

	const BRegion &DirtyRegion() const
	{
		return fDirty;
	}

	bool WasOverlaysHidden() const
	{
		return fOverlaysHidden;
	}

private:
	DrawingEngine *fEngine;
	bool fOverlaysHidden;
	BRegion fDirty;
};


//	#pragma mark -


DrawingEngine::DrawingEngine(HWInterface* interface)
	:
	fPainter(new Painter()),
	fGraphicsCard(NULL),
	fAvailableHWAccleration(0),
	fSuspendSyncLevel(0),
	fCopyToFront(true)
{
	SetHWInterface(interface);
}


DrawingEngine::~DrawingEngine()
{
	SetHWInterface(NULL);
}


// #pragma mark - locking


bool
DrawingEngine::LockParallelAccess()
{
	return fGraphicsCard->LockParallelAccess();
}


#if DEBUG
bool
DrawingEngine::IsParallelAccessLocked() const
{
	return fGraphicsCard->IsParallelAccessLocked();
}
#endif


void
DrawingEngine::UnlockParallelAccess()
{
	fGraphicsCard->UnlockParallelAccess();
}


bool
DrawingEngine::LockExclusiveAccess()
{
	return fGraphicsCard->LockExclusiveAccess();
}


bool
DrawingEngine::IsExclusiveAccessLocked() const
{
	return fGraphicsCard->IsExclusiveAccessLocked();
}


void
DrawingEngine::UnlockExclusiveAccess()
{
	fGraphicsCard->UnlockExclusiveAccess();
}


// #pragma mark -


void
DrawingEngine::FrameBufferChanged()
{
	if (!fGraphicsCard) {
		fPainter->DetachFromBuffer();
		fAvailableHWAccleration = 0;
		return;
	}

	// NOTE: locking is probably bogus, since we are called
	// in the thread that changed the frame buffer...
	if (LockExclusiveAccess()) {
		fPainter->AttachToBuffer(fGraphicsCard->DrawingBuffer());
		// available HW acceleration might have changed
		fAvailableHWAccleration = fGraphicsCard->AvailableHWAcceleration();
		UnlockExclusiveAccess();
	}
}


void
DrawingEngine::SetHWInterface(HWInterface* interface)
{
	if (fGraphicsCard == interface)
		return;

	if (fGraphicsCard)
		fGraphicsCard->RemoveListener(this);

	fGraphicsCard = interface;

	if (fGraphicsCard)
		fGraphicsCard->AddListener(this);

	FrameBufferChanged();
}


void
DrawingEngine::SetCopyToFrontEnabled(bool enable)
{
	fCopyToFront = enable;
}


void
DrawingEngine::CopyToFront(/*const*/ BRegion& region)
{
	fGraphicsCard->InvalidateRegion(region);
}


// #pragma mark -


//! the DrawingEngine needs to be locked!
void
DrawingEngine::ConstrainClippingRegion(const BRegion* region)
{
	ASSERT_PARALLEL_LOCKED();

	fPainter->ConstrainClipping(region);
}


void
DrawingEngine::SetDrawState(const DrawState* state, int32 xOffset,
	int32 yOffset)
{
	fPainter->SetDrawState(state, xOffset, yOffset);
}


void
DrawingEngine::SetHighColor(const rgb_color& color)
{
	fPainter->SetHighColor(color);
}


void
DrawingEngine::SetLowColor(const rgb_color& color)
{
	fPainter->SetLowColor(color);
}


void
DrawingEngine::SetPenSize(float size)
{
	fPainter->SetPenSize(size);
}


void
DrawingEngine::SetStrokeMode(cap_mode lineCap, join_mode joinMode,
	float miterLimit)
{
	fPainter->SetStrokeMode(lineCap, joinMode, miterLimit);
}


void
DrawingEngine::SetFillRule(int32 fillRule)
{
	fPainter->SetFillRule(fillRule);
}


void
DrawingEngine::SetBlendingMode(source_alpha srcAlpha, alpha_function alphaFunc)
{
	fPainter->SetBlendingMode(srcAlpha, alphaFunc);
}


void
DrawingEngine::SetPattern(const struct pattern& pattern)
{
	fPainter->SetPattern(pattern);
}


void
DrawingEngine::SetDrawingMode(drawing_mode mode)
{
	fPainter->SetDrawingMode(mode);
}


void
DrawingEngine::SetDrawingMode(drawing_mode mode, drawing_mode& oldMode)
{
	oldMode = fPainter->DrawingMode();
	fPainter->SetDrawingMode(mode);
}


void
DrawingEngine::SetFont(const ServerFont& font)
{
	fPainter->SetFont(font);
}


void
DrawingEngine::SetFont(const DrawState* state)
{
	fPainter->SetFont(state);
}


void
DrawingEngine::SetTransform(const BAffineTransform& transform, int32 xOffset,
	int32 yOffset)
{
	fPainter->SetTransform(transform, xOffset, yOffset);
}


// #pragma mark -


void
DrawingEngine::SuspendAutoSync()
{
	ASSERT_PARALLEL_LOCKED();

	fSuspendSyncLevel++;
}


void
DrawingEngine::Sync()
{
	ASSERT_PARALLEL_LOCKED();

	fSuspendSyncLevel--;
	if (fSuspendSyncLevel == 0)
		fGraphicsCard->Sync();
}


// #pragma mark -


// CopyRegion() does a topological sort of the rects in the
// region. The algorithm was suggested by Ingo Weinhold.
// It compares each rect with each rect and builds a tree
// of successors so we know the order in which they can be copied.
// For example, let's suppose these rects are in a BRegion:
//                        ************
//                        *    B     *
//                        ************
//      *************
//      *           *
//      *     A     ****************
//      *           **             *
//      **************             *
//                   *     C       *
//                   *             *
//                   *             *
//                   ***************
// When copying stuff from LEFT TO RIGHT, TOP TO BOTTOM, the
// result of the sort will be C, A, B. For this direction, we search
// for the rects that have no neighbors to their right and to their
// bottom, These can be copied without drawing into the area of
// rects yet to be copied. If you move from RIGHT TO LEFT, BOTTOM TO TOP,
// you go look for the ones that have no neighbors to their top and left.
//
// Here I draw some rays to illustrate LEFT TO RIGHT, TOP TO BOTTOM:
//                        ************
//                        *    B     *
//                        ************
//      *************
//      *           *
//      *     A     ****************-----------------
//      *           **             *
//      **************             *
//                   *     C       *
//                   *             *
//                   *             *
//                   ***************
//                   |
//                   |
//                   |
//                   |
// There are no rects in the area defined by the rays to the right
// and bottom of rect C, so that's the one we want to copy first
// (for positive x and y offsets).
// Since A is to the left of C and B is to the top of C, The "node"
// for C will point to the nodes of A and B as its "successors". Therefor,
// A and B will have an "indegree" of 1 for C pointing to them. C will
// have an "indegree" of 0, because there was no rect to which C
// was to the left or top of. When comparing A and B, neither is left
// or top from the other and in the sense that the algorithm cares about.

// NOTE: comparison of coordinates assumes that rects don't overlap
// and don't share the actual edge either (as is the case in BRegions).

struct node {
	node()
	{
		pointers = NULL;
	}

	node(const BRect& r, int32 maxPointers)
	{
		init(r, maxPointers);
	}

	~node()
	{
		delete [] pointers;
	}

	void init(const BRect& r, int32 maxPointers)
	{
		rect = r;
		pointers = new(std::nothrow) node*[maxPointers];
		in_degree = 0;
		next_pointer = 0;
	}

	void push(node* node)
	{
		pointers[next_pointer] = node;
		next_pointer++;
	}

	node* top()
	{
		return pointers[next_pointer];
	}

	node* pop()
	{
		node* ret = top();
		next_pointer--;
		return ret;
	}

	BRect	rect;
	int32	in_degree;
	node**	pointers;
	int32	next_pointer;
};


static bool
is_left_of(const BRect& a, const BRect& b)
{
	return (a.right < b.left);
}


static bool
is_above(const BRect& a, const BRect& b)
{
	return (a.bottom < b.top);
}


void
DrawingEngine::CopyRegion(/*const*/ BRegion* region, int32 xOffset,
	int32 yOffset)
{
	// NOTE: region is already clipped
	ASSERT_PARALLEL_LOCKED();

	BRect frame = region->Frame();
	frame = frame | frame.OffsetByCopy(xOffset, yOffset);

	AutoFloatingOverlaysHider _(fGraphicsCard, frame);

	int32 count = region->CountRects();

	// TODO: make this step unnecessary
	// (by using different stack impl inside node)
	BStackOrHeapArray<node, 64> nodes(count);
	for (int32 i= 0; i < count; i++) {
		nodes[i].init(region->RectAt(i), count);
		if (nodes[i].pointers == NULL)
			return;
	}

	for (int32 i = 0; i < count; i++) {
		BRect a = region->RectAt(i);
		for (int32 k = i + 1; k < count; k++) {
			BRect b = region->RectAt(k);
			int cmp = 0;
			// compare horizontally
			if (xOffset > 0) {
				if (is_left_of(a, b)) {
					cmp -= 1;
				} else if (is_left_of(b, a)) {
					cmp += 1;
				}
			} else if (xOffset < 0) {
				if (is_left_of(a, b)) {
					cmp += 1;
				} else if (is_left_of(b, a)) {
					cmp -= 1;
				}
			}
			// compare vertically
			if (yOffset > 0) {
				if (is_above(a, b)) {
					cmp -= 1;
				} else if (is_above(b, a)) {
					cmp += 1;
				}
			} else if (yOffset < 0) {
				if (is_above(a, b)) {
					cmp += 1;
				} else if (is_above(b, a)) {
					cmp -= 1;
				}
			}
			// add appropriate node as successor
			if (cmp > 0) {
				nodes[i].push(&nodes[k]);
				nodes[k].in_degree++;
			} else if (cmp < 0) {
				nodes[k].push(&nodes[i]);
				nodes[i].in_degree++;
			}
		}
	}
	// put all nodes onto a stack that have an "indegree" count of zero
	std::stack<node*> inDegreeZeroNodes;
	for (int32 i = 0; i < count; i++) {
		if (nodes[i].in_degree == 0) {
			inDegreeZeroNodes.push(&nodes[i]);
		}
	}
	// pop the rects from the stack, do the actual copy operation
	// and decrease the "indegree" count of the other rects not
	// currently on the stack and to which the current rect pointed
	// to. If their "indegree" count reaches zero, put them onto the
	// stack as well.

	clipping_rect* sortedRectList = NULL;
	int32 nextSortedIndex = 0;

	if (fAvailableHWAccleration & HW_ACC_COPY_REGION) {
		sortedRectList = new(std::nothrow) clipping_rect[count];
		if (sortedRectList == NULL)
			return;
	}

	while (!inDegreeZeroNodes.empty()) {
		node* n = inDegreeZeroNodes.top();
		inDegreeZeroNodes.pop();

		// do the software implementation or add to sorted
		// rect list for using the HW accelerated version
		// later
		if (sortedRectList) {
			sortedRectList[nextSortedIndex].left	= (int32)n->rect.left;
			sortedRectList[nextSortedIndex].top		= (int32)n->rect.top;
			sortedRectList[nextSortedIndex].right	= (int32)n->rect.right;
			sortedRectList[nextSortedIndex].bottom	= (int32)n->rect.bottom;
			nextSortedIndex++;
		} else {
			BRect touched = CopyRect(n->rect, xOffset, yOffset);
			fGraphicsCard->Invalidate(touched);
		}

		for (int32 k = 0; k < n->next_pointer; k++) {
			n->pointers[k]->in_degree--;
			if (n->pointers[k]->in_degree == 0)
				inDegreeZeroNodes.push(n->pointers[k]);
		}
	}

	// trigger the HW accelerated version if it was available
	if (sortedRectList) {
		fGraphicsCard->CopyRegion(sortedRectList, count, xOffset, yOffset);
		if (fGraphicsCard->IsDoubleBuffered()) {
			fGraphicsCard->Invalidate(
				region->Frame().OffsetByCopy(xOffset, yOffset));
		}
	}

	delete[] sortedRectList;
}


void
DrawingEngine::InvertRect(BRect r)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	// NOTE: Currently ignores view transformation, so no TransformAndClipRect()
	DrawTransaction transaction(this, fPainter->ClipRect(r));
	if (!transaction.IsDirty())
		return;

	// try hardware optimized version first
	if (fAvailableHWAccleration & HW_ACC_INVERT_REGION) {
		BRegion region(r);
		region.IntersectWith(fPainter->ClippingRegion());
		fGraphicsCard->InvertRegion(region);
	} else {
		fPainter->InvertRect(r);
	}
}


void
DrawingEngine::DrawBitmap(ServerBitmap* bitmap, const BRect& bitmapRect,
	const BRect& viewRect, uint32 options)
{
	ASSERT_PARALLEL_LOCKED();

	DrawTransaction transaction(this, fPainter->TransformAndClipRect(viewRect));
	if (transaction.IsDirty())
		fPainter->DrawBitmap(bitmap, bitmapRect, viewRect, options);
}


void
DrawingEngine::DrawArc(BRect r, const float& angle, const float& span,
	bool filled)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	fPainter->AlignEllipseRect(&r, filled);

	BRect clipped(r);
	if (!filled)
		extend_by_stroke_width(clipped, fPainter->PenSize());
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(clipped));
	if (!transaction.IsDirty())
		return;

	float xRadius = r.Width() / 2.0;
	float yRadius = r.Height() / 2.0;
	BPoint center(r.left + xRadius,
				  r.top + yRadius);

	if (filled)
		fPainter->FillArc(center, xRadius, yRadius, angle, span);
	else
		fPainter->StrokeArc(center, xRadius, yRadius, angle, span);
}


void
DrawingEngine::FillArc(BRect r, const float& angle, const float& span,
	const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	fPainter->AlignEllipseRect(&r, true);
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(r));
	if (!transaction.IsDirty())
		return;

	float xRadius = r.Width() / 2.0;
	float yRadius = r.Height() / 2.0;
	BPoint center(r.left + xRadius,
				  r.top + yRadius);

	fPainter->FillArc(center, xRadius, yRadius, angle, span, gradient);
}


void
DrawingEngine::DrawBezier(BPoint* pts, bool filled)
{
	ASSERT_PARALLEL_LOCKED();

	// TODO: figure out bounds and hide cursor depending on that
	DrawTransaction transaction(this);

	transaction.SetDirty(fPainter->DrawBezier(pts, filled));
}


void
DrawingEngine::FillBezier(BPoint* pts, const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	// TODO: figure out bounds and hide cursor depending on that
	DrawTransaction transaction(this);

	transaction.SetDirty(fPainter->FillBezier(pts, gradient));
}


void
DrawingEngine::DrawEllipse(BRect r, bool filled)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	BRect clipped = r;
	fPainter->AlignEllipseRect(&clipped, filled);

	if (!filled)
		extend_by_stroke_width(clipped, fPainter->PenSize());

	clipped.left = floorf(clipped.left);
	clipped.top = floorf(clipped.top);
	clipped.right = ceilf(clipped.right);
	clipped.bottom = ceilf(clipped.bottom);

	DrawTransaction transaction(this, fPainter->TransformAndClipRect(clipped));
	if (!transaction.IsDirty())
		return;

	fPainter->DrawEllipse(r, filled);
}


void
DrawingEngine::FillEllipse(BRect r, const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	BRect clipped = r;
	fPainter->AlignEllipseRect(&clipped, true);

	clipped.left = floorf(clipped.left);
	clipped.top = floorf(clipped.top);
	clipped.right = ceilf(clipped.right);
	clipped.bottom = ceilf(clipped.bottom);

	DrawTransaction transaction(this, fPainter->TransformAndClipRect(clipped));
	if (!transaction.IsDirty())
		return;

	fPainter->FillEllipse(r, gradient);
}


void
DrawingEngine::DrawPolygon(BPoint* ptlist, int32 numpts, BRect bounds,
	bool filled, bool closed)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(bounds);
	if (!filled)
		extend_by_stroke_width(bounds, fPainter->PenSize());
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(bounds));
	if (!transaction.IsDirty())
		return;

	fPainter->DrawPolygon(ptlist, numpts, filled, closed);
}


void
DrawingEngine::FillPolygon(BPoint* ptlist, int32 numpts, BRect bounds,
	const BGradient& gradient, bool closed)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(bounds);
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(bounds));
	if (!transaction.IsDirty())
		return;

	fPainter->FillPolygon(ptlist, numpts, gradient, closed);
}


// #pragma mark - rgb_color


void
DrawingEngine::StrokePoint(const BPoint& pt, const rgb_color& color)
{
	StrokeLine(pt, pt, color);
}


/*!	This function is only used by Decorators,
	it assumes a one pixel wide line
*/
void
DrawingEngine::StrokeLine(const BPoint& start, const BPoint& end,
	const rgb_color& color)
{
	ASSERT_PARALLEL_LOCKED();

	BRect touched(start, end);
	make_rect_valid(touched);
	touched = fPainter->ClipRect(touched);
	DrawTransaction transaction(this, touched);

	if (!fPainter->StraightLine(start, end, color)) {
		rgb_color previousColor = fPainter->HighColor();
		drawing_mode previousMode = fPainter->DrawingMode();

		fPainter->SetHighColor(color);
		fPainter->SetDrawingMode(B_OP_OVER);
		fPainter->StrokeLine(start, end);

		fPainter->SetDrawingMode(previousMode);
		fPainter->SetHighColor(previousColor);
	}
}


//!	This function is used to draw a one pixel wide rect
void
DrawingEngine::StrokeRect(BRect r, const rgb_color& color)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	DrawTransaction transaction(this, fPainter->ClipRect(r));
	if (!transaction.IsDirty())
		return;

	fPainter->StrokeRect(r, color);
}


void
DrawingEngine::FillRect(BRect r, const rgb_color& color)
{
	ASSERT_PARALLEL_LOCKED();

	// NOTE: Write locking because we might use HW acceleration.
	// This needs to be investigated, I'm doing this because of
	// gut feeling.
	make_rect_valid(r);
	r = fPainter->ClipRect(r);
	DrawTransaction transaction(this, r);
	if (!transaction.IsDirty())
		return;

	// try hardware optimized version first
	if (fAvailableHWAccleration & HW_ACC_FILL_REGION) {
		BRegion region(r);
		region.IntersectWith(fPainter->ClippingRegion());
		fGraphicsCard->FillRegion(region, color,
			fSuspendSyncLevel == 0 || transaction.WasOverlaysHidden());
	} else {
		fPainter->FillRect(r, color);
	}
}


void
DrawingEngine::FillRegion(BRegion& r, const rgb_color& color)
{
	ASSERT_PARALLEL_LOCKED();

	// NOTE: region expected to be already clipped correctly!!
	BRect frame = r.Frame();
	if (!fPainter->Bounds().Contains(frame)) {
		// NOTE: I am not quite sure yet how this can happen, but apparently it
		// can (see bug #634).
		// This function is used for internal app_server painting, in the case of
		// bug #634, the background of views is painted. But the view region
		// should never be outside the frame buffer bounds.
//		char message[1024];
//		BRect bounds = fPainter->Bounds();
//		sprintf(message, "FillRegion() - painter: (%d, %d)->(%d, %d), region: (%d, %d)->(%d, %d)",
//			(int)bounds.left, (int)bounds.top, (int)bounds.right, (int)bounds.bottom,
//			(int)frame.left, (int)frame.top, (int)frame.right, (int)frame.bottom);
//		debugger(message);
		return;
	}

	DrawTransaction transaction(this, r);

	// try hardware optimized version first
	if ((fAvailableHWAccleration & HW_ACC_FILL_REGION) != 0
		&& frame.Width() * frame.Height() > 100) {
		fGraphicsCard->FillRegion(r, color, fSuspendSyncLevel == 0
			|| transaction.WasOverlaysHidden());
	} else {
		int32 count = r.CountRects();
		for (int32 i = 0; i < count; i++)
			fPainter->FillRectNoClipping(r.RectAtInt(i), color);
	}
}


// #pragma mark - DrawState


void
DrawingEngine::StrokeRect(BRect r)
{
	ASSERT_PARALLEL_LOCKED();

	// support invalid rects
	make_rect_valid(r);
	BRect clipped(r);
	extend_by_stroke_width(clipped, fPainter->PenSize());
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(clipped));
	if (!transaction.IsDirty())
		return;

	fPainter->StrokeRect(r);
}


void
DrawingEngine::FillRect(BRect r)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);

	r = fPainter->AlignRect(r);

	DrawTransaction transaction(this, fPainter->TransformAndClipRect(r));
	if (!transaction.IsDirty())
		return;

	if (fPainter->IsIdentityTransform()) {
		// TODO the accelerated code path may also be used for transforms that
		// only scale and translate (but don't shear or rotate).

		if ((r.Width() + 1) * (r.Height() + 1) > 100.0) {
			// try hardware optimized version first
			// if the rect is large enough
			if ((fAvailableHWAccleration & HW_ACC_FILL_REGION) != 0) {
				if (fPainter->Pattern() == B_SOLID_HIGH
					&& (fPainter->DrawingMode() == B_OP_COPY
						|| fPainter->DrawingMode() == B_OP_OVER)) {
					BRegion region(r);
					region.IntersectWith(fPainter->ClippingRegion());
					fGraphicsCard->FillRegion(region, fPainter->HighColor(),
						fSuspendSyncLevel == 0
							|| transaction.WasOverlaysHidden());
					return;
				} else if (fPainter->Pattern() == B_SOLID_LOW
						&& fPainter->DrawingMode() == B_OP_COPY) {
					BRegion region(r);
					region.IntersectWith(fPainter->ClippingRegion());
					fGraphicsCard->FillRegion(region, fPainter->LowColor(),
						fSuspendSyncLevel == 0
							|| transaction.WasOverlaysHidden());
					return;
				}
			}
		}

		if ((fAvailableHWAccleration & HW_ACC_INVERT_REGION) != 0
			&& fPainter->Pattern() == B_SOLID_HIGH
			&& fPainter->DrawingMode() == B_OP_INVERT) {
			BRegion region(r);
			region.IntersectWith(fPainter->ClippingRegion());
			fGraphicsCard->InvertRegion(region);
			return;
		}
	}

	fPainter->FillRect(r);
}


void
DrawingEngine::FillRect(BRect r, const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	r = fPainter->AlignRect(r);

	DrawTransaction transaction(this, fPainter->TransformAndClipRect(r));
	if (!transaction.IsDirty())
		return;

	fPainter->FillRect(r, gradient);
}


void
DrawingEngine::FillRegion(BRegion& r)
{
	ASSERT_PARALLEL_LOCKED();

	BRect clipped = fPainter->TransformAndClipRect(r.Frame());
	DrawTransaction transaction(this, clipped);
	if (!transaction.IsDirty())
		return;

	if (fPainter->IsIdentityTransform()) {
		// try hardware optimized version first
		if ((fAvailableHWAccleration & HW_ACC_FILL_REGION) != 0) {
			if (fPainter->Pattern() == B_SOLID_HIGH
				&& (fPainter->DrawingMode() == B_OP_COPY
					|| fPainter->DrawingMode() == B_OP_OVER)) {
				r.IntersectWith(fPainter->ClippingRegion());
				fGraphicsCard->FillRegion(r, fPainter->HighColor(),
					fSuspendSyncLevel == 0 || transaction.WasOverlaysHidden());
				return;
			} else if (fPainter->Pattern() == B_SOLID_LOW
				&& fPainter->DrawingMode() == B_OP_COPY) {
				r.IntersectWith(fPainter->ClippingRegion());
				fGraphicsCard->FillRegion(r, fPainter->LowColor(),
					fSuspendSyncLevel == 0 || transaction.WasOverlaysHidden());
				return;
			}
		}

		if ((fAvailableHWAccleration & HW_ACC_INVERT_REGION) != 0
			&& fPainter->Pattern() == B_SOLID_HIGH
			&& fPainter->DrawingMode() == B_OP_INVERT) {
			r.IntersectWith(fPainter->ClippingRegion());
			fGraphicsCard->InvertRegion(r);
			return;
		}
	}

	int32 count = r.CountRects();
	for (int32 i = 0; i < count; i++)
		fPainter->FillRect(r.RectAt(i));
}


void
DrawingEngine::FillRegion(BRegion& r, const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	BRect clipped = fPainter->TransformAndClipRect(r.Frame());
	DrawTransaction transaction(this, clipped);
	if (!transaction.IsDirty())
		return;

	int32 count = r.CountRects();
	for (int32 i = 0; i < count; i++)
		fPainter->FillRect(r.RectAt(i), gradient);
}


void
DrawingEngine::DrawRoundRect(BRect r, float xrad, float yrad, bool filled)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	if (!filled)
		extend_by_stroke_width(r, fPainter->PenSize());
	BRect clipped = fPainter->TransformAndClipRect(r);

	clipped.left = floorf(clipped.left);
	clipped.top = floorf(clipped.top);
	clipped.right = ceilf(clipped.right);
	clipped.bottom = ceilf(clipped.bottom);

	DrawTransaction transaction(this, clipped);
	if (!transaction.IsDirty())
		return;

	if (filled)
		fPainter->FillRoundRect(r, xrad, yrad);
	else
		fPainter->StrokeRoundRect(r, xrad, yrad);
}


void
DrawingEngine::FillRoundRect(BRect r, float xrad, float yrad,
	const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	make_rect_valid(r);
	BRect clipped = fPainter->TransformAndClipRect(r);

	clipped.left = floorf(clipped.left);
	clipped.top = floorf(clipped.top);
	clipped.right = ceilf(clipped.right);
	clipped.bottom = ceilf(clipped.bottom);

	DrawTransaction transaction(this, clipped);
	if (!transaction.IsDirty())
		return;

	fPainter->FillRoundRect(r, xrad, yrad, gradient);
}


void
DrawingEngine::DrawShape(const BRect& bounds, int32 opCount,
	const uint32* opList, int32 ptCount, const BPoint* ptList, bool filled,
	const BPoint& viewToScreenOffset, float viewScale)
{
	ASSERT_PARALLEL_LOCKED();

// TODO: bounds probably does not take curves and arcs into account...
//	BRect clipped(bounds);
//	if (!filled)
//		extend_by_stroke_width(clipped, fPainter->PenSize());
//	clipped = fPainter->TransformAndClipRect(bounds);
//
//	clipped.left = floorf(clipped.left);
//	clipped.top = floorf(clipped.top);
//	clipped.right = ceilf(clipped.right);
//	clipped.bottom = ceilf(clipped.bottom);
//
//	DrawTransaction transaction(this, clipped);
//	if (!transaction.IsDirty())
//		return;
	DrawTransaction transaction(this);

	transaction.SetDirty(fPainter->DrawShape(opCount, opList, ptCount, ptList,
		filled, viewToScreenOffset, viewScale));
}


void
DrawingEngine::FillShape(const BRect& bounds, int32 opCount,
	const uint32* opList, int32 ptCount, const BPoint* ptList,
	const BGradient& gradient, const BPoint& viewToScreenOffset,
	float viewScale)
{
	ASSERT_PARALLEL_LOCKED();

// TODO: bounds probably does not take curves and arcs into account...
//	BRect clipped = fPainter->TransformAndClipRect(bounds);
//
//	clipped.left = floorf(clipped.left);
//	clipped.top = floorf(clipped.top);
//	clipped.right = ceilf(clipped.right);
//	clipped.bottom = ceilf(clipped.bottom);
//
//	DrawTransaction transaction(this, clipped);
//	if (!transaction.IsDirty())
//		return;
	DrawTransaction transaction(this);

	transaction.SetDirty(fPainter->FillShape(opCount, opList, ptCount, ptList,
		gradient, viewToScreenOffset, viewScale));
}


void
DrawingEngine::DrawTriangle(BPoint* pts, const BRect& bounds, bool filled)
{
	ASSERT_PARALLEL_LOCKED();

	BRect clipped(bounds);
	if (!filled)
		extend_by_stroke_width(clipped, fPainter->PenSize());
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(clipped));
	if (!transaction.IsDirty())
		return;

	if (filled)
		fPainter->FillTriangle(pts[0], pts[1], pts[2]);
	else
		fPainter->StrokeTriangle(pts[0], pts[1], pts[2]);
}


void
DrawingEngine::FillTriangle(BPoint* pts, const BRect& bounds,
	const BGradient& gradient)
{
	ASSERT_PARALLEL_LOCKED();

	DrawTransaction transaction(this, fPainter->TransformAndClipRect(bounds));
	if (!transaction.IsDirty())
		return;

	fPainter->FillTriangle(pts[0], pts[1], pts[2], gradient);
}


void
DrawingEngine::StrokeLine(const BPoint& start, const BPoint& end)
{
	ASSERT_PARALLEL_LOCKED();

	BRect touched(start, end);
	make_rect_valid(touched);
	extend_by_stroke_width(touched, fPainter->PenSize());
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(touched));
	if (!transaction.IsDirty())
		return;

	fPainter->StrokeLine(start, end);
}


void
DrawingEngine::StrokeLineArray(int32 numLines,
	const ViewLineArrayInfo *lineData)
{
	ASSERT_PARALLEL_LOCKED();

	if (!lineData || numLines <= 0)
		return;

	// figure out bounding box for line array
	const ViewLineArrayInfo* data = (const ViewLineArrayInfo*)&lineData[0];
	BRect touched(min_c(data->startPoint.x, data->endPoint.x),
		min_c(data->startPoint.y, data->endPoint.y),
		max_c(data->startPoint.x, data->endPoint.x),
		max_c(data->startPoint.y, data->endPoint.y));

	for (int32 i = 1; i < numLines; i++) {
		data = (const ViewLineArrayInfo*)&lineData[i];
		BRect box(min_c(data->startPoint.x, data->endPoint.x),
			min_c(data->startPoint.y, data->endPoint.y),
			max_c(data->startPoint.x, data->endPoint.x),
			max_c(data->startPoint.y, data->endPoint.y));
		touched = touched | box;
	}
	extend_by_stroke_width(touched, fPainter->PenSize());
	DrawTransaction transaction(this, fPainter->TransformAndClipRect(touched));
	if (!transaction.IsDirty())
		return;

	data = (const ViewLineArrayInfo*)&(lineData[0]);

	// store current graphics state, we mess with the
	// high color and pattern...
	rgb_color oldColor = fPainter->HighColor();
	struct pattern pattern = fPainter->Pattern();

	fPainter->SetHighColor(data->color);
	fPainter->SetPattern(B_SOLID_HIGH);
	fPainter->StrokeLine(data->startPoint, data->endPoint);

	for (int32 i = 1; i < numLines; i++) {
		data = (const ViewLineArrayInfo*)&(lineData[i]);
		fPainter->SetHighColor(data->color);
		fPainter->StrokeLine(data->startPoint, data->endPoint);
	}

	// restore correct drawing state highcolor and pattern
	fPainter->SetHighColor(oldColor);
	fPainter->SetPattern(pattern);
}


// #pragma mark -


BPoint
DrawingEngine::DrawString(const char* string, int32 length,
	const BPoint& pt, escapement_delta* delta)
{
	ASSERT_PARALLEL_LOCKED();

	BPoint penLocation = pt;

	// try a fast clipping path
	if (fPainter->ClippingRegion() != NULL
		&& fPainter->Font().Rotation() == 0.0f
		&& fPainter->IsIdentityTransform()) {
		float fontSize = fPainter->Font().Size();
		BRect clippingFrame = fPainter->ClippingRegion()->Frame();
		if (pt.x - fontSize > clippingFrame.right
			|| pt.y + fontSize < clippingFrame.top
			|| pt.y - fontSize > clippingFrame.bottom) {
			penLocation.x += StringWidth(string, length, delta);
			return penLocation;
		}
	}

	// use a FontCacheRefernece to speed up the second pass of
	// drawing the string
	FontCacheReference cacheReference;

//bigtime_t now = system_time();
// TODO: BoundingBox is quite slow!! Optimizing it will be beneficial.
// Cursiously, the DrawString after it is actually faster!?!
// TODO: make the availability of the hardware cursor part of the
// HW acceleration flags and skip all calculations for HideFloatingOverlays
// in case we don't have one.
// TODO: Watch out about penLocation and use Painter::PenLocation() when
// not using BoundindBox anymore.
	BRect b = fPainter->BoundingBox(string, length, pt, &penLocation, delta,
		&cacheReference);
	// stop here if we're supposed to render outside of the clipping
	DrawTransaction transaction(this, fPainter->ClipRect(b));
	if (transaction.IsDirty()) {
//printf("bounding box '%s': %lld µs\n", string, system_time() - now);

//now = system_time();
		fPainter->DrawString(string, length, pt, delta, &cacheReference);
//printf("drawing string: %lld µs\n", system_time() - now);
	}

	return penLocation;
}


BPoint
DrawingEngine::DrawString(const char* string, int32 length,
	const BPoint* offsets)
{
	ASSERT_PARALLEL_LOCKED();

	// use a FontCacheReference to speed up the second pass of
	// drawing the string
	FontCacheReference cacheReference;

	BPoint penLocation;
	BRect b = fPainter->BoundingBox(string, length, offsets, &penLocation,
		&cacheReference);
	// stop here if we're supposed to render outside of the clipping
	DrawTransaction transaction(this, fPainter->ClipRect(b));
	if (transaction.IsDirty()) {
//printf("bounding box '%s': %lld µs\n", string, system_time() - now);

//now = system_time();
		fPainter->DrawString(string, length, offsets, &cacheReference);
//printf("drawing string: %lld µs\n", system_time() - now);
	}

	return penLocation;
}


float
DrawingEngine::StringWidth(const char* string, int32 length,
	escapement_delta* delta)
{
	return fPainter->StringWidth(string, length, delta);
}


float
DrawingEngine::StringWidth(const char* string, int32 length,
	const ServerFont& font, escapement_delta* delta)
{
	return font.StringWidth(string, length, delta);
}


BPoint
DrawingEngine::DrawStringDry(const char* string, int32 length,
	const BPoint& pt, escapement_delta* delta)
{
	ASSERT_PARALLEL_LOCKED();

	BPoint penLocation = pt;

	// try a fast path first
	if (fPainter->Font().Rotation() == 0.0f
		&& fPainter->IsIdentityTransform()) {
		penLocation.x += StringWidth(string, length, delta);
		return penLocation;
	}

	fPainter->BoundingBox(string, length, pt, &penLocation, delta, NULL);

	return penLocation;
}


BPoint
DrawingEngine::DrawStringDry(const char* string, int32 length,
	const BPoint* offsets)
{
	ASSERT_PARALLEL_LOCKED();

	BPoint penLocation;
	fPainter->BoundingBox(string, length, offsets, &penLocation, NULL);

	return penLocation;
}


// #pragma mark -


ServerBitmap*
DrawingEngine::DumpToBitmap()
{
	return NULL;
}


status_t
DrawingEngine::ReadBitmap(ServerBitmap* bitmap, bool drawCursor, BRect bounds)
{
	ASSERT_EXCLUSIVE_LOCKED();

	RenderingBuffer* buffer = fGraphicsCard->FrontBuffer();
	if (buffer == NULL)
		return B_ERROR;

	BRect clip(0, 0, buffer->Width() - 1, buffer->Height() - 1);
	bounds = bounds & clip;
	AutoFloatingOverlaysHider _(fGraphicsCard, bounds);

	status_t result = bitmap->ImportBits(buffer->Bits(), buffer->BitsLength(),
		buffer->BytesPerRow(), buffer->ColorSpace(),
		bounds.LeftTop(), BPoint(0, 0),
		bounds.IntegerWidth() + 1, bounds.IntegerHeight() + 1);

	if (drawCursor) {
		ServerCursorReference cursorRef = fGraphicsCard->Cursor();
		ServerCursor* cursor = cursorRef.Get();
		if (!cursor)
			return result;
		int32 cursorWidth = cursor->Width();
		int32 cursorHeight = cursor->Height();

		BPoint cursorPosition = fGraphicsCard->CursorPosition();
		cursorPosition -= bounds.LeftTop() + cursor->GetHotSpot();

		BBitmap cursorArea(BRect(0, 0, cursorWidth - 1, cursorHeight - 1),
			B_BITMAP_NO_SERVER_LINK, B_RGBA32);

		cursorArea.ImportBits(bitmap->Bits(), bitmap->BitsLength(),
			bitmap->BytesPerRow(), bitmap->ColorSpace(),
			cursorPosition,	BPoint(0, 0),
			cursorArea.Bounds().Size());

		uint8* bits = (uint8*)cursorArea.Bits();
		uint8* cursorBits = (uint8*)cursor->Bits();
		for (int32 i = 0; i < cursorHeight; i++) {
			for (int32 j = 0; j < cursorWidth; j++) {
				uint8 alpha = 255 - cursorBits[3];
				bits[0] = ((bits[0] * alpha) >> 8) + cursorBits[0];
				bits[1] = ((bits[1] * alpha) >> 8) + cursorBits[1];
				bits[2] = ((bits[2] * alpha) >> 8) + cursorBits[2];
				cursorBits += 4;
				bits += 4;
			}
		}

		bitmap->ImportBits(cursorArea.Bits(), cursorArea.BitsLength(),
			cursorArea.BytesPerRow(), cursorArea.ColorSpace(),
			BPoint(0, 0), cursorPosition,
			cursorWidth, cursorHeight);
	}

	return result;
}


// #pragma mark -


BRect
DrawingEngine::CopyRect(BRect src, int32 xOffset, int32 yOffset) const
{
	// TODO: assumes drawing buffer is 32 bits (which it currently always is)
	BRect dst;
	RenderingBuffer* buffer = fGraphicsCard->DrawingBuffer();
	if (buffer) {
		BRect clip(0, 0, buffer->Width() - 1, buffer->Height() - 1);

		dst = src;
		dst.OffsetBy(xOffset, yOffset);

		if (clip.Intersects(src) && clip.Intersects(dst)) {
			uint32 bytesPerRow = buffer->BytesPerRow();
			uint8* bits = (uint8*)buffer->Bits();

			// clip source rect
			src = src & clip;
			// clip dest rect
			dst = dst & clip;
			// move dest back over source and clip source to dest
			dst.OffsetBy(-xOffset, -yOffset);
			src = src & dst;

			// calc offset in buffer
			bits += (ssize_t)src.left * 4 + (ssize_t)src.top * bytesPerRow;

			uint32 width = src.IntegerWidth() + 1;
			uint32 height = src.IntegerHeight() + 1;

			_CopyRect(buffer->IsGraphicsMemory(), bits, width, height, bytesPerRow,
				xOffset, yOffset);

			// offset dest again, because it is return value
			dst.OffsetBy(xOffset, yOffset);
		}
	}
	return dst;
}


void
DrawingEngine::SetRendererOffset(int32 offsetX, int32 offsetY)
{
	fPainter->SetRendererOffset(offsetX, offsetY);
}


void
DrawingEngine::_CopyRect(bool isGraphicsMemory, uint8* src, uint32 width, uint32 height,
	uint32 bytesPerRow, int32 xOffset, int32 yOffset) const
{
	// TODO: assumes drawing buffer is 32 bits (which it currently always is)
	int32 yIncrement;
	const bool needMemmove = (yOffset == 0 && xOffset > 0 && uint32(xOffset) <= width);

	if (yOffset > 0) {
		// copy from bottom to top
		yIncrement = -bytesPerRow;
		src += (height - 1) * bytesPerRow;
	} else {
		// copy from top to bottom
		yIncrement = bytesPerRow;
	}

	uint8* dst = src + (ssize_t)yOffset * bytesPerRow + (ssize_t)xOffset * 4;

	if (!needMemmove) {
		if (!isGraphicsMemory) {
			// NOTE: this (instead of the two pass copy below) might
			// speed up QEMU -> ?!? (would depend on how it emulates
			// the PCI bus...)
			for (uint32 y = 0; y < height; y++) {
				memcpy(dst, src, width * 4);
				src += yIncrement;
				dst += yIncrement;
			}
		} else {
			uint8 tmpBuffer[width * 4];
			for (uint32 y = 0; y < height; y++) {
				// NOTE: read into temporary scanline buffer,
				// NOTE: **don't read and write over the PCI bus
				// at the same time**
				memcpy(tmpBuffer, src, width * 4);
				// write back temporary scanline buffer
				memcpy(dst, tmpBuffer, width * 4);
				src += yIncrement;
				dst += yIncrement;
			}
		}
	} else {
		for (uint32 y = 0; y < height; y++) {
			memmove(dst, src, width * 4);
			src += yIncrement;
			dst += yIncrement;
		}
	}
}
