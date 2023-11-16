/*
 * Copyright 2007, Haiku. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Michael Pfeiffer
 */

#include "PictureTestCases.h"

#include <GradientLinear.h>
#include <GradientRadial.h>
#include <GradientRadialFocus.h>
#include <GradientDiamond.h>
#include <GradientConic.h>

#include <stdio.h>

static const rgb_color kBlack = {0, 0, 0};
static const rgb_color kWhite = {255, 255, 255};
static const rgb_color kRed = {255, 0, 0};
static const rgb_color kGreen = {0, 255, 0};
static const rgb_color kBlue = {0, 0, 255};

static BPoint centerPoint(BRect rect)
{
	int x = (int)(rect.left + rect.IntegerWidth() / 2);
	int y = (int)(rect.top + rect.IntegerHeight() / 2);
	return BPoint(x, y);
}

static void testNoOp(BView *view, BRect frame)
{
	// no op
}

static void testDrawChar(BView *view, BRect frame)
{
	view->MovePenTo(frame.left, frame.bottom - 5);
	view->DrawChar('A');
	
	view->DrawChar('B', BPoint(frame.left + 20, frame.bottom - 5));
}

static void testDrawString(BView *view, BRect frame)
{
	BFont font;
	view->GetFont(&font);
	font_height height;
	font.GetHeight(&height);
	float baseline = frame.bottom - height.descent;
	// draw base line
	view->SetHighColor(kGreen);
	view->StrokeLine(BPoint(frame.left, baseline - 1), BPoint(frame.right, baseline -1));

	view->SetHighColor(kBlack);
	view->DrawString("Haiku [ÖÜÄöüä]", BPoint(frame.left, baseline));
}

static void testDrawStringWithLength(BView *view, BRect frame)
{
	BFont font;
	view->GetFont(&font);
	font_height height;
	font.GetHeight(&height);
	float baseline = frame.bottom - height.descent;
	// draw base line
	view->SetHighColor(kGreen);
	view->StrokeLine(BPoint(frame.left, baseline - 1), BPoint(frame.right, baseline -1));

	view->SetHighColor(kBlack);
	view->DrawString("Haiku [ÖÜÄöüä]", 13, BPoint(frame.left, baseline));
}


static void testDrawStringWithOffsets(BView* view, BRect frame)
{
	BFont font;
	view->GetFont(&font);
	font_height height;
	font.GetHeight(&height);
	float baseline = frame.bottom - height.descent;
	// draw base line
	view->SetHighColor(kGreen);
	view->StrokeLine(BPoint(frame.left, baseline - 1), BPoint(frame.right, baseline -1));

	view->SetHighColor(kBlack);
	BPoint point(frame.left, baseline);
	BPoint pointArray[] = {
		point,
		point,
		point,
		point,
		point
	};
	
	for (size_t i = 1; i < (sizeof(pointArray) / sizeof(pointArray[0])); i++)
		pointArray[i] = pointArray[i - 1] + BPoint(10, 0);

	view->DrawString("Haiku", pointArray, sizeof(pointArray) / sizeof(pointArray[0]));
}


static void testDrawStringWithoutPosition(BView* view, BRect frame)
{
	BFont font;
	view->GetFont(&font);
	font_height height;
	font.GetHeight(&height);
	float baseline = frame.bottom - height.descent;
	// draw base line
	view->SetHighColor(kGreen);
	view->StrokeLine(BPoint(frame.left, baseline - 1), BPoint(frame.right, baseline -1));

	view->SetHighColor(kBlack);
	view->MovePenTo(BPoint(frame.left, baseline));
	view->DrawString("H");
	view->DrawString("a");
	view->DrawString("i");
	view->DrawString("k");
	view->DrawString("u");
}


static void testFillArc(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->FillArc(frame, 45, 180);
}

static void testStrokeArc(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->StrokeArc(frame, 45, 180);
}

static void testFillBezier(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint points[4];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(frame.left, frame.top);
	points[1] = BPoint(frame.left, frame.top);
	points[3] = BPoint(frame.right, frame.top);
	view->FillBezier(points);
}

static void testStrokeBezier(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint points[4];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(frame.left, frame.top);
	points[1] = BPoint(frame.left, frame.top);
	points[3] = BPoint(frame.right, frame.top);
	view->StrokeBezier(points);
}

static void testFillEllipse(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->FillEllipse(frame);

	view->SetHighColor(kRed);
	float r = frame.Width() / 3;
	float s = frame.Height() / 4;
	view->FillEllipse(centerPoint(frame), r, s);
}

static void testStrokeEllipse(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->StrokeEllipse(frame);

	view->SetHighColor(kRed);
	float r = frame.Width() / 3;
	float s = frame.Height() / 4;
	view->StrokeEllipse(centerPoint(frame), r, s);
}

static void testFillPolygon(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);

	BPoint points[4];
	points[0] = BPoint(frame.left, frame.top);
	points[1] = BPoint(frame.right, frame.bottom);
	points[2] = BPoint(frame.right, frame.top);
	points[3] = BPoint(frame.left, frame.bottom);

	view->FillPolygon(points, 4);
}

static void testStrokePolygon(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);

	BPoint points[4];
	points[0] = BPoint(frame.left, frame.top);
	points[1] = BPoint(frame.right, frame.bottom);
	points[2] = BPoint(frame.right, frame.top);
	points[3] = BPoint(frame.left, frame.bottom);

	view->StrokePolygon(points, 4);
}

static void testFillRect(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->FillRect(frame);
}

static void testFillRectGradientLinear(BView* view, BRect frame)
{
	BGradientLinear gradient(0, 0, frame.right, frame.bottom);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRect(frame, gradient);
}

static void testFillRectGradientRadial(BView* view, BRect frame)
{
	BGradientRadial gradient(10, 10, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRect(frame, gradient);
}

static void testFillRectGradientRadialFocus(BView* view, BRect frame)
{
	BGradientRadialFocus gradient(0, 0, 10, 10, 5);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRect(frame, gradient);
}

static void testFillRectGradientDiamond(BView* view, BRect frame)
{
	BGradientDiamond gradient(0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRect(frame, gradient);
}

static void testFillRectGradientConic(BView* view, BRect frame)
{
	BGradientConic gradient(0, 0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRect(frame, gradient);
}

static void testStrokeRect(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->StrokeRect(frame);
}

static void testFillRegion(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BRegion region(frame);
	frame.InsetBy(10, 10);
	region.Exclude(frame);
	view->FillRegion(&region);
}

static void testFillRegionGradientLinear(BView* view, BRect frame)
{
	BGradientLinear gradient(0, 0, frame.right, frame.bottom);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BRegion region(frame);
	frame.InsetBy(10, 10);
	region.Exclude(frame);
	view->FillRegion(&region, gradient);
}

static void testFillRegionGradientRadial(BView* view, BRect frame)
{
	BGradientRadial gradient(10, 10, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BRegion region(frame);
	frame.InsetBy(10, 10);
	region.Exclude(frame);
	view->FillRegion(&region, gradient);
}

static void testFillRegionGradientRadialFocus(BView* view, BRect frame)
{
	BGradientRadialFocus gradient(0, 0, 10, 10, 5);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BRegion region(frame);
	frame.InsetBy(10, 10);
	region.Exclude(frame);
	view->FillRegion(&region, gradient);
}

static void testFillRegionGradientDiamond(BView* view, BRect frame)
{
	BGradientDiamond gradient(0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BRegion region(frame);
	frame.InsetBy(10, 10);
	region.Exclude(frame);
	view->FillRegion(&region, gradient);
}

static void testFillRegionGradientConic(BView* view, BRect frame)
{
	BGradientConic gradient(0, 0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BRegion region(frame);
	frame.InsetBy(10, 10);
	region.Exclude(frame);
	view->FillRegion(&region, gradient);
}

static void testFillRoundRect(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->FillRoundRect(frame, 5, 3);
}

static void testFillRoundRectGradientLinear(BView* view, BRect frame)
{
	BGradientLinear gradient(0, 0, frame.right, frame.bottom);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRoundRect(frame, 5, 3, gradient);
}

static void testFillRoundRectGradientRadial(BView* view, BRect frame)
{
	BGradientRadial gradient(10, 10, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRoundRect(frame, 5, 3, gradient);
}

static void testFillRoundRectGradientRadialFocus(BView* view, BRect frame)
{
	BGradientRadialFocus gradient(0, 0, 10, 10, 5);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	view->FillRoundRect(frame, 5, 3, gradient);
}

static void testFillRoundRectGradientDiamond(BView* view, BRect frame)
{
	BGradientDiamond gradient(0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRoundRect(frame, 5, 3, gradient);
}

static void testFillRoundRectGradientConic(BView* view, BRect frame)
{
	BGradientConic gradient(0, 0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	view->FillRoundRect(frame, 5, 3, gradient);
}

static void testStrokeRoundRect(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->StrokeRoundRect(frame, 5, 3);
}

static void testFillTriangle(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->FillTriangle(points[0], points[1], points[2]); 
}

static void testFillTriangleGradientLinear(BView* view, BRect frame)
{
	BGradientLinear gradient(0, 0, frame.right, frame.bottom);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->FillTriangle(points[0], points[1], points[2], gradient);
}

static void testFillTriangleGradientRadial(BView* view, BRect frame)
{
	BGradientRadial gradient(10, 10, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->FillTriangle(points[0], points[1], points[2], gradient);
}

static void testFillTriangleGradientRadialFocus(BView* view, BRect frame)
{
	BGradientRadialFocus gradient(0, 0, 10, 10, 5);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->FillTriangle(points[0], points[1], points[2], gradient);
}

static void testFillTriangleGradientDiamond(BView* view, BRect frame)
{
	BGradientDiamond gradient(0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->FillTriangle(points[0], points[1], points[2], gradient);
}

static void testFillTriangleGradientConic(BView* view, BRect frame)
{
	BGradientConic gradient(0, 0, 10);
	gradient.AddColor(kRed, 0);
	gradient.AddColor(kBlue, 255);
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->FillTriangle(points[0], points[1], points[2], gradient);
}

static void testStrokeTriangle(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint points[3];
	points[0] = BPoint(frame.left, frame.bottom);
	points[1] = BPoint(centerPoint(frame).x, frame.top);
	points[2] = BPoint(frame.right, frame.bottom);
	view->StrokeTriangle(points[0], points[1], points[2]); 
}

static void testStrokeLine(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(frame.right, frame.top));
	
	frame.top += 2;
	frame.bottom -= 2;
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(frame.right, frame.bottom));
	
	frame.bottom += 2;;
	frame.top = frame.bottom;
	view->StrokeLine(BPoint(frame.right, frame.top), BPoint(frame.left, frame.top));
}

static void testFillShape(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BShape shape;
	shape.MoveTo(BPoint(frame.left, frame.bottom));
	shape.LineTo(BPoint(frame.right, frame.top));
	shape.LineTo(BPoint(frame.left, frame.top));
	shape.LineTo(BPoint(frame.right, frame.bottom));
	view->FillShape(&shape);
}

static void testStrokeShape(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BShape shape;
	shape.MoveTo(BPoint(frame.left, frame.bottom));
	shape.LineTo(BPoint(frame.right, frame.top));
	shape.LineTo(BPoint(frame.left, frame.top));
	shape.LineTo(BPoint(frame.right, frame.bottom));
	view->StrokeShape(&shape);
}

static void testRecordPicture(BView *view, BRect frame)
{
	BPicture *picture = new BPicture();
	view->BeginPicture(picture);
	view->FillRect(frame);
	view->EndPicture();
	delete picture;
}

static void testRecordAndPlayPicture(BView *view, BRect frame)
{
	BPicture *picture = new BPicture();
	view->BeginPicture(picture);
	frame.InsetBy(2, 2);
	view->FillRect(frame);
	view->EndPicture();
	view->DrawPicture(picture);
	delete picture;
}

static void testRecordAndPlayPictureWithOffset(BView *view, BRect frame)
{
	BPicture *picture = new BPicture();
	view->BeginPicture(picture);
	frame.InsetBy(frame.Width() / 4, frame.Height() / 4);
	frame.OffsetTo(0, 0);
	view->FillRect(frame);
	view->EndPicture();
	
	view->DrawPicture(picture, BPoint(10, 10));
	// color of picture should not change
	view->SetLowColor(kGreen);
	view->SetLowColor(kRed);
	view->DrawPicture(picture, BPoint(0, 0));
	delete picture;
}

static void testAppendToPicture(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->BeginPicture(new BPicture());
	view->FillRect(frame);
	BPicture* picture = view->EndPicture();
	if (picture == NULL)
		return;
	
	frame.InsetBy(2, 2);
	view->AppendToPicture(picture);
	view->SetHighColor(kRed);
	view->FillRect(frame);
	if (view->EndPicture() != picture)
		return;
	
	view->DrawPicture(picture);
	delete picture;
}

static void testDrawScaledPicture(BView* view, BRect frame)
{
	view->BeginPicture(new BPicture());
	view->FillRect(BRect(0, 0, 15, 15));
	BPicture* picture = view->EndPicture();

	// first unscaled at left, top
	view->DrawPicture(picture, BPoint(2, 2));
	
	// draw scaled at middle top
	view->SetScale(0.5);
	// the drawing offset must be scaled too!
	view->DrawPicture(picture, BPoint(frame.Width(), 4));
	
	delete picture;
}

static void testLineArray(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->BeginLineArray(3);
	view->AddLine(BPoint(frame.left, frame.top), BPoint(frame.right, frame.top), kBlack);
	
	frame.top += 2;
	frame.bottom -= 2;
	view->AddLine(BPoint(frame.left, frame.top), BPoint(frame.right, frame.bottom), kRed);
	
	frame.bottom += 2;;
	frame.top = frame.bottom;
	view->AddLine(BPoint(frame.right, frame.top), BPoint(frame.left, frame.top), kGreen);	
	
	view->EndLineArray();
}

static void testInvertRect(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->InvertRect(frame);
}

static void testInvertRectSetDrawingMode(BView *view, BRect frame)
{
	view->SetDrawingMode(B_OP_ALPHA);
	view->SetHighColor(128, 128, 128, 128);
	frame.InsetBy(2, 2);
	view->InvertRect(frame);
	frame.InsetBy(10, 10);
	view->FillRect(frame, B_SOLID_HIGH);
}

static bool isBorder(int32 x, int32 y, int32 width, int32 height) {
	return x == 0 || y == 0 || x == width - 1 || y == height - 1;
}

static void fillBitmap(BBitmap &bitmap) {
	int32 height = bitmap.Bounds().IntegerHeight()+1;
	int32 width = bitmap.Bounds().IntegerWidth()+1;
	for (int32 y = 0; y < height; y ++) {
		for (int32 x = 0; x < width; x ++) {
			char *pixel = (char*)bitmap.Bits();
			pixel += bitmap.BytesPerRow() * y + 4 * x;
			if (isBorder(x, y, width, height)) {
				// fill with green
				pixel[0] = 255;
				pixel[1] = 0;
				pixel[2] = 255;
				pixel[3] = 0;
			} else  {
				// fill with blue
				pixel[0] = 255;
				pixel[1] = 0;
				pixel[2] = 0;
				pixel[3] = 255;
			}	
		}
	}
}

static void testDrawBitmap(BView *view, BRect frame) {
	BBitmap bitmap(frame, B_RGBA32);
	fillBitmap(bitmap);
	view->DrawBitmap(&bitmap, BPoint(0, 0));
}

static void testDrawBitmapAtPoint(BView *view, BRect frame) {
	frame.InsetBy(2, 2);
	
	BRect bounds(frame);
	bounds.OffsetTo(0, 0);
	bounds.right /= 2;
	bounds.bottom /= 2;
	
	BBitmap bitmap(bounds, B_RGBA32);
	fillBitmap(bitmap);
	view->DrawBitmap(&bitmap, centerPoint(frame));
}

static void testDrawBitmapAtRect(BView *view, BRect frame) {
	BRect bounds(frame);
	BBitmap bitmap(bounds, B_RGBA32);
	fillBitmap(bitmap);
	frame.InsetBy(2, 2);
	view->DrawBitmap(&bitmap, frame);
}

static void testDrawLargeBitmap(BView *view, BRect frame) {
	BRect bounds(frame);
	bounds.OffsetTo(0, 0);
	bounds.right *= 4;
	bounds.bottom *= 4;
	BBitmap bitmap(bounds, B_RGBA32);
	fillBitmap(bitmap);
	frame.InsetBy(2, 2);
	view->DrawBitmap(&bitmap, frame);
}

static void testConstrainClippingRegion(BView *view, BRect frame) 
{
	frame.InsetBy(2, 2);
	// draw background
	view->SetHighColor(kRed);
	view->FillRect(frame);
	
	frame.InsetBy(1, 1);
	BRegion region(frame);
	BRect r(frame);
	r.InsetBy(r.IntegerWidth() / 4, r.IntegerHeight() / 4);
	region.Exclude(r);
	view->ConstrainClippingRegion(&region);
	
	frame.InsetBy(-1, -1);
	view->SetHighColor(kBlack);
	view->FillRect(frame);
	// a filled black rectangle with a red one pixel border
	// and inside a red rectangle should be drawn.
}

static void testClipToPicture(BView *view, BRect frame) 
{
	frame.InsetBy(2, 2);
	view->BeginPicture(new BPicture());
	view->FillEllipse(frame);
	BPicture *picture = view->EndPicture();
	if (picture == NULL)
		return;
	
	view->ClipToPicture(picture);
	delete picture;
	
	view->FillRect(frame);
	// black ellipse should be drawn
}

static void testClipToInversePicture(BView *view, BRect frame) 
{
	frame.InsetBy(2, 2);
	
	view->BeginPicture(new BPicture());
	view->FillEllipse(frame);
	BPicture *picture = view->EndPicture();
	if (picture == NULL)
		return;
	
	view->ClipToInversePicture(picture);
	delete picture;
	
	view->FillRect(frame);
	// white ellipse inside a black rectangle
}

static void testSetPenSize(BView *view, BRect frame)
{
	frame.InsetBy(8, 2);
	float x = centerPoint(frame).x;
	
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(frame.right, frame.top));
	
	frame.OffsetBy(0, 5);
	view->SetPenSize(1);
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(x, frame.top));
	view->SetPenSize(0);
	view->StrokeLine(BPoint(x+1, frame.top), BPoint(frame.right, frame.top));

	frame.OffsetBy(0, 5);
	view->SetPenSize(1);
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(x, frame.top));
	view->SetPenSize(2);
	view->StrokeLine(BPoint(x+1, frame.top), BPoint(frame.right, frame.top));

	frame.OffsetBy(0, 5);
	view->SetPenSize(1);
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(x, frame.top));
	view->SetPenSize(3);
	view->StrokeLine(BPoint(x+1, frame.top), BPoint(frame.right, frame.top));

	frame.OffsetBy(0, 5);
	view->SetPenSize(1);
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(x, frame.top));
	view->SetPenSize(4);
	view->StrokeLine(BPoint(x+1, frame.top), BPoint(frame.right, frame.top));
}

static void testSetPenSize2(BView *view, BRect frame)
{
	// test if pen size is scaled too
	frame.InsetBy(2, 2);
	frame.OffsetBy(0, 5);
	view->SetPenSize(4);
	view->StrokeLine(BPoint(frame.left, frame.top), BPoint(frame.right, frame.top));
	view->SetScale(0.5);
	view->StrokeLine(BPoint(frame.left + 2, frame.bottom), BPoint(frame.right + 2, frame.bottom));	
	
	// black line from left to right, 4 pixel size
	// below black line with half the length of the first one
	// and 2 pixel size
}

static void testPattern(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	int x = frame.IntegerWidth() / 3;
	frame.right = frame.left + x - 2;
		// -2 for an empty pixel row between 
		// filled rectangles

	view->SetLowColor(kGreen);
	view->SetHighColor(kRed);
	
	view->FillRect(frame, B_SOLID_HIGH);
	
	frame.OffsetBy(x, 0);
	view->FillRect(frame, B_MIXED_COLORS);
	
	frame.OffsetBy(x, 0);
	view->FillRect(frame, B_SOLID_LOW);
}

static void testSetOrigin(BView *view, BRect frame)
{
	BPoint origin = view->Origin();
	BPoint center = centerPoint(frame);
	view->SetOrigin(center);

	BRect r(0, 0, center.x, center.y);
	view->SetHighColor(kBlue);
	view->FillRect(r);
	
	view->SetOrigin(origin);
	view->SetHighColor(kRed);
	view->FillRect(r);
	
	// red rectangle in left, top corner
	// blue rectangle in right, bottom corner
	// the red rectangle overwrites the
	// top, left pixel of the blue rectangle
}

static void testSetOrigin2(BView *view, BRect frame)
{
	BPoint center = centerPoint(frame);
	BRect r(0, 0, center.x, center.y);
	view->SetOrigin(center);
	view->PushState();
		view->SetOrigin(BPoint(-center.x, 0));
		view->FillRect(r);
	view->PopState();	
	// black rectangle in left, bottom corner
}

static void testSetScale(BView *view, BRect frame)
{
	view->SetScale(0.5);
	view->FillRect(frame);
	// black rectangle in left, top corner
}

static void testSetScale2(BView *view, BRect frame)
{
	view->SetScale(0.5);
	view->PushState();
		view->SetScale(0.5);
		view->FillRect(frame);
	view->PopState();	
	// black rectangle in left, top corner
	// with half the size of the rectangle
	// from test testSetScaling
}

static void testSetScale3(BView *view, BRect frame)
{
	view->SetScale(0.5);
	view->PushState();
		// if the second scale value differs slightly
		// the bug under BeOS R5 in testSetScale2 
		// does not occur
		view->SetScale(0.5000001);
		view->FillRect(frame);
	view->PopState();	
	// black rectangle in left, top corner
	// with half the size of the rectangle
	// from test testSetScaling
}

static void testSetOriginAndScale(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint center = centerPoint(frame);
	
	BRect r(0, 0, frame.IntegerWidth() / 2, frame.IntegerHeight() / 2);
	view->SetOrigin(center);
	view->FillRect(r);
	
	view->SetScale(0.5);
	view->SetHighColor(kRed);
	view->FillRect(r);
}

static void testSetOriginAndScale2(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint center = centerPoint(frame);
	
	BRect r(0, 0, frame.IntegerWidth() / 2, frame.IntegerHeight() / 2);
	view->SetOrigin(center);
	view->FillRect(r);
	
	view->SetScale(0.5);
	view->SetHighColor(kRed);
	view->FillRect(r);
	
	view->SetOrigin(0, 0);
	view->SetHighColor(kGreen);
	view->FillRect(r);
}

static void testSetOriginAndScale3(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint center = centerPoint(frame);
	
	BRect r(0, 0, frame.IntegerWidth() / 2, frame.IntegerHeight() / 2);
	view->SetOrigin(center);
	view->FillRect(r);
	
	view->SetScale(0.5);
	view->SetHighColor(kRed);
	view->FillRect(r);
	
	view->SetScale(0.25);
	view->SetHighColor(kGreen);
	view->FillRect(r);
}

static void testSetOriginAndScale4(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint center = centerPoint(frame);
	
	BRect r(0, 0, frame.IntegerWidth() / 2, frame.IntegerHeight() / 2);
	view->SetOrigin(center);
	view->FillRect(r);
	
	view->SetScale(0.5);
	view->SetHighColor(kRed);
	view->FillRect(r);
	
	view->PushState();
		// 
		view->SetOrigin(center.x+1, center.y);
			// +1 to work around BeOS bug
			// where setting the origin has no
			// effect if it is the same as
			// the previous value althou
			// it is from the "outer" coordinate
			// system
		view->SetHighColor(kGreen);
		view->FillRect(r);
	view->PopState();
}

static void testSetOriginAndScale5(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	BPoint center = centerPoint(frame);
	
	BRect r(0, 0, frame.IntegerWidth() / 2, frame.IntegerHeight() / 2);
	view->SetOrigin(center);
	view->FillRect(r);
	
	view->SetScale(0.5);
	view->SetHighColor(kRed);
	view->FillRect(r);
	
	view->PushState();
	view->SetScale(0.75);
	view->SetHighColor(kGreen);
	view->FillRect(r);
	view->PopState();
}

static void testSetFontSize(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	int size = frame.IntegerHeight() / 3;
	
	frame.OffsetBy(0, size);
	view->MovePenTo(BPoint(frame.left, frame.top));
	view->SetFontSize(size);
	view->DrawString("Haiku");
	
	size *= 2;	
	frame.OffsetBy(0, size);
	view->MovePenTo(BPoint(frame.left, frame.top));
	view->SetFontSize(size);
	view->DrawString("Haiku");	
}

static void testSetFontFamilyAndStyle(BView *view, BRect frame)
{
	view->DrawString("This is a test", BPoint(2, 6));
	
	BFont font;
	view->GetFont(&font);
	
	int32 families = count_font_families();
	font_family familyName;
	get_font_family(families - 1, &familyName);
	
	int32 styles = count_font_styles(familyName);
	font_style styleName;
	get_font_style(familyName, styles - 1, &styleName);
	font.SetFamilyAndStyle(familyName, styleName);
	view->SetFont(&font);
	view->DrawString( "This is a test", BPoint(2, 19));
}

static void testSetDrawingMode(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->StrokeLine(frame.LeftTop(), frame.RightBottom());
	view->StrokeLine(frame.LeftBottom(), frame.RightTop());
	view->SetDrawingMode(B_OP_ALPHA);
	rgb_color color = kRed;
	color.alpha = 127;	
	view->SetHighColor(color);
	view->FillRect(frame, B_SOLID_HIGH);
}

static void testPushPopState(BView *view, BRect frame)
{
	frame.InsetBy(2, 2);
	view->SetHighColor(kGreen);
	view->PushState();
	view->SetHighColor(kRed);
	view->PopState();
	
	view->FillRect(frame, B_SOLID_HIGH);
}

static void testFontRotation(BView* view, BRect frame)
{
	BFont font;
	view->GetFont(&font);
	
	font.SetRotation(90);
	view->SetFont(&font, B_FONT_ROTATION);
	view->DrawString("This is a test!", BPoint(frame.Width() / 2, frame.bottom - 3));
	
	view->GetFont(&font);
	if (font.Rotation() != 90.0)
		fprintf(stderr, "Error: Rotation is %f but should be 90.0\n", font.Rotation());
}


static void testClipToRect(BView* view, BRect frame)
{
	BRect clipped = frame;
	clipped.InsetBy(5, 5);
	
	view->ClipToRect(clipped);
	
	view->FillRect(frame);
}


static void testClipToInverseRect(BView* view, BRect frame)
{
	BRect clipped = frame;
	clipped.InsetBy(5, 5);
	
	view->ClipToInverseRect(clipped);
	
	view->FillRect(frame);
}


static void testClipToShape(BView* view, BRect frame)
{
	frame.InsetBy(2, 2);
	BShape shape;
	shape.MoveTo(BPoint(frame.left, frame.bottom));
	shape.LineTo(BPoint(frame.right, frame.top));
	shape.LineTo(BPoint(frame.left, frame.top));
	shape.LineTo(BPoint(frame.right, frame.bottom));
	view->ClipToShape(&shape);
	
	view->FillRect(frame);
}


static void testClipToInverseShape(BView* view, BRect frame)
{
	frame.InsetBy(2, 2);
	BShape shape;
	shape.MoveTo(BPoint(frame.left, frame.bottom));
	shape.LineTo(BPoint(frame.right, frame.top));
	shape.LineTo(BPoint(frame.left, frame.top));
	shape.LineTo(BPoint(frame.right, frame.bottom));
	view->ClipToInverseShape(&shape);
	
	view->FillRect(frame);
}


// TODO
// - blending mode
// - line mode
// - push/pop state
// - move pen
// - set font


TestCase gTestCases[] = {
	{ "Test No Operation", testNoOp },
	{ "Test DrawChar", testDrawChar },
	{ "Test Draw String", testDrawString },
	{ "Test Draw String With Length", testDrawStringWithLength },
	{ "Test Draw String With Offsets", testDrawStringWithOffsets },
	{ "Test Draw String Without Position", testDrawStringWithoutPosition },
	{ "Test FillArc", testFillArc },
	{ "Test StrokeArc", testStrokeArc },
	// testFillBezier fails under BeOS because the
	// direct draw version is not correct
	{ "Test FillBezier", testFillBezier },
	{ "Test StrokeBezier", testStrokeBezier },
	{ "Test FillEllipse", testFillEllipse },
	{ "Test StrokeEllipse", testStrokeEllipse },
	{ "Test FillPolygon", testFillPolygon },
	{ "Test StrokePolygon", testStrokePolygon },
	{ "Test FillRect", testFillRect },
	{ "Test FillRectGradientLinear", testFillRectGradientLinear },
	{ "Test FillRectGradientRadial", testFillRectGradientRadial },
	{ "Test FillRectGradientRadialFocus", testFillRectGradientRadialFocus },
	{ "Test FillRectGradientDiamond", testFillRectGradientDiamond },
	{ "Test FillRectGradientConic", testFillRectGradientConic },
	{ "Test StrokeRect", testStrokeRect },
	{ "Test FillRegion", testFillRegion },
	{ "Test FillRegionGradientLinear", testFillRegionGradientLinear },
	{ "Test FillRegionGradientRadial", testFillRegionGradientRadial },
	{ "Test FillRegionGradientRadialFocus", testFillRegionGradientRadialFocus },
	{ "Test FillRegionGradientDiamond", testFillRegionGradientDiamond },
	{ "Test FillRegionGradientConic", testFillRegionGradientConic },
	{ "Test FillRoundRect", testFillRoundRect },
	{ "Test FillRoundRectGradientLinear", testFillRoundRectGradientLinear },
	{ "Test FillRoundRectGradientRadial", testFillRoundRectGradientRadial },
	{ "Test FillRoundRectGradientRadialFocus", testFillRoundRectGradientRadialFocus },
	{ "Test FillRoundRectGradientDiamond", testFillRoundRectGradientDiamond },
	{ "Test FillRoundRectGradientConic", testFillRoundRectGradientConic },
	{ "Test StrokeRoundRect", testStrokeRoundRect },
	{ "Test FillTriangle", testFillTriangle },
	{ "Test FillTriangleGradientLinear", testFillTriangleGradientLinear },
	{ "Test FillTriangleGradientRadial", testFillTriangleGradientRadial },
	{ "Test FillTriangleGradientRadialFocus", testFillTriangleGradientRadialFocus },
	{ "Test FillTriangleGradientDiamond", testFillTriangleGradientDiamond },
	{ "Test FillTriangleGradientConic", testFillTriangleGradientConic },
	{ "Test StrokeTriangle", testStrokeTriangle },
	{ "Test StrokeLine", testStrokeLine },
	{ "Test FillShape", testFillShape },
	{ "Test StrokeShape", testStrokeShape },
	{ "Test Record Picture", testRecordPicture },
	{ "Test Record And Play Picture", testRecordAndPlayPicture },
	{ "Test Record And Play Picture With Offset", testRecordAndPlayPictureWithOffset },
	{ "Test AppendToPicture", testAppendToPicture },
	{ "Test Draw Scaled Picture", testDrawScaledPicture },
	{ "Test LineArray", testLineArray },
	{ "Test InvertRect", testInvertRect },
	{ "Test InvertRectSetDrawingMode", testInvertRectSetDrawingMode },
	{ "Test DrawBitmap", testDrawBitmap },
	{ "Test DrawBitmapAtPoint", testDrawBitmapAtPoint },
	{ "Test DrawBitmapAtRect", testDrawBitmapAtRect },
	{ "Test DrawLargeBitmap", testDrawLargeBitmap },
	{ "Test ConstrainClippingRegion", testConstrainClippingRegion }, 
	{ "Test ClipToPicture", testClipToPicture },
	{ "Test ClipToInversePicture", testClipToInversePicture },
	{ "Test ClipToRect", testClipToRect },
	{ "Test ClipToInverseRect", testClipToInverseRect },
	{ "Test ClipToShape", testClipToShape },
	{ "Test ClipToInverseShape", testClipToInverseShape },
	{ "Test SetPenSize", testSetPenSize },
	{ "Test SetPenSize2", testSetPenSize2 },
	{ "Test Pattern", testPattern },
	{ "Test SetOrigin", testSetOrigin },
	{ "Test SetOrigin2", testSetOrigin2 },
	{ "Test SetScale", testSetScale },
	// testSetScale2 fails under BeOS. The picture versions of the
	// rectangle are twice as large as the direct draw version
	{ "Test SetScale2", testSetScale2 },
	{ "Test SetScale3", testSetScale3 },
	{ "Test SetOriginAndScale", testSetOriginAndScale },
	{ "Test SetOriginAndScale2", testSetOriginAndScale2 },
	{ "Test SetOriginAndScale3", testSetOriginAndScale3 },
	{ "Test SetOriginAndScale4", testSetOriginAndScale4 },
	{ "Test SetOriginAndScale5", testSetOriginAndScale5 },
	{ "Test SetFontSize", testSetFontSize },
	{ "Test SetFontFamilyAndStyle", testSetFontFamilyAndStyle },
	{ "Test SetDrawingMode", testSetDrawingMode },
	{ "Test PushPopState", testPushPopState },
	{ "Test FontRotation", testFontRotation },
	{ NULL, NULL }
};


