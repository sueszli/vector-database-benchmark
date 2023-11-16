/**
*   @name   ScanLines
*   @file   scanlines.cpp
*   @brief  generate horizontal and vertical scanlines.
*   @author David Budden
*   @date   23/03/2012
*/
#include "scanlines.h"
#include "debug.h"
#include "Vision/visionconstants.h"
#include <boost/foreach.hpp>

void ScanLines::generateScanLines()
{
    #if VISION_SCAN_VERBOSITY > 1
        debug << "ScanLines::generateScanLines() - Begin" << std::endl;
    #endif
    VisionBlackboard *vbb = VisionBlackboard::getInstance();
    std::vector<int> horizontal_scan_lines;
    const std::vector<Vector2<double> >& horizon_points = vbb->getGreenHorizon().getInterpolatedPoints();   //need this to get the left and right

    Vector2<double> left = horizon_points.front();
    Vector2<double> right = horizon_points.back();

    if(left.y >= vbb->getImageHeight())
        errorlog << "left: " << left.y << std::endl;
    
    if(right.y >= vbb->getImageHeight())
        errorlog << "right: " << right.y << std::endl;
    
    //unsigned int bottom_horizontal_scan = (left.y + right.y) / 2;
    int bottom_horizontal_scan = vbb->getImageHeight() - 1;    //we need h-scans under the GH for field lines

    if(bottom_horizontal_scan >= vbb->getImageHeight())
        errorlog << "avg: " << bottom_horizontal_scan << std::endl;

    for (int y = bottom_horizontal_scan; y >= 0; y -= VisionConstants::HORIZONTAL_SCANLINE_SPACING) {
        if(y >= vbb->getImageHeight())
            errorlog << " y: " << y << std::endl;
        horizontal_scan_lines.push_back(y);
    }
    
    vbb->setHorizontalScanlines(horizontal_scan_lines);
}

void ScanLines::classifyHorizontalScanLines()
{
    VisionBlackboard* vbb = VisionBlackboard::getInstance();
    const NUImage& img = vbb->getOriginalImage();
    const std::vector<int>& horizontal_scan_lines = vbb->getHorizontalScanlines();
    std::vector< std::vector<ColourSegment> > classifications;

    BOOST_FOREACH(int y, horizontal_scan_lines) {
        classifications.push_back(classifyHorizontalScan(vbb->getLUT(), img, y));
    }
    
    vbb->setHorizontalSegments(classifications);
}

void ScanLines::classifyVerticalScanLines()
{
    VisionBlackboard* vbb = VisionBlackboard::getInstance();
    const NUImage& img = vbb->getOriginalImage();
    const std::vector<Vector2<double> >& vertical_start_points = vbb->getGreenHorizon().getInterpolatedSubset(VisionConstants::VERTICAL_SCANLINE_SPACING);
    std::vector< std::vector<ColourSegment> > classifications;

    for(unsigned int i=0; i<vertical_start_points.size(); i++) {
        classifications.push_back(classifyVerticalScan(vbb->getLUT(), img, vertical_start_points.at(i)));
    }
    
    vbb->setVerticalSegments(classifications);
}

std::vector<ColourSegment> ScanLines::classifyHorizontalScan(const LookUpTable& lut, const NUImage& img, unsigned int y)
{
    std::vector<ColourSegment> result;
    if(y >= img.getHeight()) {
		errorlog << "ScanLines::classifyHorizontalScan invalid y: " << y << std::endl;
		return result;
	}
    //simple and nasty first
    //Colour previous, current, next
    int     start_pos = 0,
            x;
    Colour start_colour = getColourFromIndex(lut.classifyPixel(img(0,y)));
    Colour current_colour;
    ColourSegment segment;

    for(x = 0; x < img.getWidth(); x++) {
        current_colour = getColourFromIndex(lut.classifyPixel(img(x,y)));
        if(current_colour != start_colour) {
            //start of new segment
            //make new segment and push onto std::vector
            segment.set(Point(start_pos, y), Point(x, y), start_colour);
            result.push_back(segment);
            //start new segment
            start_colour = current_colour;
            start_pos = x;
        }
    }
    segment.set(Point(start_pos, y), Point(x-1, y), start_colour);
    result.push_back(segment);
    
    #if VISION_SCANLINE_VERBOSITY > 1
        Point end;
        for(int i=0; i<result.size(); i++) {
            debug << result.at(i).getStart() << " " << result.at(i).getEnd() << " " << (end==result.at(i).getStart()) << std::endl;
            end = result.at(i).getEnd();
        }
    #endif
    return result;
}

std::vector<ColourSegment> ScanLines::classifyVerticalScan(const LookUpTable& lut, const NUImage& img, const Vector2<double> &start)
{
    std::vector<ColourSegment> result;
    if(start.y >= img.getHeight() || start.y < 0 || start.x >= img.getWidth() || start.x < 0) {
		errorlog << "ScanLines::classifyVerticalScan invalid start position: " << start << std::endl; 
		return result;
    }
    //simple and nasty first
    //Colour previous, current, next
    Colour start_colour = getColourFromIndex(lut.classifyPixel(img(start.x,start.y))),
                        current_colour;
    ColourSegment segment;
    int     start_pos = start.y,
            x = start.x,
            y;
            
    for(y = start.y; y < img.getHeight(); y++) {
        current_colour = getColourFromIndex(lut.classifyPixel(img(x,y)));
        if(current_colour != start_colour) {
            //start of new segment
            //make new segment and push onto std::vector
            segment.set(Point(x, start_pos), Point(x, y), start_colour);
            result.push_back(segment);
            //start new segment
            start_colour = current_colour;
            start_pos = y;
        }
    }
    segment.set(Point(x, start_pos), Point(x, y), start_colour);
    result.push_back(segment);
    
    return result;
}
