#include "linedetectorsam.h"
//#include "Tools/Profiling/Profiler.h"
#include "debug.h"
#include "Vision/visionconstants.h"
#include "Tools/Math/General.h"
#include "Vision/visionblackboard.h"
#include "Vision/visionconstants.h"

#include <boost/foreach.hpp>

LineDetectorSAM::LineDetectorSAM() {}
LineDetectorSAM::~LineDetectorSAM()
{
    noisePoints.clear();
}

std::vector<FieldLine> LineDetectorSAM::run(const std::vector<NUPoint>& points)
{
    std::vector<std::pair<LSFittedLine, LSFittedLine> > linePairs = fitLines(points, true);
    std::vector<FieldLine> finalLines;
    for(size_t i=0; i<linePairs.size(); i++) {
        finalLines.push_back(FieldLine(linePairs[i].second, linePairs[i].first));
    }

    return finalLines;
}

std::vector< std::pair<LSFittedLine, LSFittedLine> > LineDetectorSAM::fitLines(const std::vector<NUPoint>& points, bool noise) {
    //Performs split-and-merge algorithm with input consisting of a set of point clusters
    // and a set of unclustered points, putting the resulting lines into a reference
    // passed std::vector
    //Import parameters from constants file
    SPLIT_DISTANCE = VisionConstants::SAM_SPLIT_DISTANCE;
    MIN_POINTS_OVER = VisionConstants::SAM_MIN_POINTS_OVER;
    MAX_ANGLE_DIFF_TO_MERGE = VisionConstants::SAM_MAX_ANGLE_DIFF_TO_MERGE;
    MAX_DISTANCE_TO_MERGE = VisionConstants::SAM_MAX_DISTANCE_TO_MERGE;
    MIN_POINTS_TO_LINE = VisionConstants::SAM_MIN_POINTS_TO_LINE;
    MIN_POINTS_TO_LINE_FINAL = VisionConstants::SAM_MIN_POINTS_TO_LINE_FINAL;
    MIN_LINE_R2_FIT = VisionConstants::SAM_MIN_LINE_R2_FIT;
    MAX_LINE_MSD = VisionConstants::SAM_MAX_LINE_MSD;
    //MAX_POINTS = VisionConstants::SAM_MAX_POINTS;
    MAX_LINES = VisionConstants::SAM_MAX_LINES;
    CLEAR_SMALL = VisionConstants::SAM_CLEAR_SMALL;
    CLEAR_DIRTY = VisionConstants::SAM_CLEAR_DIRTY;

    std::vector< std::pair<LSFittedLine, LSFittedLine> > lines;
    noisePoints.clear();

    //splitIterative(lines, points);
    split(lines, points);

    //Then noise
    if(noise) {
        splitNoise(lines);
    }

    //Do Centre Circle fitting before merge - To do later

    //Then Merge
    //convertLinesEndPoints(lines, vision, linedetector);
    lines = mergeColinear(lines, MAX_ANGLE_DIFF_TO_MERGE, MAX_DISTANCE_TO_MERGE);

    //Then clear unwanted lines
    if(CLEAR_SMALL) {
        clearSmallLines(lines);
    }
    if(CLEAR_DIRTY) {
        clearDirtyLines(lines);
    }

    return lines;
}



void LineDetectorSAM::split(std::vector< std::pair<LSFittedLine, LSFittedLine> >& lines, const std::vector<NUPoint>& points) {
    // Recursive split algorithm

    //Assumes:
    //	- constant detirmined for limit - MIN_POINTS_OVER
    //	- constant for min splitting distance - SAM_THRESHOLD

    //Boundary Conds
    if(lines.size() >= MAX_LINES) {
        addToNoise(points);
        return;
    }
    if(points.size() < MIN_POINTS_TO_LINE) {
        //add points to noise
        addToNoise(points);
        return;
    }

    //temp variables
    unsigned int points_over = 0; //how many points are further from the line than SAM_THRESHOLD
    int greatest_point = 0; //which point in vector is the furthest

    //generate new LSFittedLine
    std::pair<LSFittedLine, LSFittedLine> line;
    BOOST_FOREACH(const NUPoint& g, points) {
        line.first.addPoint(g.groundCartesian);
        line.second.addPoint(g.screenCartesian);
    }

    //check for points over threshold
    findPointsOver(line.first, points_over, greatest_point);

    //if num points over threshold > limit -> split at greatest distance point.
    if(points_over >= MIN_POINTS_OVER) {
        //there are enough points distant to justify a split
        std::vector<NUPoint> left;    //holder vectors
        std::vector<NUPoint> right;
        if(separate(left, right, points[greatest_point], points, line.first)) {
            //split was valid - recursively split new lines
            split(lines, left);
            split(lines, right);
        }
        else {
            //remove furthest point and retry
            vector<NUPoint> newlist = points;
            addToNoise(newlist[greatest_point]);
            newlist.erase(newlist.begin() + greatest_point);
            split(lines, newlist);
        }

    }
    else if(points_over > 0) {
        //not enough points over to split so remove point as noisy, and regen line
        if(points.size() > MIN_POINTS_TO_LINE_FINAL) {
            //removal of a point will still leave enough to form a reasonable line
            vector<NUPoint> newlist = points;
            addToNoise(newlist[greatest_point]);
            for(unsigned int i=greatest_point; i<newlist.size()-1; i++) {
                newlist[i] = newlist[i+1];
            }
            newlist.pop_back();
            line.first.clearPoints();
            line.second.clearPoints();
            generateLines(line, newlist);
            lines.push_back(line);
        }
        else {
            //NOT SURE ??
            //Add points to noise and delete line
            addToNoise(points);
        }
    }
    else {
        //no points over, just push line
        lines.push_back(line);
    }
}

//void LineDetectorSAM::splitIterative(std::vector<LSFittedLine>& lines, std::vector<Point>& points) {
//    //Iterative split algorithm, uses a stack of lines and iterates over it, splitting
//    //each line or adding it to a final list.

//    //Boundary Conds
//    if(lines.size() >= MAX_LINES) {
//        return;
//    }
//    if(points.size() < MIN_POINTS_TO_LINE) {
//        //add points to noise
//        for(unsigned int i=0; i<points.size(); i++)
//            addToNoise(points[i]);
//        return;
//    }

//    //Locals
//    std::vector<LSFittedLine> stack;
//    stack.clear();
//    int furthest_point;
//    int points_over;
//    std::vector<Point> left, right;

//    LSFittedLine* tempLine = new LSFittedLine();
//    //generate first line
//    generateLine(*tempLine, points);
//    //push line onto stack
//    stack.push_back(tempLine);

//    //Begin iteration
//    while(!stack.empty() && lines.size() + stack.size() < MAX_LINES) {
//        //Pop the top line off and split it if warranted
//        //if not, slap it on the end of lines and go again
//        //until stack is empty or maximum lines reached

//        //Clear left and right
//        left.clear();
//        right.clear();

//        //pop top line
//        tempLine = stack.back();
//        stack.pop_back();

//        //check for points over threshold
//        findFurthestPoint(*tempLine, points_over, furthest_point);
//        //qDebug() << points_over << furthest_point;
//        //Options
//        if((unsigned int)points_over >= MIN_POINTS_OVER) {
//            //See if separation is an option
//            //qDebug() << "going to seperate the line: " << furthest_point << (tempLine->getPoints()[furthest_point])->x << (tempLine->getPoints()[furthest_point])->y;
//            if(separate(left, right, tempLine->getPoints()[furthest_point], *tempLine)) {
//                //qDebug() << "separating worked";
//                //clear old line
//                if(!useTripleSplit)
//                {
//                    tempLine->getPoints().clear();
//                    delete tempLine;
//                    //check if left is big enough
//                    if(left.size() >= MIN_POINTS_TO_LINE) {
//                        //generate line and push it to stack
//                        tempLine = new LSFittedLine();
//                        generateLine(*tempLine, left);
//                        stack.push_back(tempLine);
//                    }
//                    else {
//                        //throw left points to noise
//                        for(unsigned int i=0; i<left.size(); i++)
//                            addToNoise(left[i]);
//                        left.clear();
//                    }
//                    //check if right is big enough
//                    if(right.size() >= MIN_POINTS_TO_LINE) {
//                        //generate line and push it to stack
//                        tempLine = new LSFittedLine();
//                        generateLine(*tempLine, right);
//                        stack.push_back(tempLine);
//                    }
//                    else {
//                        //throw right points to noise
//                        for(unsigned int i=0; i<right.size(); i++)
//                            addToNoise(right[i]);
//                        right.clear();
//                    }
//                }
//                else //triple split
//                {
//                    //qDebug() << "Using the Tripple Split";
//                    tempLine->getPoints().clear();
//                    delete tempLine;
//                    //check if left is big enough
//                    if(centre.size() >= MIN_POINTS_TO_LINE) {
//                        //generate line and push it to stack
//                        tempLine = new LSFittedLine();
//                        generateLine(*tempLine, centre);
//                        stack.push_back(tempLine);
//                    }
//                    else {
//                        //throw left points to noise
//                        for(unsigned int i=0; i<centre.size(); i++)
//                            addToNoise(centre[i]);
//                        centre.clear();
//                    }
//                    //check if right is big enough
//                    if(above.size() >= MIN_POINTS_TO_LINE) {
//                        //generate line and push it to stack
//                        tempLine = new LSFittedLine();
//                        generateLine(*tempLine, above);
//                        stack.push_back(tempLine);
//                    }
//                    else {
//                        //throw right points to noise
//                        for(unsigned int i=0; i<above.size(); i++)
//                            addToNoise(above[i]);
//                        above.clear();
//                    }
//                    if(below.size() >= MIN_POINTS_TO_LINE) {
//                        //generate line and push it to stack
//                        tempLine = new LSFittedLine();
//                        generateLine(*tempLine, below);
//                        stack.push_back(tempLine);
//                    }
//                    else {
//                        //throw right points to noise
//                        for(unsigned int i=0; i<below.size(); i++)
//                            addToNoise(below[i]);
//                        below.clear();
//                    }

//                }
//            } //if(separate())
//            else {
//                //Separation didn't work
//                //remove furthest point and push new line
//                vector<Point> newlist = tempLine->getPoints();
//                //remove noisy point and update points list
//                addToNoise(newlist[furthest_point]);
//                newlist[furthest_point] = newlist[newlist.size() - 1];
//                newlist[newlist.size()-1] = 0;
//                newlist.pop_back();
//                //clear old line
//                tempLine->getPoints().clear();
//                delete tempLine;
//                tempLine = new LSFittedLine();
//                generateLine(*tempLine, newlist); //make new line
//                stack.push_back(tempLine); //push it back
//            }
//        } //if(points_over >= MIN_POINTS_OVER)
//        else if(points_over > 0){
//            //not enough points over to split - but there are points over
//            if(tempLine->getPoints().size() > MIN_POINTS_TO_LINE_FINAL) {
//                //i.e. removal of a point will still leave enough to form a reasonable line
//                //remove noisy point and regen line
//                vector<Point> newlist = tempLine->getPoints();
//                //remove noisy point and update points list
//                addToNoise(newlist[furthest_point]);
//                newlist[furthest_point] = newlist[newlist.size() - 1];
//                newlist[newlist.size()-1] = 0;
//                newlist.pop_back();
//                tempLine->getPoints().clear();
//                delete tempLine; //clear memory of old line
//                tempLine = new LSFittedLine();
//                generateLine(*tempLine, newlist); //make new line
//                //qDebug() << "pushing line";
//                lines.push_back(tempLine); //push it back to finals
//                noFieldLines++;
//            }
//            else {
//                //Add points to noise and delete line
//                for(unsigned int i=0; i<tempLine->getPoints().size(); i++)
//                    addToNoise(tempLine->getPoints()[i]);
//                tempLine->getPoints().clear();
//                delete tempLine;
//            }
//        } //elseif(points_over > 0)
//        else {
//            //no points over, just push line to finals
//            //qDebug() << "pushing line";
//            lines.push_back(tempLine);
//            noFieldLines++;
//        }
//    } // While(!stack.empty() && noFieldLines + stack.size() <= MAX_LINES)


////qDebug() << stack.size();
//    //If MAX_LINES reached, but stack is not empty push stack lines to finals
//    if(noFieldLines + stack.size() == MAX_LINES) {
//        //qDebug() << "Maximum Lines reached in split";
//    }
//    while(!stack.empty()) {
//        lines.push_back(stack.back());
//        stack.pop_back();
//    }
//}

void LineDetectorSAM::findPointsOver(LSFittedLine& line, unsigned int& points_over, int& furthest_point) {
    //this method finds the furthest point from a line and returns (via parameters)
    //the number of points over the SPLIT_DISTANCE threshold and the index of
    //the furthest point

    //temp variables
    double distance; 	//holder for calculated PointDistance
    double greatest_distance = 0.0; //saves recalculation of greatest point distance
    unsigned int current_point = 0;
    points_over = 0;
    furthest_point = -1;
    std::vector<Point> points = line.getPoints();

    //check points for perp distance over threshold
    for(current_point = 0; current_point < points.size(); current_point++) {
        distance = line.getLinePointDistance(points[current_point]);
        //qDebug() <<current_point <<distance;

        if(distance > SPLIT_DISTANCE) {
            //potential splitting point
            points_over++; //increment points_over counter
            //check if greatest point
            if(distance > greatest_distance) {
                //new furthest point found
                greatest_distance = distance;
                furthest_point = current_point;
            }
        }
    }
    //qDebug() <<furthest_point <<greatest_distance;
}

void LineDetectorSAM::splitNoise(std::vector<std::pair<LSFittedLine, LSFittedLine> > &lines) {
    //this method creates a copy of the noisePoints std::vector,
    //clears the current noisePoints std::vector and runs
    //the split algorithm on the copy

    if(noisePoints.size() >= MIN_POINTS_TO_LINE_FINAL) {
        std::vector<NUPoint> noiseCopy;

        noiseCopy = noisePoints;
        noisePoints.clear();
        //splitIterative(lines, noiseCopy);
        split(lines, noiseCopy);
    }
}

bool LineDetectorSAM::separate(std::vector<NUPoint> &left, std::vector<NUPoint> &right, NUPoint split_point, const std::vector<NUPoint> &points, const LSFittedLine& line) {
    /*splits a section of points around a splitting point by rotating and translating onto the line about the splitting point
     *Pre: left and right should be empty std::vectors
     *		points contains all the points to be split
     *		split_point is a valid point in points
     *		line is the LSFittedLine
     *Post: left contains all points with negative transformed x-vals
     *		right contains all points with non-negative transformed x-vals (note split_point will be in right)
     *      if left or right is empty, returns false indicating no actual split occurred
    */

    //temp holder vars
    double x_split = split_point.groundCartesian.x;
    double y_split = split_point.groundCartesian.y;

    left.push_back(split_point);    //splitting point should be included in both groups
    right.push_back(split_point);
    if(line.isHorizontal()) {
        //horizontal line - no rotation
        BOOST_FOREACH(NUPoint pt, points) {
            if(pt.groundCartesian != split_point.groundCartesian) {
                if(pt.groundCartesian.x < x_split) //point is to the left
                    left.push_back(pt);
                else
                    right.push_back(pt);
            }
        }
    }
    else if(line.isVertical()) {
        //vertical line - 90 degree rotation
        BOOST_FOREACH(NUPoint pt, points) {
            if(pt.groundCartesian != split_point.groundCartesian) {
                if(pt.groundCartesian.y < y_split) //point is to the left
                    left.push_back(pt);
                else
                    right.push_back(pt);
            }
        }
    }
    else {
        double xsplit = line.projectOnto(split_point.groundCartesian).x;
        BOOST_FOREACH(NUPoint pt, points) {
            //check all points, calculate translated x coord
            //and place in appropriate std::vector
            if(pt.groundCartesian != split_point.groundCartesian) {
                if(line.projectOnto(pt.groundCartesian).x < xsplit) {
                    //point is to the left
                    left.push_back(pt);
                }
                else {
                    right.push_back(pt);
                }
            }
        }
    }

    return (left.size() < points.size() && right.size() < points.size());
}

void LineDetectorSAM::generateLines(std::pair<LSFittedLine, LSFittedLine>& lines, const std::vector<NUPoint>& points) {
    //creates a Least Squared Fitted line

    lines.first.clearPoints();
    lines.second.clearPoints();
    BOOST_FOREACH(const NUPoint& g, points) {
        lines.first.addPoint(g.groundCartesian);
        lines.second.addPoint(g.screenCartesian);
    }
}

//GENERIC

void LineDetectorSAM::addToNoise(const NUPoint& point) {
    //NOT EFFICIENT
    //O(M) for every insertion - where M is the size of noisePoints
    BOOST_FOREACH(NUPoint pt, noisePoints) {
        if(pt.groundCartesian == point.groundCartesian)
            return;
    }
    //only occurs if there are not copies of the point in the noise std::list
    noisePoints.push_back(point);
}

void LineDetectorSAM::addToNoise(const std::vector<NUPoint > &points) {
    BOOST_FOREACH(NUPoint pt, points) {
        addToNoise(pt);
    }
}

void LineDetectorSAM::clearSmallLines(std::vector<std::pair<LSFittedLine, LSFittedLine> >& lines) {
    //removes any lines from the std::vector whose std::vector of
    //member points is too small

    std::vector<std::pair<LSFittedLine, LSFittedLine> >::iterator it = lines.begin();

    while(it < lines.end()) {
        if(it->first.getNumPoints() < MIN_POINTS_TO_LINE_FINAL)
            it = lines.erase(it);
        else
            it++;
    }
}


void LineDetectorSAM::clearDirtyLines(std::vector<std::pair<LSFittedLine, LSFittedLine> > &lines) {
    //removes any lines from the std::vector whose R^2 value is
    //less than MIN_LINE_R2_FIT
    std::vector<std::pair<LSFittedLine, LSFittedLine> >::iterator it = lines.begin();

    while(it < lines.end()) {
        if(it->first.getr2tls() < MIN_LINE_R2_FIT || it->first.getMSD() > MAX_LINE_MSD) {
            it = lines.erase(it);
        }
        else {
            it++;
        }
    }
}
