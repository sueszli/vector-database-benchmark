#include "RobotCandidate.h"

RobotCandidate::RobotCandidate():ObjectCandidate()
{

}

RobotCandidate::RobotCandidate(int left, int top, int right, int bottom): ObjectCandidate(left, top, right, bottom)
{
    //red = 4
    //blue = 9
    //unknown = 3
    colour = 3;
}

RobotCandidate::RobotCandidate(int left, int top, int right, int bottom, std::vector<Vector2<int> > points): ObjectCandidate(left, top, right, bottom), skeleton(points)
{
    colour = 3;
}

RobotCandidate::RobotCandidate(int left, int top, int right, int bottom, unsigned char teamColour): ObjectCandidate(left, top, right, bottom, teamColour)
{

}

RobotCandidate::RobotCandidate(int left, int top, int right, int bottom, unsigned char teamColour, std::vector<Vector2<int> > points): ObjectCandidate(left, top, right, bottom, teamColour), skeleton(points)
{

}

RobotCandidate::~RobotCandidate()
{
    return;
}

std::vector<Vector2<int> > RobotCandidate::getSkeleton() const
{
       return skeleton;
}

unsigned char RobotCandidate::getTeamColour() const
{
    return colour;
}


