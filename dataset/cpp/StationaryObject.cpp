#include "StationaryObject.h"
#include <sstream>
StationaryObject::StationaryObject(const Vector2<float>& initialFieldLocation, int id, const std::string& initName):
        Object(id, initName),
        fieldLocation(initialFieldLocation)
{

}

StationaryObject::StationaryObject(float x, float y, int id, const std::string& initName):
        Object(id, initName),
        fieldLocation(x,y)
{

}

StationaryObject::StationaryObject(const StationaryObject& otherObject):
        Object(otherObject.getID(), otherObject.getName()),
        fieldLocation(otherObject.getFieldLocation())
{

}

StationaryObject::~StationaryObject()
{

}

std::string StationaryObject::toString() const
{
    std::stringstream result;
    result << Object::toString();
    result << "Location: (" << fieldLocation.x << "," << fieldLocation.y << ")" << std::endl;
    return result.str();
}

std::ostream& operator<< (std::ostream& output, const StationaryObject& p_stat)
{
    output << *static_cast<const Object*>(&p_stat);

    output.write(reinterpret_cast<const char*>(&p_stat.fieldLocation.x), sizeof(p_stat.fieldLocation.x));
    output.write(reinterpret_cast<const char*>(&p_stat.fieldLocation.y), sizeof(p_stat.fieldLocation.y));

    return output;
}

std::istream& operator>> (std::istream& input, StationaryObject& p_stat)
{
    input >> *static_cast<Object*>(&p_stat);

    float temp;
    input.read(reinterpret_cast<char*>(&temp), sizeof(temp));
    input.read(reinterpret_cast<char*>(&temp), sizeof(temp));
//    input.read(reinterpret_cast<char*>(&p_stat.fieldLocation.x), sizeof(p_stat.fieldLocation.x));
//    input.read(reinterpret_cast<char*>(&p_stat.fieldLocation.y), sizeof(p_stat.fieldLocation.y));

    return input;
}

