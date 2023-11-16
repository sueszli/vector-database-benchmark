#include "datawrapperrpi.h"

#include "Kinematics/Kinematics.h"
#include "debug.h"
#include "nubotdataconfig.h"
#include "debugverbosityvision.h"

#include "Vision/VisionTypes/coloursegment.h"
#include "Vision/basicvisiontypes.h"
#include "Vision/visionconstants.h"

//for controlling GPIO pins
#include <wiringPi.h>



DataWrapper* DataWrapper::instance = 0;

DataWrapper::DataWrapper(std::string istrm, std::string sstrm, std::string cfg, std::string lname)
{
    m_ok = true;
    string cam_spec_name = string(CONFIG_DIR) + string("CameraSpecs.cfg");
    string sen_calib_name = string(CONFIG_DIR) + string("SensorCalibration.cfg");

    debug << "opening camera config: " << cam_spec_name << std::endl;
    if( ! camera_data.LoadFromConfigFile(cam_spec_name.c_str())) {
        errorlog << "DataWrapper::DataWrapper() - failed to load camera specifications: " << cam_spec_name << std::endl;
        m_ok = false;
    }

    debug << "opening sensor calibration config: " << sen_calib_name << std::endl;
    if( ! sensor_calibration.ReadSettings(sen_calib_name)) {
        errorlog << "DataWrapper::DataWrapper() - failed to load sensor calibration: " << sen_calib_name << ". Using default values." << std::endl;
        sensor_calibration = SensorCalibration();
    }

    kinematics_horizon.setLine(0, 1, 0);
    numFramesDropped = numFramesProcessed = 0;

    streamname = istrm;
    debug << "openning image stream: " << streamname << std::endl;
    imagestrm.open(streamname.c_str());

    using_sensors = !sstrm.empty();
    sensorstreamname = sstrm;
    if(m_ok && using_sensors) {
        debug << "openning sensor stream: " << sensorstreamname << std::endl;
        sensorstrm.open(sensorstreamname.c_str());
        if(!sensorstrm.is_open()) {
            std::cerr << "Error : Failed to read sensors from: " << sensorstreamname << " defaulting to sensors off." << std::endl;
            using_sensors = false;
        }
    }

    if(!imagestrm.is_open()) {
        errorlog << "DataWrapper::DataWrapper() - failed to load stream: " << streamname << std::endl;
        m_ok = false;
    }

    debug << "config: " << cfg << std::endl;
    VisionConstants::loadFromFile(cfg);

    if(!loadLUTFromFile(lname)){
        errorlog << "DataWrapper::DataWrapper() - failed to load LUT: " << lname << std::endl;
        m_ok = false;
    }
    gpio = (wiringPiSetupSys() != -1);
    if(gpio) {
        system("gpio -g mode 17 out");
        system("gpio -g mode 18 out");
        system("gpio -g mode 22 out");
        system("gpio -g mode 23 out");
        system("gpio export 17 out");
        system("gpio export 18 out");
        system("gpio export 22 out");
        system("gpio export 23 out");
    }
    else {
        std::cerr << "Failed to setup wiringPi - GPIO pins unavailable" << std::endl;
    }
}

DataWrapper::~DataWrapper()
{
    imagestrm.close();
    if(using_sensors)
        sensorstrm.close();
}

DataWrapper* DataWrapper::getInstance()
{
    return instance;
}

/**
*   @brief Fetches the next frame.
*/
NUImage* DataWrapper::getFrame()
{
    return &current_frame;
}

/*! @brief Retrieves the horizon data and builds a Horizon and returns it.
*   @return kinematics_horizon A reference to the kinematics horizon line.
*   @note This method has a chance of retrieving an invalid line, in this case
*           the old line is returned with the "exists" flag set to false.
*/
const Horizon& DataWrapper::getKinematicsHorizon()
{
    return kinematics_horizon;
}

//! @brief Retrieves the camera height returns it.
float DataWrapper::getCameraHeight() const
{
    return camera_height;
}

//! @brief Retrieves the camera pitch returns it.
float DataWrapper::getHeadPitch() const
{
    return head_pitch;
}

//! @brief Retrieves the camera yaw returns it.
float DataWrapper::getHeadYaw() const
{
    return head_yaw;
}

//! @brief Retrieves the body pitch returns it.
Vector3<float> DataWrapper::getOrientation() const
{
    return orientation;
}

//! @brief Returns the neck position snapshot.
Vector3<double> DataWrapper::getNeckPosition() const
{
    return neck_position;
}

Vector2<double> DataWrapper::getCameraFOV() const
{
    return Vector2<double>(camera_data.m_horizontalFov, camera_data.m_verticalFov);
}

//! @brief Returns camera settings.
CameraSettings DataWrapper::getCameraSettings() const
{
    return CameraSettings();
}

SensorCalibration DataWrapper::getSensorCalibration() const
{
    return sensor_calibration;
}

/*! @brief Returns a reference to the stored Lookup Table
*   @return LUT A reference to the current LUT
*/
const LookUpTable& DataWrapper::getLUT() const
{
    return LUT;
}

void DataWrapper::publish(const std::vector<const VisionFieldObject*>& visual_objects)
{
}

void DataWrapper::publish(const VisionFieldObject* visual_object)
{
    if( isGoal(visual_object->getID()) )
    {
        digitalWrite(18, HIGH);
    }
    else if( visual_object->getID() == BALL )
    {
        digitalWrite(17, HIGH);
    }
    else if( visual_object->getID() == FIELDLINE )
    {
        digitalWrite(23, HIGH);
    }
    else if( visual_object->getID() == OBSTACLE )
    {
        digitalWrite(22, HIGH);
    }

    #if VISION_WRAPPER_VERBOSITY > 0
    visual_object->printLabel(debug);
    debug << std::endl;
    #endif
}

void DataWrapper::debugPublish(const std::vector<Ball>& data) 
{
    #if VISION_WRAPPER_VERBOSITY > 1
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(DBID_BALLS) << std::endl;
        for(Ball ball : data) {
            debug << "DataWrapper::debugPublish - Ball = " << ball << std::endl;
        }
    #endif
}

void DataWrapper::debugPublish(const std::vector<Goal>& data) 
{
    #if VISION_WRAPPER_VERBOSITY > 1
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(DBID_GOALS) << std::endl;
        for(Goal post : data) {
            debug << "DataWrapper::debugPublish - Goal = " << post << std::endl;
        }
    #endif
}

void DataWrapper::debugPublish(const std::vector<Obstacle>& data) 
{
    #if VISION_WRAPPER_VERBOSITY > 1
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(DBID_OBSTACLES) << std::endl;
        for(Obstacle obst : data) {
            debug << "DataWrapper::debugPublish - Obstacle = " << obst << std::endl;
        }
    #endif
}

void DataWrapper::debugPublish(const std::vector<FieldLine>& data)
{
    #if VISION_WRAPPER_VERBOSITY > 2
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(id) << std::endl;
        for(const FieldLine& l : data) {
            debug << "DataWrapper::debugPublish - Line = ";
            l.printLabel(debug);
            debug << std::endl;
        }
    #endif
}

void DataWrapper::debugPublish(const std::vector<CentreCircle>& data)
{
    #if VISION_WRAPPER_VERBOSITY > 2
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(id) << std::endl;
        for(const CentreCircle& c : data) {
            debug << "DataWrapper::debugPublish - CentreCircle = ";
            debug << c << std::endl;
        }
    #endif
}

void DataWrapper::debugPublish(const std::vector<CornerPoint>& data)
{
    #if VISION_WRAPPER_VERBOSITY > 2
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(id) << std::endl;
        for(const CornerPoint& c : data) {
            debug << "DataWrapper::debugPublish - CornerPoint = ";
            debug << c << std::endl;
        }
    #endif
}

void DataWrapper::debugPublish(DEBUG_ID id, const std::vector<Point>& data_points)
{
    #if VISION_WRAPPER_VERBOSITY > 2
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(id) << std::endl;
        debug << "\t" << id << std::endl;
        debug << "\t" << data_points << std::endl;
    #endif
}

void DataWrapper::debugPublish(DEBUG_ID id, const SegmentedRegion& region)
{
    #if VISION_WRAPPER_VERBOSITY > 2
        debug << "DataWrapper::debugPublish - DEBUG_ID = " << getIDName(id) << std::endl;
        for(const vector<ColourSegment>& line : region.getSegments()) {
            if(region.getDirection() == VisionID::HORIZONTAL)
                debug << "y: " << line.front().getStart().y << std::endl;
            else
                debug << "x: " << line.front().getStart().x << std::endl;
            for(const ColourSegment& seg : line) {
                debug << "\t" << seg;
            }
        }
    #endif
}

void DataWrapper::debugPublish(DEBUG_ID id, NUImage const* const img)
{
}

void DataWrapper::debugPublish(DEBUG_ID id, const std::vector<Goal>& data)
{
}

void DataWrapper::plotCurve(std::string name, std::vector< Point > pts)
{
}

void DataWrapper::plotLineSegments(std::string name, std::vector< Point > pts)
{
}

void DataWrapper::plotHistogram(std::string name, const Histogram1D& hist, Colour colour)
{
}

/*! @brief Updates the held information ready for a new frame.
*   Gets copies of the actions and sensors pointers from the blackboard and
*   gets a new image from the blackboard. Updates framecounts.
*   @return Whether the fetched data is valid.
*/
bool DataWrapper::updateFrame()
{
    digitalWrite (17, LOW);
    digitalWrite (18, LOW);
    digitalWrite (22, LOW);
    digitalWrite (23, LOW);

    if(m_ok) {
        if(!imagestrm.is_open()) {
            errorlog << "No image stream - " << streamname << std::endl;
            return false;
        }
        if(using_sensors && !sensorstrm.is_open()) {
            errorlog << "No sensor stream - " << sensorstreamname << std::endl;
            return false;
        }
        try {
            imagestrm >> current_frame;
        }
        catch(std::exception& e) {
            return false;
        }
        if(using_sensors) {
            try {
                sensorstrm >> sensor_data;
            }
            catch(std::exception& e){
                errorlog << "Sensor stream error: " << e.what() << std::endl;
                return false;
            }
        }

        //overwrite sensor horizon if using sensors
        std::vector<float> hor_data;
        if(using_sensors && sensor_data.getHorizon(hor_data)) {
            kinematics_horizon.setLine(hor_data.at(0), hor_data.at(1), hor_data.at(2));
        }

        //update kinematics snapshot
        if(using_sensors) {

            vector<float> orientation(3, 0);

            if(!sensor_data.getCameraHeight(camera_height))
                errorlog << "DataWrapperQt - updateFrame() - failed to get camera height from NUSensorsData" << std::endl;
            if(!sensor_data.getPosition(NUSensorsData::HeadPitch, head_pitch))
                errorlog << "DataWrapperQt - updateFrame() - failed to get head pitch from NUSensorsData" << std::endl;
            if(!sensor_data.getPosition(NUSensorsData::HeadYaw, head_yaw))
                errorlog << "DataWrapperQt - updateFrame() - failed to get head yaw from NUSensorsData" << std::endl;
            if(!sensor_data.getOrientation(orientation))
                errorlog << "DataWrapperQt - updateFrame() - failed to get orientation from NUSensorsData" << std::endl;
            orientation = Vector3<float>(orientation.at(0), orientation.at(1), orientation.at(2));

            vector<float> left, right;
            if(sensor_data.get(NUSensorsData::LLegTransform, left) and sensor_data.get(NUSensorsData::RLegTransform, right))
            {
                neck_position = Kinematics::CalculateNeckPosition(Matrix4x4fromVector(left), Matrix4x4fromVector(right), sensor_calibration.m_neck_position_offset);
            }
            else
            {
                errorlog << "DataWrapperRPi - updateFrame() - failed to get left or right leg transforms from NUSensorsData" << std::endl;
                // Default in case kinemtaics not available. Base height of darwin.
                neck_position = Vector3<double>(0.0, 0.0, 39.22);
            }
        }
        else {
            camera_height = head_pitch = head_yaw = 0;
            orientation = Vector3<float>(0,0,0);
            neck_position = Vector3<double>(0.0, 0.0, 39.22);
        }

        numFramesProcessed++;

        return current_frame.getHeight() > 0 && current_frame.getWidth() > 0;
    }
    return false;
}

/**
*   @brief loads the colour look up table
*   @param filename The filename for the LUT stored on disk
*/
bool DataWrapper::loadLUTFromFile(const std::string& fileName)
{
    #if VISION_WRAPPER_VERBOSITY > 1
    debug << "DataWrapper::loadLUTFromFile() - " << fileName << std::endl;
    #endif
    return LUT.loadLUTFromFile(fileName);
}
