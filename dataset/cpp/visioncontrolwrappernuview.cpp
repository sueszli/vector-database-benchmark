#include "visioncontrolwrappernuview.h"

VisionControlWrapper* VisionControlWrapper::instance = 0;

VisionControlWrapper* VisionControlWrapper::getInstance()
{
    if(!instance)
        instance = new VisionControlWrapper();
    return instance;
}

VisionControlWrapper::VisionControlWrapper()
{
    wrapper = DataWrapper::getInstance();
}

//void VisionControlWrapper::setCallBack(virtualNUbot *virtual_nubot)
//{
//    wrapper->setCallBack(virtual_nubot);
//}

int VisionControlWrapper::runFrame()
{
    static unsigned int frame = 0;
    int status;
    #if VISION_WRAPPER_VERBOSITY > 1
        debug << "VisionControlWrapper::runFrame(): - frame " << frame << std::endl;
    #endif
    frame = (frame + 1) % 10000;
    
    if(!wrapper->updateFrame()) {
        #if VISION_WRAPPER_VERBOSITY > 1
            debug << "VisionControlWrapper::runFrame() - updateFrame() failed" << std::endl;
        #endif
        return -1;  //failure - do not run vision
    }

    status = controller.runFrame(Blackboard->lookForBall, Blackboard->lookForGoals, Blackboard->lookForFieldPoints, Blackboard->lookForObstacles); //run vision on the frame

    wrapper->postProcess();

    return status;
}

void VisionControlWrapper::saveAnImage() const
{
    #if VISION_WRAPPER_VERBOSITY > 1
        debug << "VisionControlWrapper::saveAnImage():" << std::endl;
    #endif
    wrapper->saveAnImage();
}

void VisionControlWrapper::setRawImage(const NUImage* image)
{
    wrapper->setRawImage(image);
}

void VisionControlWrapper::setSensorData(NUSensorsData* sensors)
{
    wrapper->setSensorData(sensors);
}

void VisionControlWrapper::setFieldObjects(FieldObjects *field_objects)
{
    wrapper->setFieldObjects(field_objects);
}

void VisionControlWrapper::setLUT(unsigned char *vals)
{
    wrapper->setLUT(vals);
}

void VisionControlWrapper::classifyImage(ClassifiedImage &classed_image)
{
    wrapper->classifyImage(classed_image);
}

void VisionControlWrapper::classifyPreviewImage(ClassifiedImage &target,unsigned char* temp_vals) const
{
    wrapper->classifyPreviewImage(target, temp_vals);
}
