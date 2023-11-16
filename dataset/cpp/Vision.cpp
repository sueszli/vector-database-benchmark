/*!
  @file Vision.h
  @brief Declaration of NUbots Vision class.
  @author Steven Nicklin
*/

#include "Vision.h"
#include "Infrastructure/NUImage/NUImage.h"
#include "Tools/Math/Line.h"
#include "ClassificationColours.h"
#include "Ball.h"
#include "GoalDetection.h"
#include "Tools/Math/General.h"
#include <boost/circular_buffer.hpp>
#include <queue>
#include <algorithm>
#include "debug.h"
#include "debugverbosityvision.h"
#include "nubotdataconfig.h"

#include "Kinematics/Kinematics.h"
#include "NUPlatform/NUPlatform.h"
#include "Infrastructure/NUBlackboard.h"
#include "Infrastructure/Jobs/JobList.h"
#include "Infrastructure/Jobs/CameraJobs/ChangeCameraSettingsJob.h"
#include "Infrastructure/Jobs/VisionJobs/SaveImagesJob.h"
#include "Infrastructure/NUSensorsData/NUSensorsData.h"
#include "Infrastructure/NUActionatorsData/NUActionatorsData.h"
#include "NUPlatform/NUActionators/NUSounds.h"
#include "NUPlatform/NUIO.h"
#include "NUPlatform/NUCamera/NUCameraData.h"

#include "Vision/Threads/SaveImagesThread.h"
#include <iostream>

#include "Tools/Profiling/Profiler.h"

//#include <QDebug>

using namespace mathGeneral;
Vision::Vision()
{
    classifiedCounter = 0;
    LUTBuffer = new unsigned char[LUTTools::LUT_SIZE];
    currentLookupTable = LUTBuffer;
    loadLUTFromFile(string(DATA_DIR) + string("default.lut"));
    m_saveimages_thread = new SaveImagesThread(this);
    isSavingImages = false;
    isSavingImagesWithVaryingSettings = false;
    numSavedImages = 0;
    ImageFrameNumber = 0;
    numFramesDropped = 0;
    numFramesProcessed = 0;
    m_camera_specs = new NUCameraData(string(CONFIG_DIR) + string("CameraSpecs.cfg"));

    m_profiling_stream.open("VisionProfiling.txt");

    return;
}

Vision::~Vision()
{
    m_profiling_stream.close();
    // delete AllFieldObjects;
    delete [] LUTBuffer;
    imagefile.close();
    sensorfile.close();
    delete m_camera_specs;
    return;
}

void Vision::process(JobList* jobs)
{
    #if DEBUG_VISION_VERBOSITY > 4
        debug  << "Vision::Process - Begin" << endl;
    #endif
    static list<Job*>::iterator it;     // the iterator over the motion jobs

    for (it = jobs->vision_begin(); it != jobs->vision_end();)
    {
        if ((*it)->getID() == Job::VISION_SAVE_IMAGES)
        {
            #if DEBUG_VISION_VERBOSITY > 4
                debug << "Vision::process(): Processing a save images job." << endl;
            #endif
            static SaveImagesJob* job;
            job = (SaveImagesJob*) (*it);
            if(isSavingImages != job->saving())
            {
                if(job->saving() == true)
                {
                    currentSettings = currentImage->getCameraSettings();
                    if (!imagefile.is_open())
                        imagefile.open((string(DATA_DIR) + string("image.strm")).c_str());
                    if (!sensorfile.is_open())
                        sensorfile.open((string(DATA_DIR) + string("sensor.strm")).c_str());
                    m_actions->add(NUActionatorsData::Sound, m_sensor_data->CurrentTime, NUSounds::START_SAVING_IMAGES);
                }
                else
                {
                    imagefile.flush();
                    sensorfile.flush();

                    ChangeCameraSettingsJob* newJob  = new ChangeCameraSettingsJob(currentSettings);
                    jobs->addCameraJob(newJob);
                    m_actions->add(NUActionatorsData::Sound, m_sensor_data->CurrentTime, NUSounds::STOP_SAVING_IMAGES);
                }
            }
            isSavingImages = job->saving();
            isSavingImagesWithVaryingSettings = job->varyCameraSettings();
            it = jobs->removeVisionJob(it);
        }
        else
        {
            ++it;
        }
    }
}


void Vision::ProcessFrame(NUImage* image, NUSensorsData* data, NUActionatorsData* actions, FieldObjects* fieldobjects)
{
    Profiler prof("Vision");
    prof.start();

    #if DEBUG_VISION_VERBOSITY > 4
        debug << "Vision::ProcessFrame()." << endl;
    #endif

    if (image == NULL || data == NULL || actions == NULL || fieldobjects == NULL)
    {
        // keep object times updated.
        if(fieldobjects && data)
        {
            AllFieldObjects->preProcess(data->GetTimestamp());
            AllFieldObjects->postProcess(data->GetTimestamp());
        }
        return;
    }
    m_sensor_data = data;
    m_actions = actions;
    setFieldObjects(fieldobjects);

    if (currentImage != NULL and image->GetTimestamp() - m_timestamp > 40)
        numFramesDropped++;
    numFramesProcessed++;

    setImage(image);
    //debug << "Camera Settings: " << image->getCameraSettings();
    AllFieldObjects->preProcess(image->GetTimestamp());

    std::vector< Vector2<int> > points;
    std::vector< Vector2<int> > green_border_points;
    //std::vector< Vector2<int> > verticalPoints;
    //std::vector< TransitionSegment > allsegments;
    //std::vector< TransitionSegment > segments;
    //std::vector< ObjectCandidate > candidates;
    //std::vector< ObjectCandidate > tempCandidates;
    //std::vector< Vector2<int> > horizontalPoints;
    //std::vector<LSFittedLine> fieldLines;
    //spacings = (int)(currentImage->getWidth()/20); //16 for Robot, 8 for simulator = width/20
    Circle circ;
    int tempNumScanLines = 0;
    //debug << "Setting Image: " <<endl;

    if(isSavingImages)
    {
        #if DEBUG_VISION_VERBOSITY > 1
            debug << "Vision::starting the save images loop." << endl;
        #endif
        m_saveimages_thread->signal();
    }
    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Generating Horizon Line: " << endl;
    #endif
    //Generate HorizonLine:
    vector <float> horizonInfo;


    if(m_sensor_data->get(NUSensorsData::Horizon, horizonInfo))
    {
        m_horizonLine.setLine((double)horizonInfo[0],(double)horizonInfo[1],(double)horizonInfo[2]);
    }
    else
    {
        #if DEBUG_VISION_VERBOSITY > 5
            debug << "No Horizon Data" << endl;
        #endif
        //AllFieldObjects->postProcess(image->m_timestamp);
        //return;
        m_horizonLine.setLine(0,1,0);
    }

    #if DEBUG_VISION_VERBOSITY > 7
        debug << "Generating Horizon Line: Finnished" <<endl;
        debug << "Image(0,0) is below: " << m_horizonLine.IsBelowHorizon(0, 0)<< endl;
    #endif

    std::vector<unsigned char> validColours;
    Vision::tCLASSIFY_METHOD method;
    const int ROBOTS = 0;
    const int BALL   = 1;
    const int YELLOW_GOALS  = 2;
    const int BLUE_GOALS  = 3;
    int mode  = ROBOTS;


    //qDebug() << "CASE YUYVGenerate Classified Image: START";
    classifiedCounter = 0;
    //ClassifiedImage target(currentImage->getWidth(),currentImage->getHeight(),true);
    //classifyImage(target);

    //qDebug() << "Generate Classified Image: finnished";
    //setImage(&image);

    prof.split("Update");

    //! Find the green edges
    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Begin Scanning: " << endl;
    #endif

    green_border_points = findGreenBorderPoints(spacings,&m_horizonLine);

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "\tFind Edges: finnished" << endl;
    #endif


    //! Find the Field border:
    points = getConvexFieldBorders(green_border_points);
    points = interpolateBorders(points,spacings);

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "\tGenerating Green Border: Finnished" <<endl;
    #endif

    prof.split("Green Horizon");

    //! Scan Below Horizon Image:
    ClassifiedSection vertScanArea = verticalScan(points,spacings);

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "\tVert ScanPaths : Finnished " << vertScanArea.getNumberOfScanLines() <<endl;
    #endif


    //! Scan Above the Horizon
    ClassifiedSection horiScanArea = horizontalScan(points,spacings);

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "\tHorizontal ScanPaths : Finnished " << horiScanArea.getNumberOfScanLines() <<endl;
    #endif

//        cout << vertScanArea.getNumberOfScanLines() << " " << horiScanArea.getNumberOfScanLines() << endl;

    //! Classify Scan Lines to find Segments
    ClassifyScanArea(&vertScanArea);
    ClassifyScanArea(&horiScanArea);


    prof.split("Classify Scanlines");

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "\tClassify ScanPaths : Finnished" <<endl;
    #endif

    //! Different Segments for Different possible objects:

    std::vector< TransitionSegment > GoalBlueSegments;
    std::vector< TransitionSegment > GoalYellowSegments;
    std::vector< TransitionSegment > BallSegments;
    std::vector< TransitionSegment > horizontalsegments;

    //! Extract and Display Vertical Scan Points:
    tempNumScanLines = vertScanArea.getNumberOfScanLines();
    for (int i = 0; i < tempNumScanLines; i++)
    {
        ScanLine* tempScanLine = vertScanArea.getScanLine(i);
        for(int seg = 0; seg < tempScanLine->getNumberOfSegments(); seg++)
        {
            if(     tempScanLine->getSegment(seg)->getColour() == ClassIndex::blue || tempScanLine->getSegment(seg)->getColour() == ClassIndex::shadow_blue)
            {
                GoalBlueSegments.push_back((*tempScanLine->getSegment(seg)));
            }
            if(     tempScanLine->getSegment(seg)->getColour() == ClassIndex::yellow || tempScanLine->getSegment(seg)->getColour() == ClassIndex::yellow_orange)
            {
                GoalYellowSegments.push_back((*tempScanLine->getSegment(seg)));
            }
            if(     tempScanLine->getSegment(seg)->getColour() == ClassIndex::orange || tempScanLine->getSegment(seg)->getColour() == ClassIndex::yellow_orange
                ||  tempScanLine->getSegment(seg)->getColour() == ClassIndex::pink_orange)
            {
                BallSegments.push_back((*tempScanLine->getSegment(seg)));
            }
        }
    }

//    std::cout << "GoalBlueSegments " << GoalBlueSegments.size() << std::endl;
//    std::cout << "GoalYellowSegments " << GoalYellowSegments.size() << std::endl;
//    std::cout <<  "BallSegments " << BallSegments.size() << std::endl;
//    std::cout << "horizontalsegments " << horizontalsegments.size() << std::endl;

    //! Extract and Display Horizontal Scan Points:
    tempNumScanLines = horiScanArea.getNumberOfScanLines();
    for (int i = 0; i < tempNumScanLines; i++)
    {
        ScanLine* tempScanLine = horiScanArea.getScanLine(i);
        for(int seg = 0; seg < tempScanLine->getNumberOfSegments(); seg++)
        {
            horizontalsegments.push_back((*tempScanLine->getSegment(seg)));
        }
    }

    //! Find Line or Robot Points:
    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Finding Line or Robot Segments: with " << horizontalsegments.size()<< "horizontalsegments"<<endl;
    #endif
    LineDetection LineDetector;

    DetectLineOrRobotPoints(&vertScanArea, &LineDetector);

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Line or Robot Segments: " << " Vertical Line Segments: "<<LineDetector.verticalLineSegments.size() << "Horizontal Line Segments: " << LineDetector.horizontalLineSegments.size()<< " Robot Segments: "<< LineDetector.robotSegments.size()<< endl;
    #endif
    //! Identify Field Objects


    std::vector< ObjectCandidate > HorizontalLineCandidates1;
    std::vector< ObjectCandidate > HorizontalLineCandidates2;
    std::vector< ObjectCandidate > HorizontalLineCandidates3;
    std::vector< ObjectCandidate > HorizontalLineCandidates;
    std::vector< ObjectCandidate > VerticalLineCandidates;
    std::vector< TransitionSegment > LeftoverPoints1;
    std::vector< TransitionSegment > LeftoverPoints2;
    std::vector< TransitionSegment > LeftoverPoints3;

    std::vector< TransitionSegment > LeftoverPoints;
    std::vector< ObjectCandidate > LineCandidates;

    validColours.clear();
    validColours.push_back(ClassIndex::white);
    //validColours.push_back(ClassIndex::blue);


    method = Vision::PRIMS;
    //HorizontalLineCandidates = classifyCandidates(LineDetector.horizontalLineSegments, points,validColours, spacings, 0.001, 10000, 4, LeftoverPoints);
    //VerticalLineCandidates = ClassifyCandidatesAboveTheHorizon(LineDetector.verticalLineSegments,validColours,spacings,4,LeftoverPoints);

    HorizontalLineCandidates1 = classifyCandidates(LineDetector.horizontalLineSegments, points,validColours, spacings, 0.000001, 10000000, 3, LeftoverPoints1);
    HorizontalLineCandidates2 = classifyCandidates(LeftoverPoints1, points,validColours, spacings*2, 0.000001, 10000000, 3, LeftoverPoints2);
    HorizontalLineCandidates3 = classifyCandidates(LeftoverPoints2, points,validColours, spacings*4, 0.000001, 10000000, 3, LeftoverPoints3);
    HorizontalLineCandidates.insert(HorizontalLineCandidates.end(), HorizontalLineCandidates1.begin(),HorizontalLineCandidates1.end());
    HorizontalLineCandidates.insert(HorizontalLineCandidates.end(), HorizontalLineCandidates2.begin(),HorizontalLineCandidates2.end());
    HorizontalLineCandidates.insert(HorizontalLineCandidates.end(), HorizontalLineCandidates3.begin(),HorizontalLineCandidates3.end());
    LeftoverPoints.insert(LeftoverPoints.end(),LeftoverPoints3.begin(),LeftoverPoints3.end());
    VerticalLineCandidates = ClassifyCandidatesAboveTheHorizon(LineDetector.verticalLineSegments,validColours,spacings*3,3,LeftoverPoints);
    LeftoverPoints.clear();

    unsigned int no_unused = 0;
    for(unsigned int i=0; i<LineDetector.horizontalLineSegments.size(); i++) {
        if(!LineDetector.horizontalLineSegments[i].isUsed)
            no_unused++;
    }
    for(unsigned int i=0; i<LineDetector.verticalLineSegments.size(); i++) {
        if(!LineDetector.verticalLineSegments[i].isUsed)
            no_unused++;
    }
    //candidates.insert(candidates.end(),HorizontalLineCandidates.begin(),HorizontalLineCandidates.end());
    //candidates.insert(candidates.end(),VerticalLineCandidates.begin(),VerticalLineCandidates.end());
    LineCandidates.insert(LineCandidates.end(), HorizontalLineCandidates.begin(),HorizontalLineCandidates.end());
    LineCandidates.insert(LineCandidates.end(),VerticalLineCandidates.begin(),VerticalLineCandidates.end());
    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Line Candidates: " << LineCandidates.size() << "\tHorizontal Line Candidates: " << HorizontalLineCandidates.size()<< "\tVertical Candidates: " << VerticalLineCandidates.size()<< endl;
    #endif

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Begin Classify Candidates: " << endl;
    #endif

    std::vector< ObjectCandidate > RobotCandidates;
    std::vector< ObjectCandidate > BallCandidates;
    std::vector< ObjectCandidate > BlueGoalCandidates;
    std::vector< ObjectCandidate > YellowGoalCandidates;
    std::vector< ObjectCandidate > BlueGoalAboveHorizonCandidates;
    std::vector< ObjectCandidate > YellowGoalAboveHorizonCandidates;

    mode = ROBOTS;
    method = Vision::PRIMS;
    for (int i = 0; i < 4; i++)
    {
        switch (i)
        {
            case ROBOTS:
                validColours.clear();
                validColours.push_back(ClassIndex::white);
                validColours.push_back(ClassIndex::pink);
                validColours.push_back(ClassIndex::pink_orange);
                validColours.push_back(ClassIndex::shadow_blue);
                //validColours.push_back(ClassIndex::blue);

                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPRE-ROBOT: Segments:\t" <<  LineDetector.robotSegments.size() << endl;
                #endif

                RobotCandidates = classifyCandidates(LineDetector.robotSegments, points ,validColours, spacings, 0.2, 2.0, 12, method);

                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPOST-ROBOT: Robots candidates:\t" << RobotCandidates.size()<< endl;
                #endif

                break;
            case BALL:
                validColours.clear();
                validColours.push_back(ClassIndex::orange);
                validColours.push_back(ClassIndex::pink_orange);
                validColours.push_back(ClassIndex::yellow_orange);

                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPRE-BALL: Segments:\t" <<  BallSegments.size() <<endl;
                #endif

                BallCandidates = classifyCandidates(BallSegments, points, validColours, spacings, 0, 10.0, 1, method);

                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPOST-BALL: Ball Candidates:\t"<< BallCandidates.size() << endl;
                #endif

                break;
            case YELLOW_GOALS:
                validColours.clear();
                validColours.push_back(ClassIndex::yellow);
                //validColours.push_back(ClassIndex::yellow_orange);
                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPRE-YELLOW-GOALS" << endl;
                #endif

                //tempCandidates = classifyCandidates(segments, points, validColours, spacings, 0.1, 4.0, 2, method);
                YellowGoalAboveHorizonCandidates = ClassifyCandidatesAboveTheHorizon(horizontalsegments,validColours,spacings*1.5,3);
                YellowGoalCandidates = classifyCandidates(GoalYellowSegments, points, validColours, spacings, 0.1, 10.0, 2, method);
                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPOST-YELLOW-GOALS" << endl;
                #endif

                break;
            case BLUE_GOALS:
                validColours.clear();
                validColours.push_back(ClassIndex::blue);
                //validColours.push_back(ClassIndex::shadow_blue);

                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPRE-BLUE-GOALS" << endl;
                #endif

                BlueGoalAboveHorizonCandidates = ClassifyCandidatesAboveTheHorizon(horizontalsegments,validColours,spacings*1.5,3);
                BlueGoalCandidates = classifyCandidates(GoalBlueSegments, points, validColours, spacings, 0.1, 10.0, 2, method);

                #if DEBUG_VISION_VERBOSITY > 5
                    debug << "\tPOST-BLUE-GOALS" <<endl;
                #endif

                break;
        }
    }

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Finnished Classify Candidates" <<endl;
    #endif

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Begin Object Recognition: " <<endl;
    #endif

    prof.split("Segment Filters");

//    std::cout << "robot candidates: " << RobotCandidates.size() << std::endl;
//    std::cout << "goal candidates: " << YellowGoalCandidates.size() << " " << BlueGoalCandidates.size() << std::endl;
//    std::cout << "line candidates: " << LineCandidates.size() << std::endl;
//    std::cout << "ball candidates: " << BallCandidates.size() << std::endl;

    //! Find Robots:

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPre-Robot Formation: " <<endl;
        #endif

        DetectRobots(RobotCandidates);

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPost-Robot Formation: " <<endl;
        #endif

        vector<ObjectCandidate> obstacle_candidates;
        vector<AmbiguousObject> obstacle_objects;

        obstacle_candidates = getObstacleCandidates(green_border_points, points, OBSTACLE_HEIGH_THRESH, OBSTACLE_WIDTH_MIN);
        obstacle_objects = getObjectsFromCandidates(obstacle_candidates);

        prof.split("Obstacle detection");

    //! Find Goals:

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPre-GOALPost Recognition: " <<endl;
        #endif

        DetectGoals(YellowGoalCandidates, YellowGoalAboveHorizonCandidates, horizontalsegments);
        DetectGoals(BlueGoalCandidates, BlueGoalAboveHorizonCandidates, horizontalsegments);

        PostProcessGoals();

        prof.split("Goals");
        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPost-GOALPost Recognition: " <<endl;
        #endif


    //! Form Lines

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPre-Line Formation: with " << LineCandidates.size() << " candidates"<<endl;
        #endif

        //SHANNON
        DetectLines(&LineDetector, LineCandidates, LeftoverPoints);
        //AARON
        //LineDetector.fieldLines.clear();
        //DetectLines(&LineDetector);

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPost-Line Formation: " <<endl;
        #endif

        prof.split("Lines");

    //! Form Ball

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPre-Ball Recognition: " <<endl;
        #endif

        if(BallCandidates.size() > 0)
        {
            circ = DetectBall(BallCandidates);
        }

        prof.split("Ball");

        #if DEBUG_VISION_VERBOSITY > 5
            debug << "\tPost-Ball Recognition: " <<endl;
        #endif

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "Finished Object Recognition: " <<endl;
    #endif
    AllFieldObjects->ambiguousFieldObjects.insert(AllFieldObjects->ambiguousFieldObjects.end(), obstacle_objects.begin(), obstacle_objects.end());
    AllFieldObjects->postProcess(image->GetTimestamp());

    prof.split("Publishing");

    //PLAY A SOUND IF OBJECT SEEN:
    /*if(AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_CORNER_CENTRE_CIRCLE].isObjectVisible())
    {
        m_actions->add(NUActionatorsData::Sound, image->m_timestamp, "error1.wav");
    }
    if(AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_PENALTY_BLUE].isObjectVisible() ||
        AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_PENALTY_YELLOW].isObjectVisible())
    {
        m_actions->add(NUActionatorsData::Sound, image->m_timestamp, "error2.wav");
    }*/
    #if DEBUG_VISION_VERBOSITY > 3
    debug 	<< "Vision::ProcessFrame - Number of Pixels Classified: " << classifiedCounter
            << "\t Percent of Image: " << classifiedCounter / float(currentImage->getWidth() * currentImage->getHeight()) * 100.00 << "%" << endl;
    #endif

    /*For Testing Ultrasonic Distances:
    debug << "US Distances: " ;
    vector<float> leftDistances, rightDistances;
    m_sensor_data->getDistanceLeftValues(leftDistances);
    m_sensor_data->getDistanceRightValues(rightDistances);
    debug << "US Left Distances: " ;
    for(unsigned int i = 0 ; i < leftDistances.size() ; i++)
    {
        debug << "\t " << leftDistances[i];
    }
    debug << "\t\tUS Right Distances: " ;
    for(unsigned int i = 0 ; i < rightDistances.size() ; i++)
    {
        debug << "\t " << rightDistances[i];
    }
    debug << endl;*/

    #if DEBUG_VISION_VERBOSITY > 4
        //! Debug information for Frame:
        debug << "Time: " << m_timestamp << endl;
        for(unsigned int i = 0; i < AllFieldObjects->stationaryFieldObjects.size();i++)
        {
            if(AllFieldObjects->stationaryFieldObjects[i].isObjectVisible() == true)
            {
                debug << " \tStationary Object: " << i << ": " << AllFieldObjects->stationaryFieldObjects[i].getName() << " Seen at \tDistance: " <<  AllFieldObjects->stationaryFieldObjects[i].measuredDistance() << "cm away."
                        << "\tBearing: "<< AllFieldObjects->stationaryFieldObjects[i].measuredBearing() << "\tElevation: " << AllFieldObjects->stationaryFieldObjects[i].measuredElevation()<<endl;
            }
        }
        for(unsigned int i = 0; i < AllFieldObjects->mobileFieldObjects.size();i++)
        {
            if(AllFieldObjects->mobileFieldObjects[i].isObjectVisible() == true)
            {
                debug << "\tMobile Object: " << i << ": " << AllFieldObjects->mobileFieldObjects[i].getName() << " Seen at \tDistance: " <<  AllFieldObjects->mobileFieldObjects[i].measuredDistance() <<  "cm away."
                        << "\tBearing: "<< AllFieldObjects->mobileFieldObjects[i].measuredBearing() << "\tElevation: " << AllFieldObjects->mobileFieldObjects[i].measuredElevation()<<endl;
            }
        }

        for(unsigned int i = 0; i < AllFieldObjects->ambiguousFieldObjects.size();i++)
        {
            if(AllFieldObjects->ambiguousFieldObjects[i].isObjectVisible() == true)
            {
                debug << "\tAmbiguous Object: " << i << ": " << AllFieldObjects->ambiguousFieldObjects[i].getName() << " Seen at \tDistance: " <<  AllFieldObjects->ambiguousFieldObjects[i].measuredDistance() << "cm away."
                        << "\tBearing: "<< AllFieldObjects->ambiguousFieldObjects[i].measuredBearing() << "\tElevation: " << AllFieldObjects->ambiguousFieldObjects[i].measuredElevation()<<endl;
            }
        }
    #endif

        //START: UNCOMMENT TO SAVE IMAGES OF A CERTIAN FIELDOBJECT!!------------------------------------------------------------------------------------

        //TESTING: Save Images which of a field object seen

        //BALL
        /*if(AllFieldObjects->mobileFieldObjects[FieldObjects::FO_BALL].isObjectVisible())
        {
            m_saveimages_thread->signal();
        }*/

    /*
        //YELLOW POST
        if (AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_YELLOW_LEFT_GOALPOST].isObjectVisible() or AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_YELLOW_RIGHT_GOALPOST].isObjectVisible())
            m_saveimages_thread->signal();

        //YELLOW AMBIGUOUS POST
        for (size_t i=0; i<AllFieldObjects->ambiguousFieldObjects.size(); i++)
        {
            int ambig_id = AllFieldObjects->ambiguousFieldObjects[i].getID();
            if (ambig_id == FieldObjects::FO_YELLOW_GOALPOST_UNKNOWN)
            {
                m_saveimages_thread->signal();
                break;
            }

        }

        //BLUE POST
        if (AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_BLUE_LEFT_GOALPOST].isObjectVisible() or AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_BLUE_RIGHT_GOALPOST].isObjectVisible())
            m_saveimages_thread->signal();

        //BLUE AMBIGUOUS POST
        for (size_t i=0; i<AllFieldObjects->ambiguousFieldObjects.size(); i++)
        {
            int ambig_id = AllFieldObjects->ambiguousFieldObjects[i].getID();
            if (ambig_id == FieldObjects::FO_BLUE_GOALPOST_UNKNOWN)
            {
                m_saveimages_thread->signal();
                break;
            }

        }
        */
        //bool BlueGoalSeen = false;
        /*
        if(AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_BLUE_LEFT_GOALPOST].isObjectVisible())
        {
            if(isnan(AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_BLUE_LEFT_GOALPOST].measuredDistance()))
            {
                SaveAnImage();
            }

        }
        if(AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_BLUE_RIGHT_GOALPOST].isObjectVisible() )
        {
            if(isnan(AllFieldObjects->stationaryFieldObjects[FieldObjects::FO_BLUE_RIGHT_GOALPOST].measuredDistance()))
            {
                SaveAnImage();
            }
        }
        for(unsigned int i = 0; i < AllFieldObjects->ambiguousFieldObjects.size();i++)
        {
            if(isnan(AllFieldObjects->ambiguousFieldObjects[i].measuredDistance()))

            {
                SaveAnImage();
            }

        }
        */
        //END: UNCOMMENT TO SAVE IMAGES OF A CERTAIN FIELDOBJECT!!------------------------------------------------------------------------------------

        #if DEBUG_VISION_VERBOSITY > 1
            debug << "Visible Objects:\n" << AllFieldObjects->toString(true) << endl;
        #endif

        prof.split("Debug publishing");
        prof.stop();
        m_profiling_stream << prof << std::endl;
}

void Vision::SaveAnImage()
{
    #if DEBUG_VISION_VERBOSITY > 1
        debug << "Vision::SaveAnImage(). Starting..." << endl;
    #endif

    if (!imagefile.is_open())
        imagefile.open((string(DATA_DIR) + string("image.strm")).c_str());
    if (!sensorfile.is_open())
        sensorfile.open((string(DATA_DIR) + string("sensor.strm")).c_str());

    if (imagefile.is_open() and numSavedImages < 2500)
    {
        if(sensorfile.is_open())
        {
            sensorfile << (*m_sensor_data) << flush;
        }
        NUImage buffer;
        buffer.cloneExisting(*currentImage);
        imagefile << buffer;
        numSavedImages++;

        if (isSavingImagesWithVaryingSettings)
        {
            CameraSettings tempCameraSettings = currentImage->getCameraSettings();
            if (numSavedImages % 10 == 0 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() - 0);
            }
            else if (numSavedImages % 10 == 1 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() - 50);
            }
            else if (numSavedImages % 10 == 2 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() - 25);
            }
            else if (numSavedImages % 10 == 3 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() - 0);
            }
            else if (numSavedImages % 10 == 4 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() + 25);
            }
            else if (numSavedImages % 10 == 5 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() + 50);
            }
            else if (numSavedImages % 10 == 6 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() + 100);
            }
            else if (numSavedImages % 10 == 7 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() + 150);
            }
            else if (numSavedImages % 10 == 8 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() + 200);
            }
            else if (numSavedImages % 10 == 9 )
            {
                tempCameraSettings.p_exposure.set(currentSettings.p_exposure.get() + 300);
            }

            //Set the Camera Setttings using Jobs:
            ChangeCameraSettingsJob* newJob = new ChangeCameraSettingsJob(tempCameraSettings);
            Blackboard->Jobs->addCameraJob(newJob);
            //m_camera->setSettings(tempCameraSettings);
        }
    }
    #if DEBUG_VISION_VERBOSITY > 1
        debug << "Vision::SaveAnImage(). Finished" << endl;
    #endif
}

void Vision::setSensorsData(NUSensorsData* data)
{
    m_sensor_data = data;
}

void Vision::setActionatorsData(NUActionatorsData* actions)
{
    m_actions = actions;
}

void Vision::setFieldObjects(FieldObjects* fieldObjects)
{
    AllFieldObjects = fieldObjects;
    return;
}

void Vision::setLUT(unsigned char* newLUT)
{
    currentLookupTable = newLUT;
    return;
}

void Vision::loadLUTFromFile(const std::string& fileName)
{
    LUTTools lutLoader;
    bool lut_loaded;
    if (lutLoader.LoadLUT(LUTBuffer, LUTTools::LUT_SIZE,fileName.c_str()) == true)
    {
        lut_loaded = true;
        setLUT(LUTBuffer);
    }
    else
    {
        lut_loaded = false;
        std::cerr << "Vision::loadLUTFromFile(" << fileName << "). Failed to load lut." << endl;
    }

#if DEBUG_VISION_VERBOSITY > 0
    if(lut_loaded)
    {
        debug << "Lookup table: " << fileName << " loaded sucesfully." << std::endl;
    }
    else
    {
        debug << "Unable to load lookup table: " << fileName << std::endl;
    }
#endif
}

void Vision::setImage(const NUImage* newImage)
{

    currentImage = newImage;
    m_timestamp = currentImage->GetTimestamp();
//    spacings = (int)(currentImage->getWidth()/20); //16 for Robot, 8 for simulator = width/20
    spacings = 6;
    ImageFrameNumber++;
}



void Vision::classifyPreviewImage(ClassifiedImage &target,unsigned char* tempLut)
{
    //qDebug() << "InVision CLASS Generation:";
    int tempClassCounter = classifiedCounter;
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    //qDebug() << sourceImage->width() << ","<< sourceImage->height();

    target.setImageDimensions(width,height);
    //qDebug() << "Set Dimensions:";
    //currentImage = sourceImage;
    const unsigned char * beforeLUT = currentLookupTable;
    currentLookupTable = tempLut;
    //qDebug() << "Begin Loop:";
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            target.image[y][x] = classifyPixel(x,y);
        }
    }
    classifiedCounter = tempClassCounter;
    currentLookupTable = beforeLUT;
    return;
}
void Vision::classifyImage(ClassifiedImage &target)
{
    //qDebug() << "InVision CLASS Generation:";
    int tempClassCounter = classifiedCounter;
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    //qDebug() << sourceImage->width() << ","<< sourceImage->height();

    target.setImageDimensions(width,height);
    //qDebug() << "Set Dimensions:";
    //currentImage = sourceImage;
    //currentLookupTable = lookUpTable;
    //qDebug() << "Begin Loop:";
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            target.image[y][x] = classifyPixel(x,y);
        }
    }
    classifiedCounter = tempClassCounter;
    return;
}

std::vector< Vector2<int> > Vision::findGreenBorderPoints(int scanSpacing, Horizon* horizonLine)
{
    classifiedCounter = 0;
    std::vector< Vector2<int> > results;
    //debug << "Finding Green Boarders: "  << scanSpacing << "  Under Horizon: " << horizonLine->getA() << "x + " << horizonLine->getB() << "y + " << horizonLine->getC() << " = 0" << endl;

    bool pixel_pushed;
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    //debug << width << " , "<< height << endl;
    int yStart;
    int consecutiveGreenPixels = 0;
    for (int x = 0; x < width; x+=scanSpacing)
    {
        pixel_pushed = false;
        yStart = (int)horizonLine->findYFromX(x);
        if(yStart >= height) {
            results.push_back(Vector2<int>(x,height));    //ensure there is always a point at least at the bottom of the image
            continue;
        }
        if(yStart < 0) yStart = 0;
        consecutiveGreenPixels = 0;
        for (int y = yStart; y < height; y++)
        {
            if(classifyPixel(x,y) == ClassIndex::green)
            {
                consecutiveGreenPixels++;
            }
            else
            {
                consecutiveGreenPixels = 0;
            }
            if(consecutiveGreenPixels >= 5)
            {
                results.push_back(Vector2<int>(x,y-consecutiveGreenPixels+1));
                pixel_pushed = true;
                break;
            }
        }
        if(!pixel_pushed)
            results.push_back(Vector2<int>(x,height));    //ensure there is always a point at least at the bottom of the image
    }
    return results;
}

#define LEFT_OF(x0, x1, x2) ((x1.x-x0.x)*(-x2.y+x0.y)-(x2.x-x0.x)*(-x1.y+x0.y) > 0)

std::vector<Vector2<int> > Vision::getConvexFieldBorders(const std::vector<Vector2<int> >& fieldBorders)
{
  //Andrew's Monotone Chain Algorithm to compute the upper hull
  std::vector<Vector2<int> > hull;
  if(!fieldBorders.size()) return hull;
  const std::vector<Vector2<int> >::const_iterator pmin = fieldBorders.begin(),
                                                   pmax = fieldBorders.end()-1;
  hull.push_back(*pmin);
  for(std::vector<Vector2<int> >::const_iterator pi = pmin + 1; pi != pmax+1; pi++)
  {
    if(!LEFT_OF((*pmin), (*pmax), (*pi)) && pi != pmax)
      continue;

    while((int)hull.size() > 1)
    {
      const std::vector<Vector2<int> >::const_iterator p1 = hull.end() - 1,
                                                       p2 = hull.end() - 2;
      if(LEFT_OF((*p1), (*p2), (*pi)))
        break;
      hull.pop_back();
    }
    hull.push_back(*pi);
  }
  return hull;
}

std::vector<Vector2<int> > Vision::interpolateBorders(const std::vector<Vector2<int> >& fieldBorders, int scanSpacing)
{
    std::vector<Vector2<int> > interpolatedBorders;
    if(!fieldBorders.size()) return interpolatedBorders;
    std::vector<Vector2<int> >::const_iterator nextPoint = fieldBorders.begin();
    std::vector<Vector2<int> >::const_iterator prevPoint = nextPoint++;

    int height = currentImage->getHeight();

    int x = prevPoint->x;
    Vector2<int> deltaPoint, temp;
    for (; nextPoint != fieldBorders.end(); nextPoint++)
    {
        deltaPoint = (*nextPoint) - (*prevPoint);
        for (; x <= nextPoint->x; x+=scanSpacing)
        {
            temp.x = x;
            temp.y = (x - prevPoint->x) * deltaPoint.y / deltaPoint.x + prevPoint->y;
            if (temp.y < 0) temp.y = 0;
            if (temp.y >= height) temp.y = height - 1;
            interpolatedBorders.push_back(temp);
        }
        prevPoint = nextPoint;
    }
    return interpolatedBorders;
}

ClassifiedSection Vision::verticalScan(const std::vector<Vector2<int> >&fieldBorders,int scanSpacing)
{
    //std::vector<Vector2<int> > scanPoints;
    ClassifiedSection scanArea(ScanLine::DOWN);
    if(!fieldBorders.size()) return scanArea;
    std::vector<Vector2<int> >::const_iterator nextPoint = fieldBorders.begin();
    //std::vector<Vector2<int> >::const_iterator prevPoint = nextPoint++; //This iterator is unused
    int x = 0;
    int y = 0;
    int fullLineLength = 0;
    int halfLineLength = 0;
    int quarterLineLength = 0;
    int eightLineLength = 0;
    int midX = 0;
    int skip = int(scanSpacing/2);

    int height = currentImage->getHeight();

    Vector2<int> temp;
    for (; nextPoint != fieldBorders.end(); ++nextPoint)
    {
        x = nextPoint->x;
        y = nextPoint->y;

        //!Create Full ScanLine
        temp.x = x;
        temp.y = y;

        fullLineLength = int(height - y);
        ScanLine tempScanLine(temp, fullLineLength);
        scanArea.addScanLine(tempScanLine);

        //!Create half ScanLine
        midX = x+skip;
        temp.x = midX;
        halfLineLength = int((height - y));
        ScanLine tempMidScanLine(temp,halfLineLength);
        scanArea.addScanLine(tempMidScanLine);

        //!Create Quarter ScanLines
        temp.x = int(midX - skip/2);
        quarterLineLength = int((height - y));
        ScanLine tempLeftQuarterLine(temp,quarterLineLength);
        scanArea.addScanLine(tempLeftQuarterLine);
        temp.x = int(midX + skip/2);
        ScanLine tempRightQuarterLine(temp,quarterLineLength);
        scanArea.addScanLine(tempRightQuarterLine);


        //!Create Eight ScanLines

        temp.x = int(midX - 3*skip/4);
        eightLineLength = int((height - y));
        ScanLine tempLeft1EightLine(temp,eightLineLength);
        scanArea.addScanLine(tempLeft1EightLine);
        temp.x = int(midX - skip/4);
        ScanLine tempLeft2EightLine(temp,eightLineLength);
        scanArea.addScanLine(tempLeft2EightLine);
        temp.x = int(midX + skip/4);
        ScanLine tempRight1EightLine(temp,eightLineLength);
        scanArea.addScanLine(tempRight1EightLine);
        temp.x = int(midX + 3*skip/4);
        ScanLine tempRight2EightLine(temp,eightLineLength);
        scanArea.addScanLine(tempRight2EightLine);
    }

    return scanArea;
}

ClassifiedSection Vision::horizontalScan(const std::vector<Vector2<int> >&fieldBorders,int scanSpacing)
{
    ClassifiedSection scanArea(ScanLine::RIGHT);
    if(!currentImage) return scanArea;
    Vector2<int> temp;
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    //! Case for No FieldBorders
    if(!fieldBorders.size())
    {

        for(int y = 0; y < height; y = y + scanSpacing)
        {
            temp.x = 0;
            temp.y = y;
            ScanLine tempScanLine(temp,width);
            scanArea.addScanLine(tempScanLine);
        }
        return scanArea;
    }

    //! Find the minimum Y, and scan above the field boarders

    std::vector<Vector2<int> >::const_iterator nextPoint = fieldBorders.begin();
    std::vector<Vector2<int> >::const_iterator prevPoint = nextPoint++;
    int minY = height;
    int minX = width;
    int maxY = 0;
    int maxX = 0;
    for (; nextPoint != fieldBorders.end(); nextPoint++)
    {
        if(nextPoint->y < minY)
        {
            minY = nextPoint->y;
        }
        if(nextPoint->y >maxY)
        {
            maxY = nextPoint->y;
        }
        if(nextPoint->x < minX)
        {
            minX = nextPoint->x;
        }
        if(nextPoint->x > maxX)
        {
            maxX = nextPoint->x;
        }
    }

    //! Then calculate horizontal scanlines above the field boarder
    //! Generate Scan pattern for above the max of green boarder.
    for(int y = 0; y < minY; y = y + scanSpacing*0.5)
    {
        temp.x =0;
        temp.y = y;
        ScanLine tempScanLine(temp,width);
        scanArea.addScanLine(tempScanLine);
    }
    //! Generate Scan Pattern for in between the max and min of green horizon.
    for(int y = minY; y < maxY; y = y + scanSpacing*1)
    {
        temp.x =0;
        temp.y = y;
        ScanLine tempScanLine(temp,width);
        scanArea.addScanLine(tempScanLine);
    }
    //! Generate Scan Pattern premature end of horizon
    for(int y = maxY; y < height; y = y + scanSpacing)
    {
        if(maxX > width - scanSpacing*3)
        {
            break;
        }
        temp.x = maxX;
        temp.y = y;
        ScanLine tempScanLine(temp,width-maxX);
        scanArea.addScanLine(tempScanLine);
    }
    for(int y = maxY; y < height; y = y + scanSpacing)
    {
        if(minX < scanSpacing*3)
        {
            break;
        }
        temp.x = 0;
        temp.y = y;
        ScanLine tempScanLine(temp,minX);
        scanArea.addScanLine(tempScanLine);
    }
    return scanArea;
}

void Vision::ClassifyScanArea(ClassifiedSection* scanArea)
{
    int direction = scanArea->getDirection();
    int numOfLines = scanArea->getNumberOfScanLines();
    int lineLength = 0;
    int skipPixel = 1;
    ScanLine* tempLine;
    Vector2<int> currentPoint;
    Vector2<int> tempStartPoint;
    Vector2<int> startPoint;
    unsigned char beforeColour = ClassIndex::unclassified; //!< Colour Before the segment
    unsigned char afterColour = ClassIndex::unclassified;  //!< Colour in the next Segment
    unsigned char currentColour = ClassIndex::unclassified; //!< Colour in the current segment
    //! initialising circular buffer
    int bufferSize = 1;
    boost::circular_buffer<unsigned char> colourBuff(bufferSize);
    for (int i = 0; i < bufferSize; i++)
    {
        colourBuff.push_back(0);
    }
    for (int i = 0; i < numOfLines; i++)
    {
        tempLine = scanArea->getScanLine(i);
        startPoint = tempLine->getStart();
        lineLength = tempLine->getLength();
        tempStartPoint = startPoint;
        bool greenSeen = false;

        beforeColour    = ClassIndex::unclassified; //!< Colour Before the segment
        afterColour     = ClassIndex::unclassified;  //!< Colour in the next Segment
        currentColour   = ClassIndex::unclassified; //!< Colour in the current segment

        //! No point in scanning lines less then the buffer size
        if(lineLength < bufferSize+2) continue;

        for(int j = 0; j < lineLength; j = j+skipPixel)
        {
            if(direction == ScanLine::DOWN)
            {
                currentPoint.x = startPoint.x;
                currentPoint.y = startPoint.y + j;
            }
            else if (direction == ScanLine::RIGHT)
            {
                currentPoint.x = startPoint.x + j;
                currentPoint.y = startPoint.y;
            }
            else if(direction == ScanLine::UP)
            {
                currentPoint.x = startPoint.x;
                currentPoint.y = startPoint.y - j;
            }
            else if(direction == ScanLine::LEFT)
            {
                currentPoint.x = startPoint.x - j;
                currentPoint.y = startPoint.y;
            }
            //debug << currentPoint.x << " " << currentPoint.y;
            if(isPixelOnScreen(currentPoint.x,currentPoint.y) == false)
            {
                //qDebug() << "-----------------------------------------OverShoot Image:"<< currentPoint.x<< ","<<currentPoint.y;
                continue;
            }
            afterColour = classifyPixel(currentPoint.x,currentPoint.y);
            colourBuff.push_back(afterColour);

            /*qDebug() << "Scanning: " << skipPixel<<","<<j << "\t"<< currentPoint.x << "," << currentPoint.y <<
                    "\t"<<currentColour<< "," << afterColour <<
                    "\t"<< currentPoint.x << "," << currentImage->getWidth() <<
                    "\t"<< currentPoint.y << "," << currentImage->getHeight();
            */
            if(j >= lineLength - skipPixel)
            {
                //! End Of SCANLINE detected: Continue scnaning and when buffer ends or end of screen Generate new segment and add to the line
                if((currentColour == ClassIndex::green || currentColour == ClassIndex::unclassified || currentColour == ClassIndex::shadow_object))
                {
                    if(currentColour == ClassIndex::green)
                    {
                        greenSeen = true;
                    }
                    tempStartPoint = currentPoint;
                    beforeColour = ClassIndex::unclassified;
                    currentColour = afterColour;
                    for (int i = 0; i < bufferSize; i++)
                    {
                        colourBuff.push_back(0);
                    }
                    continue;
                }

                while( (currentColour == afterColour) )
                {

                    if(direction == ScanLine::DOWN)
                    {

                        if(startPoint.y + j < currentImage->getHeight())
                        {
                            currentPoint.y = startPoint.y + j;
                            currentPoint.x = startPoint.x;
                        }
                        else
                        {
                            break;
                        }
                    }
                    else if (direction == ScanLine::RIGHT)
                    {
                        if(startPoint.x + j < currentImage->getWidth())
                        {
                            currentPoint.x = startPoint.x + j;
                            currentPoint.y = startPoint.y;
                        }
                        else
                        {
                            break;
                        }

                    }
                    else if(direction == ScanLine::UP)
                    {

                        if(startPoint.y - j > 0)
                        {
                            currentPoint.y = startPoint.y - j;
                            currentPoint.x = startPoint.x;
                        }
                        else
                        {
                            break;
                        }

                    }
                    else if(direction == ScanLine::LEFT)
                    {
                        if(startPoint.x - j > 0)
                        {
                            currentPoint.x = startPoint.x - j;
                            currentPoint.y = startPoint.y;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if(isPixelOnScreen(currentPoint.x,currentPoint.y) == false)
                    {
                        //qDebug() << "-----------------------------------------OverShoot Image:"<< currentPoint.y<< ","<<currentPoint.y;
                        break;
                    }
                    afterColour = classifyPixel(currentPoint.x,currentPoint.y);
                    colourBuff.push_back(afterColour);
                    j = j+6;

                }

                TransitionSegment tempTransition(tempStartPoint, currentPoint, beforeColour, currentColour, afterColour);
                tempLine->addSegement(tempTransition);

                tempStartPoint = currentPoint;
                beforeColour = ClassIndex::unclassified;
                currentColour = afterColour;
                for (int i = 0; i < bufferSize; i++)
                {
                    colourBuff.push_back(0);
                }
                if(direction == ScanLine::DOWN)
                {
                    skipPixel = CalculateSkipSpacing(currentPoint.y,startPoint.y,greenSeen); //current point y check
                }
                else
                {
                    skipPixel = 2;
                }
                continue;
            }

            if(checkIfBufferSame(colourBuff))
            {
                if(currentColour != afterColour)
                {
                    //! Transition detected: Generate new segment and add to the line
                    //Adjust the position:
                    if(!(currentColour == ClassIndex::green || currentColour == ClassIndex::unclassified || currentColour == ClassIndex::shadow_object))
                    {
                        //SHIFTING THE POINTS TO THE START OF BUFFER:
                        if(direction == ScanLine::DOWN)
                        {
                            currentPoint.x = startPoint.x;
                            currentPoint.y = startPoint.y + j - bufferSize * skipPixel/2;
                            if(tempStartPoint.y > startPoint.y)
                            {
                                tempStartPoint.y = tempStartPoint.y - bufferSize * skipPixel/2;
                            }
                        }
                        else if (direction == ScanLine::RIGHT)
                        {
                            currentPoint.x = startPoint.x + j - bufferSize * skipPixel/2;
                            currentPoint.y = startPoint.y;
                            if(tempStartPoint.x > startPoint.x)
                            {
                                tempStartPoint.x = tempStartPoint.x - bufferSize * skipPixel/2;
                            }
                        }
                        else if(direction == ScanLine::UP)
                        {
                            currentPoint.x = startPoint.x;
                            currentPoint.y = startPoint.y - j + bufferSize * skipPixel/2;
                            if(tempStartPoint.y < startPoint.y)
                            {
                                tempStartPoint.y = tempStartPoint.y + bufferSize * skipPixel/2;
                            }
                        }
                        else if(direction == ScanLine::LEFT)
                        {
                            currentPoint.x = startPoint.x - j + bufferSize * skipPixel/2;
                            currentPoint.y = startPoint.y;
                            if(tempStartPoint.x < startPoint.x)
                            {
                                tempStartPoint.x = tempStartPoint.x + bufferSize * skipPixel/2;
                            }
                        }
                        //This rule removes
                        if(tempLine->getNumberOfSegments() == 0 && currentColour == ClassIndex::white && greenSeen == false)
                        {
                            beforeColour = currentColour;
                        }
                        TransitionSegment tempTransition(tempStartPoint, currentPoint, beforeColour, currentColour, afterColour);
                        tempLine->addSegement(tempTransition);
                    }
                    if(currentColour == ClassIndex::green)
                    {
                        greenSeen = true;
                    }
                    tempStartPoint = currentPoint;
                    beforeColour = currentColour;
                    currentColour = afterColour;
                    for (int i = 0; i < bufferSize; i++)
                    {
                        colourBuff.push_back(0);
                    }

                    if(direction == ScanLine::DOWN)
                    {
                        skipPixel = CalculateSkipSpacing(currentPoint.y,startPoint.y,greenSeen);
                    }
                    else
                    {
                        skipPixel = 2;
                    }
                }
            }
        }

    }
    return;
}

int Vision::CalculateSkipSpacing(int currentPosition, int linestartPosition, bool greenSeen)
{
    int skip = 1;
    int lengthToBottom = getImageHeight() - linestartPosition;
    if (greenSeen == false)
    {
        skip = 3;
        return skip;
    }
    if(currentPosition < linestartPosition + lengthToBottom/6)
    {
        skip = 1;
    }
    else if(currentPosition < linestartPosition + lengthToBottom/3)
    {
        skip = 2;
    }
    else
    {
        skip = 3;
    }
    return skip;

}

//! @brief  Pass a transition segment into this function, and will return a scanline which contains
//!         many different at interval of "spacing" transition segments classified in the orthogonal to the "direction"
void Vision::CloselyClassifyScanline(ScanLine* tempLine, TransitionSegment* tempTransition,int spacings, int direction, const std::vector<unsigned char> &colourList, int bufferSize)// Vector2<int> tempStartPoint, unsigned char currentColour, int length, int spacings, int direction)
{
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    int skipPixel = 2;
    if((direction == ScanLine::DOWN || direction == ScanLine::UP))
    {
        Vector2<int> StartPoint = tempTransition->getStartPoint();
        //int bufferSize = 10;
        boost::circular_buffer<unsigned char> colourBuff(bufferSize);
        for (int i = 0; i < bufferSize; i++)
        {
            colourBuff.push_back(tempTransition->getColour());
        }

        int length = abs(tempTransition->getEndPoint().y - tempTransition->getStartPoint().y);
        Vector2<int> tempSubEndPoint;
        Vector2<int> tempSubStartPoint;
        unsigned char subAfterColour;
        unsigned char subBeforeColour;
        unsigned char tempColour = tempTransition->getColour();

        for(int k = 0; k < length; k = k+spacings)
        {
            tempSubEndPoint.y   = StartPoint.y+k;
            tempSubStartPoint.y = StartPoint.y+k;
            int tempsubPoint    = StartPoint.x;
            tempColour          = tempTransition->getColour();
            //Reset Buffer: to OriginalColour
            for (int i = 0; i < bufferSize; i++)
            {
                colourBuff.push_back(tempTransition->getColour());
            }
            //qDebug() << "Condition:" << checkIfBufferSame(colourBuff) << colourBuff[0] << tempTransition->getColour();
            while(checkIfBufferContains(colourBuff,colourList))
            {
                if(tempsubPoint+skipPixel >= width)
                {
                    break;
                }

                tempsubPoint = tempsubPoint+skipPixel;
                //qDebug() << "Checking Point: " << tempsubPoint << StartPoint.y+k;
                if(StartPoint.y+k < height && StartPoint.y+k > 0
                   && tempsubPoint < width && tempsubPoint > 0)
                {

                    tempColour= classifyPixel(tempsubPoint,StartPoint.y+k);
                    colourBuff.push_back(tempColour);
                }
                else
                {
                    break;
                }
            }

            tempSubEndPoint.x = tempsubPoint - bufferSize*skipPixel;
            tempColour = tempTransition->getColour();
            while(isValidColour(tempColour,colourList))
            {
                if(StartPoint.y+k < height && StartPoint.y+k > 0
                   && tempSubEndPoint.x+1 < width && tempSubEndPoint.x+1 > 0)
                {
                    tempSubEndPoint.x = tempSubEndPoint.x+1;
                    tempColour= classifyPixel(tempSubEndPoint.x,StartPoint.y+k);
                }
                else
                {
                    break;
                }
            }
            subAfterColour = tempColour;

            //START SCANING LEFT:
            tempsubPoint = StartPoint.x;
            tempColour = tempTransition->getColour();
            //Reset Buffer: to OriginalColour
            for (int i = 0; i < bufferSize; i++)
            {
                colourBuff.push_back(tempTransition->getColour());
            }
            //qDebug() << "Condition:" << checkIfBufferSame(colourBuff) << colourBuff[0] << tempTransition->getColour();
            while(checkIfBufferContains(colourBuff, colourList))
            {
                if(tempsubPoint-skipPixel < 0)
                {
                    //tempSubStartPoint.x = 0;
                    break;
                }
                tempsubPoint = tempsubPoint - skipPixel;
                //qDebug() << "Checking Point: " << tempsubPoint << StartPoint.y+k;
                if(StartPoint.y+k < height && StartPoint.y+k > 0
                   && tempsubPoint < width && tempsubPoint > 0)
                {
                    tempColour = classifyPixel(tempsubPoint,StartPoint.y+k);
                    colourBuff.push_back(tempColour);
                }
                else
                {
                    //tempSubStartPoint.x = = tempsubPoint + bufferSize;
                    break;
                }
            }
            tempSubStartPoint.x = tempsubPoint + bufferSize*skipPixel;
            tempColour = tempTransition->getColour();
            while(isValidColour(tempColour,colourList))
            {
                if(StartPoint.y+k < height && StartPoint.y+k > 0
                   && tempSubStartPoint.x-1 < width && tempSubStartPoint.x-1 > 0)
                {
                    tempSubStartPoint.x = tempSubStartPoint.x-1;
                    tempColour= classifyPixel(tempSubStartPoint.x,StartPoint.y+k);
                }
                else
                {
                    break;
                }
            }
            subBeforeColour = tempColour;
            //THEN ADD TO LINE
            //qDebug() << "Adding Line: " << tempSubStartPoint.x << tempSubStartPoint.y << tempSubEndPoint.x << tempSubEndPoint.y;
            TransitionSegment tempTransitionA(tempSubStartPoint, tempSubEndPoint, subBeforeColour , tempTransition->getColour(), subAfterColour);
            if(tempTransitionA.getSize() >1)
            {
                tempLine->addSegement(tempTransitionA);
            }
        }
    }

    else if (direction == ScanLine::RIGHT || direction == ScanLine::LEFT)
    {
        Vector2<int> StartPoint = tempTransition->getStartPoint();

        //int bufferSize = 10;
        boost::circular_buffer<unsigned char> colourBuff(bufferSize);


        int length = abs(tempTransition->getEndPoint().x - tempTransition->getStartPoint().x);
        Vector2<int> tempSubEndPoint;
        Vector2<int> tempSubStartPoint;
        unsigned char subAfterColour;
        unsigned char subBeforeColour;
        unsigned char tempColour = tempTransition->getColour();
        for(int k = 0; k < length; k = k+spacings)
        {
            tempSubEndPoint.x = StartPoint.x+k;
            tempSubStartPoint.x = StartPoint.x+k;
            int tempY = StartPoint.y;
            tempColour = tempTransition->getColour();
            //Reseting ColourBuffer
            //qDebug() << "Resetting:";
            for (int i = 0; i < bufferSize; i++)
            {
                colourBuff.push_back(tempTransition->getColour());
            }
            //Search for End of Perpendicular Segment
            //qDebug() << "Searching roughly for end:";
            while(checkIfBufferContains(colourBuff, colourList))
            {
                if(tempY+skipPixel >= height) break;
                tempY = tempY+skipPixel;

                if(StartPoint.x+k < width && StartPoint.x+k > 0 &&
                   tempY < height && tempY > 0)
                {
                    tempColour= classifyPixel(StartPoint.x+k,tempY);
                    colourBuff.push_back(tempColour);
                }
                else
                {
                    break;
                }
            }

            tempSubEndPoint.y = tempY - bufferSize*skipPixel;
            tempColour = tempTransition->getColour();
            //qDebug() << "Searching closely for end:" ;
            while(isValidColour(tempColour,colourList))
            {
                //qDebug() << StartPoint.x+k << tempSubEndPoint.y;
                if(StartPoint.x+k < width && StartPoint.x+k > 0 &&
                   tempSubEndPoint.y+1 < height && tempSubEndPoint.y+1 > 0)
                {
                    tempSubEndPoint.y = tempSubEndPoint.y+1;
                    tempColour= classifyPixel(StartPoint.x+k,tempSubEndPoint.y);
                }
                else
                {
                    break;
                }

            }
            subAfterColour = tempColour;
            tempY = StartPoint.y;
            tempColour = tempTransition->getColour();
            //Reseting ColourBuffer
            //qDebug() << "Resetting:";
            for (int i = 0; i < bufferSize; i++)
            {
                colourBuff.push_back(tempTransition->getColour());
            }
            //qDebug() << "Searching roughly:";
            //Search for Start of Perpendicular Segment
            while(checkIfBufferContains(colourBuff, colourList))
            {
                if(tempY-skipPixel < 0)
                {
                    break;
                }
                tempY = tempY-skipPixel;
                if(StartPoint.x+k < width && StartPoint.x+k > 0
                   && tempY < height && tempY > 0)
                {
                    tempColour = classifyPixel(StartPoint.x+k,tempY);
                    //debug << tempY<< "," << (int)tempColour<< endl;
                    colourBuff.push_back(tempColour);
                }
                else
                {
                    break;
                }
            }
            tempSubStartPoint.y = tempY + bufferSize*skipPixel;
            tempColour = tempTransition->getColour();
            //qDebug() << "searching closely:";
            while(isValidColour(tempColour,colourList))
            {
                if(StartPoint.x+k < width && StartPoint.x+k > 0
                   && tempSubStartPoint.y-1 < height && tempSubStartPoint.y-1 > 0)
                {
                    tempSubStartPoint.y = tempSubStartPoint.y-1;
                    tempColour= classifyPixel(StartPoint.x+k,tempSubStartPoint.y);
                }
                else
                {
                    break;
                }
            }
            subBeforeColour = tempColour;
            //THEN ADD TO LINE


            TransitionSegment tempTransitionA(tempSubStartPoint, tempSubEndPoint, subBeforeColour , tempTransition->getColour(), subAfterColour);
            if(tempTransitionA.getSize() >1)
            {
                tempLine->addSegement(tempTransitionA);
            }
        }
    }
}

bool Vision::checkIfBufferContains(boost::circular_buffer<unsigned char> cb, const std::vector<unsigned char> &colourList)
{
    for(unsigned int i = 0; i < cb.size(); i++)
    {
        if(isValidColour(cb[i],colourList))
        {
            return true;
        }
    }
    return false;
}

std::vector<ObjectCandidate> Vision::classifyCandidates(
                                        std::vector< TransitionSegment > &segments,
                                        const std::vector<Vector2<int> >&fieldBorders,
                                        const std::vector<unsigned char> &validColours,
                                        int spacing,
                                        float min_aspect, float max_aspect, int min_segments,
                                        tCLASSIFY_METHOD method)
{
    switch(method)
    {
        case PRIMS:
            return classifyCandidatesPrims(segments, fieldBorders, validColours, spacing, min_aspect, max_aspect, min_segments);
        break;
        case DBSCAN:
            return classifyCandidatesDBSCAN(segments, fieldBorders, validColours, spacing, min_aspect, max_aspect, min_segments);
        break;
        default:
            return classifyCandidatesPrims(segments, fieldBorders, validColours, spacing,  min_aspect, max_aspect, min_segments);
        break;
    }

}

std::vector<ObjectCandidate> Vision::classifyCandidates(
                                        std::vector< TransitionSegment > &segments,
                                        const std::vector<Vector2<int> >&fieldBorders,
                                        const std::vector<unsigned char> &validColours,
                                        int spacing,
                                        float min_aspect, float max_aspect, int min_segments, std::vector< TransitionSegment > &leftover)
{
    return classifyCandidatesPrims(segments, fieldBorders, validColours, spacing, min_aspect, max_aspect, min_segments, leftover);
}

std::vector<ObjectCandidate> Vision::classifyCandidatesPrims(std::vector< TransitionSegment > &segments,
                                        const std::vector<Vector2<int> >&fieldBorders,
                                        const std::vector<unsigned char> &validColours,
                                        int spacing,
                                        float min_aspect, float max_aspect, int min_segments)
{
    //! Overall runtime O( N^2 )
    std::vector<ObjectCandidate> candidateList;

    const int VERT_JOIN_LIMIT = 3;
    const int HORZ_JOIN_LIMIT = 1;


    if (!segments.empty())
    {
        //! Sorting O(N*logN)
        sort(segments.begin(), segments.end(), Vision::sortTransitionSegments);

        std::queue<int> qUnprocessed;
        std::vector<TransitionSegment> candidate_segments;
        std::vector<unsigned int> usedSegments;
        unsigned int rawSegsLeft = segments.size();
        unsigned int nextRawSeg = 0;

        bool isSegUsed [segments.size()];
        usedSegments.clear();
        //! Removing invalid colours O(N)
        for (unsigned int i = 0; i < segments.size(); i++)
        {
            //may have non-robot colour segments, so pre-mark them as used
            if (isValidColour(segments.at(i).getColour(), validColours))
            {
                //qDebug() << ClassIndex::getColourNameFromIndex(segments.at(i).getColour()) << isRobotColour(segments.at(i).getColour());
                isSegUsed[i] = false;
                //qDebug() <<  "(" << segments.at(i).getStartPoint().x << "," << segments.at(i).getStartPoint().y << ")-("<< segments.at(i).getEndPoint().x << "," << segments.at(i).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(i).getColour()) << "]";
            }
            else
            {
                rawSegsLeft--;
                isSegUsed[i] = true;
            }

        }

        //! For all valid segments O(M)
        while(rawSegsLeft)
        {

            //Roll through and find first unused segment
            nextRawSeg = 0;

            //! Find next unused segment O(M)
            while(isSegUsed[nextRawSeg] && nextRawSeg < segments.size()) nextRawSeg++;
            //Prime unprocessed segment queue to build next candidate
            qUnprocessed.push(nextRawSeg);
            //take away from pool of raw segs
            isSegUsed[nextRawSeg] = true;
            rawSegsLeft--;

            int min_x, max_x, min_y, max_y, segCount;
            int * colourHistogram = new int[validColours.size()];
            min_x = segments.at(nextRawSeg).getStartPoint().x;
            max_x = segments.at(nextRawSeg).getStartPoint().x;
            min_y = segments.at(nextRawSeg).getStartPoint().y;
            max_y = segments.at(nextRawSeg).getEndPoint().y;
            segCount = 0;
            for (unsigned int i = 0; i < validColours.size(); i++)  colourHistogram[i] = 0;

            //! For all unprocessed joined segment in a candidate O(M)
            //Build candidate

            candidate_segments.clear();

            while (!qUnprocessed.empty())
            {
                unsigned int thisSeg;
                thisSeg = qUnprocessed.front();
                qUnprocessed.pop();
                segCount++;
                for (unsigned int i = 0; i < validColours.size(); i++)
                {
                    if ( segments.at(thisSeg).getColour() == validColours.at(i) && validColours.at(i) != ClassIndex::white)
                    {
                        colourHistogram[i] += 1;
                        i = validColours.size();
                    }
                }

                if ( min_x > segments.at(thisSeg).getStartPoint().x)
                    min_x = segments.at(thisSeg).getStartPoint().x;
                if ( max_x < segments.at(thisSeg).getStartPoint().x)
                    max_x = segments.at(thisSeg).getStartPoint().x;
                if ( min_y > segments.at(thisSeg).getStartPoint().y)
                    min_y = segments.at(thisSeg).getStartPoint().y;
                if ( max_y < segments.at(thisSeg).getEndPoint().y)
                    max_y = segments.at(thisSeg).getEndPoint().y;



                //if there is a seg above AND 'close enough', then qUnprocessed->push()
                if ( thisSeg > 0 &&
                     !isSegUsed[thisSeg-1] &&
                     segments.at(thisSeg).getStartPoint().x == segments.at(thisSeg-1).getStartPoint().x &&
                     segments.at(thisSeg).getStartPoint().y - segments.at(thisSeg-1).getEndPoint().y < VERT_JOIN_LIMIT)
                {
                    //qDebug() << "Up   Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thisSeg-1)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg-1).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thisSeg-1).getStartPoint().x << "," << segments.at(thisSeg-1).getStartPoint().y << ")-("<< segments.at(thisSeg-1).getEndPoint().x << "," << segments.at(thisSeg-1).getEndPoint().y << ")";
                    qUnprocessed.push(thisSeg-1);
                    //take away from pool of raw segs
                    isSegUsed[thisSeg-1] = true;
                    rawSegsLeft--;
                }

                //if there is a seg below AND 'close enough', then qUnprocessed->push()
                if ( thisSeg+1 < segments.size() &&
                     !isSegUsed[thisSeg+1] &&
                     segments.at(thisSeg).getStartPoint().x == segments.at(thisSeg+1).getStartPoint().x &&
                     segments.at(thisSeg+1).getStartPoint().y - segments.at(thisSeg).getEndPoint().y < VERT_JOIN_LIMIT)
                {
                    //qDebug() << "Down Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thisSeg+1)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg+1).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thisSeg+1).getStartPoint().x << "," << segments.at(thisSeg+1).getStartPoint().y << ")-("<< segments.at(thisSeg+1).getEndPoint().x << "," << segments.at(thisSeg+1).getEndPoint().y << ")";
                    qUnprocessed.push(thisSeg+1);
                    //take away from pool of raw segs
                    isSegUsed[thisSeg+1] = true;
                    rawSegsLeft--;
                }

                //! For each segment being processed in a candidate to the RIGHT attempt to join segments within range O(M)
                //if there is a seg overlapping on the right AND 'close enough', then qUnprocessed->push()
                for (unsigned int thatSeg = thisSeg + 1; thatSeg < segments.size(); thatSeg++)
                {
                    if ( segments.at(thatSeg).getStartPoint().x - segments.at(thisSeg).getStartPoint().x <=  spacing*HORZ_JOIN_LIMIT)
                    {
                        if ( segments.at(thatSeg).getStartPoint().x > segments.at(thisSeg).getStartPoint().x &&
                             !isSegUsed[thatSeg])
                        {
                            //NOT in same column as thisSeg and is to the right
                            if ( segments.at(thatSeg).getStartPoint().y <= segments.at(thisSeg).getEndPoint().y &&
                                 segments.at(thisSeg).getStartPoint().y <= segments.at(thatSeg).getEndPoint().y)
                            {
                                //thisSeg overlaps with thatSeg
                                //qDebug() <<  "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << "]::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << "]";
                                //! Find intercept O(K), K is number of field border points
                                int intercept = findInterceptFromPerspectiveFrustum(fieldBorders,
                                                                                    segments.at(thisSeg).getStartPoint().x,
                                                                                    segments.at(thatSeg).getStartPoint().x,
                                                                                    spacing*HORZ_JOIN_LIMIT);
                                if ( intercept >= 0 &&
                                     segments.at(thatSeg).getEndPoint().y >= intercept &&
                                     intercept <= segments.at(thisSeg).getEndPoint().y)
                                {
                                    //within HORZ_JOIN_LIMIT
                                    //qDebug() << "Right Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thatSeg)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")";
                                    qUnprocessed.push(thatSeg);
                                    //take away from pool of raw segs
                                    isSegUsed[thatSeg] = true;
                                    rawSegsLeft--;

                                }
                                else
                                {
                                    //qDebug() << "|" << float(segments.at(thatSeg).getEndPoint().y) <<  "thatSeg End(y)>= intercept" << intercept << "|";
                                    //qDebug() << "|" << int(intercept) << "intercept <= thisSeg End(y)" << segments.at(thisSeg).getEndPoint().y << "|";
                                }
                            }
                        }
                    }
                    else
                    {
                        thatSeg = segments.size();
                    }
                }

                //! For each segment being processed in a candidate to the LEFT attempt to join segments within range O(M)
                //if there is a seg overlapping on the left AND 'close enough', then qUnprocessed->push()
                for (int thatSeg = thisSeg - 1; thatSeg >= 0; thatSeg--)
                {
                    if ( segments.at(thisSeg).getStartPoint().x - segments.at(thatSeg).getStartPoint().x <=  spacing*HORZ_JOIN_LIMIT)
                    {

                        if ( !isSegUsed[thatSeg] &&
                             segments.at(thatSeg).getStartPoint().x < segments.at(thisSeg).getStartPoint().x)
                        {
                            //NOT in same column as thisSeg and is to the right
                            if ( segments.at(thatSeg).getStartPoint().y <= segments.at(thisSeg).getEndPoint().y &&
                                 segments.at(thisSeg).getStartPoint().y <= segments.at(thatSeg).getEndPoint().y)
                            {
                                //thisSeg overlaps with thatSeg
                                //qDebug() <<  "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << "]::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << "]";
                                //! Find intercept O(K), K is number of field border points
                                int intercept = findInterceptFromPerspectiveFrustum(fieldBorders,
                                                                                    segments.at(thisSeg).getStartPoint().x,
                                                                                    segments.at(thatSeg).getStartPoint().x,
                                                                                    spacing*HORZ_JOIN_LIMIT);

                                if ( intercept >= 0 &&
                                     segments.at(thatSeg).getEndPoint().y >= intercept &&
                                     intercept <= segments.at(thisSeg).getEndPoint().y)
                                {
                                    //within HORZ_JOIN_LIMIT
                                    //qDebug() << "Left Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thatSeg)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")";
                                    qUnprocessed.push(thatSeg);
                                    //take away from pool of raw segs
                                    isSegUsed[thatSeg] = true;
                                    rawSegsLeft--;
                                }
                                else
                                {
                                    //qDebug() << "|" << float(segments.at(thatSeg).getEndPoint().y) <<  "thatSeg End(y)>= intercept" << intercept << "|";
                                    //qDebug() << "|" << int(intercept) << "intercept <= thisSeg End(y)" << segments.at(thisSeg).getEndPoint().y << "|";
                                }
                            }
                        }
                    }
                    else
                    {
                        thatSeg = -1;
                    }
                }

                //add thisSeg to CandidateVector
                candidate_segments.push_back(segments.at(thisSeg));
                segments.at(thisSeg).isUsed = true;
                usedSegments.push_back(thisSeg);
            }//while (!qUnprocessed->empty())
            //qDebug() << "Candidate ready...";
            //HEURISTICS FOR ADDING THIS CANDIDATE AS A ROBOT CANDIDATE
            if ( max_x - min_x >= 0 &&                                               // width  is non-zero
                 max_y - min_y >= 0 &&                                               // height is non-zero
                 (float)(max_x - min_x) / (float)(max_y - min_y) <= max_aspect &&    // Less    than specified landscape aspect
                 (float)(max_x - min_x) / (float)(max_y - min_y) >= min_aspect &&    // greater than specified portrait aspect
                 segCount >= min_segments                                    // greater than minimum amount of segments to remove noise
                 )
            {
                //qDebug() << "CANDIDATE FINISHED::" << segCount << " segments, aspect:" << ( (float)(max_x - min_x) / (float)(max_y - min_y)) << ", Coords:(" << min_x << "," << min_y << ")-(" << max_x << "," << max_y << "), width: " << (max_x - min_x) << ", height: " << (max_y - min_y);
                int max_col = 0;
                for (int i = 0; i < (int)validColours.size(); i++)
                {
                    if (i != max_col && colourHistogram[i] > colourHistogram[max_col])
                        max_col = i;
                }
                for(unsigned int i=0; i<candidate_segments.size(); i++) {
                    candidate_segments[i].isUsed = true;
                }
                ObjectCandidate temp(min_x, min_y, max_x, max_y, validColours.at(max_col), candidate_segments);
                candidateList.push_back(temp);
                usedSegments.clear();
            }
            else {
                while(!usedSegments.empty()){
                    segments[usedSegments.back()].isUsed = false;
                    usedSegments.pop_back();
                }
            }
        delete [] colourHistogram;
        }//while(rawSegsLeft)

    }//if (!segments.empty())
    return candidateList;
}

std::vector<ObjectCandidate> Vision::classifyCandidatesPrims(std::vector< TransitionSegment > &segments,
                                        const std::vector<Vector2<int> >&fieldBorders,
                                        const std::vector<unsigned char> &validColours,
                                        int spacing,
                                        float min_aspect, float max_aspect, int min_segments,
                                        std::vector< TransitionSegment >& leftover)
{
    //! Overall runtime O( N^2 )
    std::vector<ObjectCandidate> candidateList;

    const int VERT_JOIN_LIMIT = 3;
    const int HORZ_JOIN_LIMIT = 1;


    if (!segments.empty())
    {
        //! Sorting O(N*logN)
        sort(segments.begin(), segments.end(), Vision::sortTransitionSegments);

        std::queue<int> qUnprocessed;
        std::vector<TransitionSegment> candidate_segments;
        std::vector<unsigned int> usedSegments;
        unsigned int rawSegsLeft = segments.size();
        unsigned int nextRawSeg = 0;

        bool isSegUsed [segments.size()];
        usedSegments.clear();
        //! Removing invalid colours O(N)
        for (unsigned int i = 0; i < segments.size(); i++)
        {
            //may have non-robot colour segments, so pre-mark them as used
            if (isValidColour(segments.at(i).getColour(), validColours))
            {
                //qDebug() << ClassIndex::getColourNameFromIndex(segments.at(i).getColour()) << isRobotColour(segments.at(i).getColour());
                isSegUsed[i] = false;
                //qDebug() <<  "(" << segments.at(i).getStartPoint().x << "," << segments.at(i).getStartPoint().y << ")-("<< segments.at(i).getEndPoint().x << "," << segments.at(i).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(i).getColour()) << "]";
            }
            else
            {
                rawSegsLeft--;
                isSegUsed[i] = true;
            }

        }

        //! For all valid segments O(M)
        while(rawSegsLeft)
        {

            //Roll through and find first unused segment
            nextRawSeg = 0;

            //! Find next unused segment O(M)
            while(isSegUsed[nextRawSeg] && nextRawSeg < segments.size()) nextRawSeg++;
            //Prime unprocessed segment queue to build next candidate
            qUnprocessed.push(nextRawSeg);
            //take away from pool of raw segs
            isSegUsed[nextRawSeg] = true;
            rawSegsLeft--;

            int min_x, max_x, min_y, max_y, segCount;
            int * colourHistogram = new int[validColours.size()];
            min_x = segments.at(nextRawSeg).getStartPoint().x;
            max_x = segments.at(nextRawSeg).getStartPoint().x;
            min_y = segments.at(nextRawSeg).getStartPoint().y;
            max_y = segments.at(nextRawSeg).getEndPoint().y;
            segCount = 0;
            for (unsigned int i = 0; i < validColours.size(); i++)  colourHistogram[i] = 0;

            //! For all unprocessed joined segment in a candidate O(M)
            //Build candidate

            candidate_segments.clear();

            while (!qUnprocessed.empty())
            {
                unsigned int thisSeg;
                thisSeg = qUnprocessed.front();
                qUnprocessed.pop();
                segCount++;
                for (unsigned int i = 0; i < validColours.size(); i++)
                {
                    if ( segments.at(thisSeg).getColour() == validColours.at(i) && validColours.at(i) != ClassIndex::white)
                    {
                        colourHistogram[i] += 1;
                        i = validColours.size();
                    }
                }

                if ( min_x > segments.at(thisSeg).getStartPoint().x)
                    min_x = segments.at(thisSeg).getStartPoint().x;
                if ( max_x < segments.at(thisSeg).getStartPoint().x)
                    max_x = segments.at(thisSeg).getStartPoint().x;
                if ( min_y > segments.at(thisSeg).getStartPoint().y)
                    min_y = segments.at(thisSeg).getStartPoint().y;
                if ( max_y < segments.at(thisSeg).getEndPoint().y)
                    max_y = segments.at(thisSeg).getEndPoint().y;



                //if there is a seg above AND 'close enough', then qUnprocessed->push()
                if ( thisSeg > 0 &&
                     !isSegUsed[thisSeg-1] &&
                     segments.at(thisSeg).getStartPoint().x == segments.at(thisSeg-1).getStartPoint().x &&
                     segments.at(thisSeg).getStartPoint().y - segments.at(thisSeg-1).getEndPoint().y < VERT_JOIN_LIMIT)
                {
                    //qDebug() << "Up   Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thisSeg-1)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg-1).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thisSeg-1).getStartPoint().x << "," << segments.at(thisSeg-1).getStartPoint().y << ")-("<< segments.at(thisSeg-1).getEndPoint().x << "," << segments.at(thisSeg-1).getEndPoint().y << ")";
                    qUnprocessed.push(thisSeg-1);
                    //take away from pool of raw segs
                    isSegUsed[thisSeg-1] = true;
                    rawSegsLeft--;
                }

                //if there is a seg below AND 'close enough', then qUnprocessed->push()
                if ( thisSeg+1 < segments.size() &&
                     !isSegUsed[thisSeg+1] &&
                     segments.at(thisSeg).getStartPoint().x == segments.at(thisSeg+1).getStartPoint().x &&
                     segments.at(thisSeg+1).getStartPoint().y - segments.at(thisSeg).getEndPoint().y < VERT_JOIN_LIMIT)
                {
                    //qDebug() << "Down Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thisSeg+1)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg+1).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thisSeg+1).getStartPoint().x << "," << segments.at(thisSeg+1).getStartPoint().y << ")-("<< segments.at(thisSeg+1).getEndPoint().x << "," << segments.at(thisSeg+1).getEndPoint().y << ")";
                    qUnprocessed.push(thisSeg+1);
                    //take away from pool of raw segs
                    isSegUsed[thisSeg+1] = true;
                    rawSegsLeft--;
                }

                //! For each segment being processed in a candidate to the RIGHT attempt to join segments within range O(M)
                //if there is a seg overlapping on the right AND 'close enough', then qUnprocessed->push()
                for (unsigned int thatSeg = thisSeg + 1; thatSeg < segments.size(); thatSeg++)
                {
                    if ( segments.at(thatSeg).getStartPoint().x - segments.at(thisSeg).getStartPoint().x <=  spacing*HORZ_JOIN_LIMIT)
                    {
                        if ( segments.at(thatSeg).getStartPoint().x > segments.at(thisSeg).getStartPoint().x &&
                             !isSegUsed[thatSeg])
                        {
                            //NOT in same column as thisSeg and is to the right
                            if ( segments.at(thatSeg).getStartPoint().y <= segments.at(thisSeg).getEndPoint().y &&
                                 segments.at(thisSeg).getStartPoint().y <= segments.at(thatSeg).getEndPoint().y)
                            {
                                //thisSeg overlaps with thatSeg
                                //qDebug() <<  "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << "]::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << "]";
                                //! Find intercept O(K), K is number of field border points
                                int intercept = findInterceptFromPerspectiveFrustum(fieldBorders,
                                                                                    segments.at(thisSeg).getStartPoint().x,
                                                                                    segments.at(thatSeg).getStartPoint().x,
                                                                                    spacing*HORZ_JOIN_LIMIT);
                                if ( intercept >= 0 &&
                                     segments.at(thatSeg).getEndPoint().y >= intercept &&
                                     intercept <= segments.at(thisSeg).getEndPoint().y)
                                {
                                    //within HORZ_JOIN_LIMIT
                                    //qDebug() << "Right Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thatSeg)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")";
                                    qUnprocessed.push(thatSeg);
                                    //take away from pool of raw segs
                                    isSegUsed[thatSeg] = true;
                                    rawSegsLeft--;

                                }
                                else
                                {
                                    //qDebug() << "|" << float(segments.at(thatSeg).getEndPoint().y) <<  "thatSeg End(y)>= intercept" << intercept << "|";
                                    //qDebug() << "|" << int(intercept) << "intercept <= thisSeg End(y)" << segments.at(thisSeg).getEndPoint().y << "|";
                                }
                            }
                        }
                    }
                    else
                    {
                        thatSeg = segments.size();
                    }
                }

                //! For each segment being processed in a candidate to the LEFT attempt to join segments within range O(M)
                //if there is a seg overlapping on the left AND 'close enough', then qUnprocessed->push()
                for (int thatSeg = thisSeg - 1; thatSeg >= 0; thatSeg--)
                {
                    if ( segments.at(thisSeg).getStartPoint().x - segments.at(thatSeg).getStartPoint().x <=  spacing*HORZ_JOIN_LIMIT)
                    {

                        if ( !isSegUsed[thatSeg] &&
                             segments.at(thatSeg).getStartPoint().x < segments.at(thisSeg).getStartPoint().x)
                        {
                            //NOT in same column as thisSeg and is to the right
                            if ( segments.at(thatSeg).getStartPoint().y <= segments.at(thisSeg).getEndPoint().y &&
                                 segments.at(thisSeg).getStartPoint().y <= segments.at(thatSeg).getEndPoint().y)
                            {
                                //thisSeg overlaps with thatSeg
                                //qDebug() <<  "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << "]::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")[" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << "]";
                                //! Find intercept O(K), K is number of field border points
                                int intercept = findInterceptFromPerspectiveFrustum(fieldBorders,
                                                                                    segments.at(thisSeg).getStartPoint().x,
                                                                                    segments.at(thatSeg).getStartPoint().x,
                                                                                    spacing*HORZ_JOIN_LIMIT);

                                if ( intercept >= 0 &&
                                     segments.at(thatSeg).getEndPoint().y >= intercept &&
                                     intercept <= segments.at(thisSeg).getEndPoint().y)
                                {
                                    //within HORZ_JOIN_LIMIT
                                    //qDebug() << "Left Join Seg: " << thisSeg << "(" << ClassIndex::getColourNameFromIndex(segments.at(thisSeg).getColour()) << ") U " << (thatSeg)<< "(" << ClassIndex::getColourNameFromIndex(segments.at(thatSeg).getColour()) << ")" << "(" << segments.at(thisSeg).getStartPoint().x << "," << segments.at(thisSeg).getStartPoint().y << ")-("<< segments.at(thisSeg).getEndPoint().x << "," << segments.at(thisSeg).getEndPoint().y << ")::(" << segments.at(thatSeg).getStartPoint().x << "," << segments.at(thatSeg).getStartPoint().y << ")-("<< segments.at(thatSeg).getEndPoint().x << "," << segments.at(thatSeg).getEndPoint().y << ")";
                                    qUnprocessed.push(thatSeg);
                                    //take away from pool of raw segs
                                    isSegUsed[thatSeg] = true;
                                    rawSegsLeft--;
                                }
                                else
                                {
                                    //qDebug() << "|" << float(segments.at(thatSeg).getEndPoint().y) <<  "thatSeg End(y)>= intercept" << intercept << "|";
                                    //qDebug() << "|" << int(intercept) << "intercept <= thisSeg End(y)" << segments.at(thisSeg).getEndPoint().y << "|";
                                }
                            }
                        }
                    }
                    else
                    {
                        thatSeg = -1;
                    }
                }

                //add thisSeg to CandidateVector
                candidate_segments.push_back(segments.at(thisSeg));
                segments.at(thisSeg).isUsed = true;
                usedSegments.push_back(thisSeg);
            }//while (!qUnprocessed->empty())
            //qDebug() << "Candidate ready...";
            //HEURISTICS FOR ADDING THIS CANDIDATE AS A ROBOT CANDIDATE
            if ( max_x - min_x >= 0 &&                                               // width  is non-zero
                 max_y - min_y >= 0 &&                                               // height is non-zero
                 (float)(max_x - min_x) / (float)(max_y - min_y) <= max_aspect &&    // Less    than specified landscape aspect
                 (float)(max_x - min_x) / (float)(max_y - min_y) >= min_aspect &&    // greater than specified portrait aspect
                 segCount >= min_segments                                    // greater than minimum amount of segments to remove noise
                 )
            {
                //qDebug() << "CANDIDATE FINISHED::" << segCount << " segments, aspect:" << ( (float)(max_x - min_x) / (float)(max_y - min_y)) << ", Coords:(" << min_x << "," << min_y << ")-(" << max_x << "," << max_y << "), width: " << (max_x - min_x) << ", height: " << (max_y - min_y);
                int max_col = 0;
                for (int i = 0; i < (int)validColours.size(); i++)
                {
                    if (i != max_col && colourHistogram[i] > colourHistogram[max_col])
                        max_col = i;
                }
                for(unsigned int i=0; i<candidate_segments.size(); i++) {
                    candidate_segments[i].isUsed = true;
                }
                ObjectCandidate temp(min_x, min_y, max_x, max_y, validColours.at(max_col), candidate_segments);
                candidateList.push_back(temp);
                usedSegments.clear();
            }
            else {
                while(!usedSegments.empty()){
                    segments[usedSegments.back()].isUsed = false;
                    leftover.push_back( segments[usedSegments.back()] );
                    usedSegments.pop_back();
                }
            }
            delete [] colourHistogram;
        }//while(rawSegsLeft)

    }//if (!segments.empty())
    return candidateList;
}

std::vector<ObjectCandidate> Vision::classifyCandidatesDBSCAN(std::vector< TransitionSegment > &segments,
                                        const std::vector<Vector2<int> >&fieldBorders,
                                        const std::vector<unsigned char> &validColours,
                                        int spacing,
                                        float min_aspect, float max_aspect, int min_segments)
{
    std::vector<ObjectCandidate> candidateList;

    //unimplemented

    return candidateList;
}

bool Vision::isValidColour(unsigned char colour, const std::vector<unsigned char> &colourList)
{
    bool result = false;
    if (colourList.size())
    {
        std::vector<unsigned char>::const_iterator nextCol = colourList.begin();
        for (;nextCol != colourList.end(); nextCol++)
        {
            if (colour == *(nextCol.base()))
            {
                result = true;
                break;
            }
        }
    }
    return result;
}

int Vision::findYFromX(const std::vector<Vector2<int> >&points, int x)
{

    int y = 0;
    int left_x = -1;
    int left_y = 0;
    int right_x = -1;
    int right_y = 0;
    std::vector< Vector2<int> >::const_iterator nextPoint = points.begin();

    for(; nextPoint != points.end(); nextPoint++)
    {
        if (nextPoint->x == x)
        {
            return nextPoint->y;
        }

        if (left_x < nextPoint->x && nextPoint->x < x)
        {
            left_x = nextPoint->x;
            left_y = nextPoint->y;
        }

        if ( (right_x > nextPoint->x || right_x < 0) && nextPoint->x > x)
        {
            right_x = nextPoint->x;
            right_y = nextPoint->y;
        }

    }
    //qDebug() << "findYFromX" << y;
    if (right_x - left_x > 0)
        y = left_y + (right_y-left_y) * (x-left_x) / (right_x-left_x);
    //qDebug() << "findYFromX" << y;
    return y;
}

int Vision::findInterceptFromPerspectiveFrustum(const std::vector<Vector2<int> >&points, int current_x, int target_x, int spacing)
{
    int height = currentImage->getHeight();

    int intercept = 0;
    if (current_x == target_x)
    {
        //qDebug() << "Intercept -1 =";
        return -1;
    }

    int y = findYFromX(points, current_x);
    int diff_x = 0;
    int diff_y = height - y;

    if (current_x < target_x)
    {
        diff_x = target_x - current_x;
    }

    if (target_x < current_x)
    {
        diff_x = current_x - target_x;
    }

    if (diff_x > spacing)
    {
        intercept = height;
    }
    else if ( diff_x > spacing/2)
    {
        intercept = y + diff_y/2;
    }
    else if ( diff_x > spacing/4)
    {
        intercept = y + diff_y/4;
    }
    else
    {
        intercept = y;
    }

    //qDebug() << "findInterceptFromPerspectiveFrustum intercept:"<<intercept<<" {y:"<< y << ", height:" << currentImage->height() << ", spacing:" << spacing << ", target_x:" << target_x << ", current_x:" << current_x << "}";
    if (intercept > height)
        intercept = -2;
    return intercept;
}

bool Vision::sortTransitionSegments(TransitionSegment a, TransitionSegment b)
{
    return (a.getStartPoint().x < b.getStartPoint().x || (a.getStartPoint().x == b.getStartPoint().x && a.getEndPoint().y <= b.getStartPoint().y));
}

bool Vision::checkIfBufferSame(boost::circular_buffer<unsigned char> cb)
{

    unsigned char currentClass = cb[0];
    for (unsigned int i = 1; i < cb.size(); i++)
    {
        if(cb[i] != currentClass)
        {
            return false;
        }
    }
    return true;

}

std::vector< ObjectCandidate > Vision::ClassifyCandidatesAboveTheHorizon(   std::vector< TransitionSegment > &horizontalsegments,
                                                                            const std::vector<unsigned char> &validColours,
                                                                            int spacing,
                                                                            int min_segments)
{
    std::vector< ObjectCandidate > candidates;
    std::vector< TransitionSegment > tempSegments;
    tempSegments.reserve(horizontalsegments.size());
    candidates.reserve(horizontalsegments.size());

    bool usedSegments[horizontalsegments.size()];
    //qDebug() << "Classify Above the horizon" << horizontalsegments.size();


    for (int i = 0; i < (int)horizontalsegments.size(); i++)
    {
        usedSegments[i] = false;
    }

    int Xstart, Xend, Ystart, Yend;
    //Work Backwards: As post width is acurrate at bottom (no crossbar)
    //ASSUMING EVERYTHING IS ALREADY ORDERED
    for(int i = horizontalsegments.size()-1; i > 0; i--)
    {
        tempSegments.clear();
        std::vector<int> tempUsedSegments;
        tempUsedSegments.reserve(horizontalsegments.size());
        if(!isValidColour(horizontalsegments[i].getColour(), validColours))
        {
            continue;
        }
        //Setting up:
        if(usedSegments[i] != false)
        {
            continue;
        }
        Ystart = horizontalsegments[i].getStartPoint().y;
        Yend = horizontalsegments[i].getEndPoint().y;
        Xstart = horizontalsegments[i].getStartPoint().x;
        Xend = horizontalsegments[i].getEndPoint().x;
        tempSegments.push_back(horizontalsegments[i]);
        horizontalsegments[i].isUsed = true;
        tempUsedSegments.push_back(i);
        int nextSegCounter = i-1;

        //We want to stop searching when it leaves the line
        //Searching for a new Xend, close to this current Xstart
        //Then update with new xstart
        //qDebug() << "While:";
        while(horizontalsegments[nextSegCounter].getStartPoint().y ==  Yend && nextSegCounter >0)
        {
            if(usedSegments[nextSegCounter] != false)
            {
                nextSegCounter--;
                continue;
            }
            if(!isValidColour(horizontalsegments[nextSegCounter].getColour(), validColours))
            {
                nextSegCounter--;
                continue;
            }
            if(horizontalsegments[nextSegCounter].getEndPoint().x     <= Xstart - spacing
               && horizontalsegments[nextSegCounter].getEndPoint().x  >= Xstart + spacing)
            {
                //Update with new info
                tempSegments.push_back(horizontalsegments[nextSegCounter]);
                horizontalsegments[nextSegCounter].isUsed = true;
                tempUsedSegments.push_back(nextSegCounter);
                if (horizontalsegments[nextSegCounter].getStartPoint().x <= Xstart)
                {
                    Xstart = horizontalsegments[nextSegCounter].getStartPoint().x;
                }

            }
            nextSegCounter--;
        }
        //Once all the segments on same line has been found scan between
        //(Xstart-spacing) and (Xend+spacing) for segments that are likely to be above the
        //Starting from your where you left off in horzontal count
        //qDebug() << "for:" << nextSegCounter;
        for(int j = nextSegCounter; j > 0; j--)
        {
            if(usedSegments[j] != false)
            {
                continue;
            }
            if(!isValidColour(horizontalsegments[j].getColour(), validColours))
            {
                continue;
            }
            if(horizontalsegments[j].getStartPoint().y < Ystart - spacing*4)
            {
                break;
            }
            if(horizontalsegments[j].getStartPoint().x   >= Xstart - spacing/2
               && horizontalsegments[j].getEndPoint().x  <= Xend + spacing/2)
            {
                if (horizontalsegments[j].getStartPoint().x < Xstart)
                {
                    Xstart = horizontalsegments[j].getStartPoint().x;
                }
                if (horizontalsegments[j].getEndPoint().x > Xend)
                {
                    Xend = horizontalsegments[j].getEndPoint().x;
                }
                if(horizontalsegments[j].getStartPoint().y < Ystart)
                {
                    Ystart = horizontalsegments[j].getStartPoint().y;
                }
                tempSegments.push_back(horizontalsegments[j]);
                horizontalsegments[j].isUsed = true;
                tempUsedSegments.push_back(j);
            }

        }
        //qDebug() << "About: Creating candidate: " << Xstart << ","<< Ystart<< ","<< Xend<< ","<< Yend << " Size: " << tempSegments.size();
        //Create Object Candidate if greater then the minimum number of segments
        if((int)tempSegments.size() >= min_segments)
        {
            //qDebug() << "Creating candidate: " << Xstart << ","<< Ystart<< ","<< Xend<< ","<< Yend << " Size: " << tempSegments.size();
            //for(size_t i =0; i < tempSegments.size(); i++)
            //{
                //qDebug() << tempSegments[i].getStartPoint().x << "," <<tempSegments[i].getStartPoint().y << " " << tempSegments[i].getEndPoint().x << "," <<tempSegments[i].getEndPoint().y;
            //}
            ObjectCandidate tempCandidate(Xstart, Ystart, Xend, Yend, validColours[0], tempSegments);
            candidates.push_back(tempCandidate);
            while (!tempUsedSegments.empty())
            {
                usedSegments[tempUsedSegments.back()] = true;
                tempUsedSegments.pop_back();
            }
        }
        else {
            //reset segments to not used
            while (!tempUsedSegments.empty())
            {
                horizontalsegments[tempUsedSegments.back()].isUsed = false;
                tempUsedSegments.pop_back();
            }
        }

    }
    //qDebug() << "candidate size: " << candidates.size();
    return candidates;
}

std::vector< ObjectCandidate > Vision::ClassifyCandidatesAboveTheHorizon(   std::vector< TransitionSegment > &horizontalsegments,
                                                                            const std::vector<unsigned char> &validColours,
                                                                            int spacing,
                                                                            int min_segments,
                                                                            std::vector< TransitionSegment > &leftover)
{
    std::vector< ObjectCandidate > candidates;
    std::vector< TransitionSegment > tempSegments;
    tempSegments.reserve(horizontalsegments.size());
    candidates.reserve(horizontalsegments.size());

    bool usedSegments[horizontalsegments.size()];
    //qDebug() << "Classify Above the horizon" << horizontalsegments.size();


    for (int i = 0; i < (int)horizontalsegments.size(); i++)
    {
        usedSegments[i] = false;
    }

    int Xstart, Xend, Ystart, Yend;
    //Work Backwards: As post width is acurrate at bottom (no crossbar)
    //ASSUMING EVERYTHING IS ALREADY ORDERED
    for(int i = horizontalsegments.size()-1; i > 0; i--)
    {
        tempSegments.clear();
        std::vector<int> tempUsedSegments;
        tempUsedSegments.reserve(horizontalsegments.size());
        if(!isValidColour(horizontalsegments[i].getColour(), validColours))
        {
            continue;
        }
        //Setting up:
        if(usedSegments[i] != false)
        {
            continue;
        }
        Ystart = horizontalsegments[i].getStartPoint().y;
        Yend = horizontalsegments[i].getEndPoint().y;
        Xstart = horizontalsegments[i].getStartPoint().x;
        Xend = horizontalsegments[i].getEndPoint().x;
        tempSegments.push_back(horizontalsegments[i]);
        horizontalsegments[i].isUsed = true;
        tempUsedSegments.push_back(i);
        int nextSegCounter = i-1;

        //We want to stop searching when it leaves the line
        //Searching for a new Xend, close to this current Xstart
        //Then update with new xstart
        //qDebug() << "While:";
        while(horizontalsegments[nextSegCounter].getStartPoint().y ==  Yend && nextSegCounter >0)
        {
            if(usedSegments[nextSegCounter] != false)
            {
                nextSegCounter--;
                continue;
            }
            if(!isValidColour(horizontalsegments[nextSegCounter].getColour(), validColours))
            {
                nextSegCounter--;
                continue;
            }
            if(horizontalsegments[nextSegCounter].getEndPoint().x     <= Xstart - spacing
               && horizontalsegments[nextSegCounter].getEndPoint().x  >= Xstart + spacing)
            {
                //Update with new info
                tempSegments.push_back(horizontalsegments[nextSegCounter]);
                horizontalsegments[nextSegCounter].isUsed = true;
                tempUsedSegments.push_back(nextSegCounter);
                if (horizontalsegments[nextSegCounter].getStartPoint().x <= Xstart)
                {
                    Xstart = horizontalsegments[nextSegCounter].getStartPoint().x;
                }

            }
            nextSegCounter--;
        }
        //Once all the segments on same line has been found scan between
        //(Xstart-spacing) and (Xend+spacing) for segments that are likely to be above the
        //Starting from your where you left off in horzontal count
        //qDebug() << "for:" << nextSegCounter;
        for(int j = nextSegCounter; j > 0; j--)
        {
            if(usedSegments[j] != false)
            {
                continue;
            }
            if(!isValidColour(horizontalsegments[j].getColour(), validColours))
            {
                continue;
            }
            if(horizontalsegments[j].getStartPoint().y < Ystart - spacing*4)
            {
                break;
            }
            if(horizontalsegments[j].getStartPoint().x   >= Xstart - spacing/2
               && horizontalsegments[j].getEndPoint().x  <= Xend + spacing/2)
            {
                if (horizontalsegments[j].getStartPoint().x < Xstart)
                {
                    Xstart = horizontalsegments[j].getStartPoint().x;
                }
                if (horizontalsegments[j].getEndPoint().x > Xend)
                {
                    Xend = horizontalsegments[j].getEndPoint().x;
                }
                if(horizontalsegments[j].getStartPoint().y < Ystart)
                {
                    Ystart = horizontalsegments[j].getStartPoint().y;
                }
                tempSegments.push_back(horizontalsegments[j]);
                horizontalsegments[j].isUsed = true;
                tempUsedSegments.push_back(j);
            }

        }
        //qDebug() << "About: Creating candidate: " << Xstart << ","<< Ystart<< ","<< Xend<< ","<< Yend << " Size: " << tempSegments.size();
        //Create Object Candidate if greater then the minimum number of segments
        if((int)tempSegments.size() >= min_segments)
        {
            //qDebug() << "Creating candidate: " << Xstart << ","<< Ystart<< ","<< Xend<< ","<< Yend << " Size: " << tempSegments.size();
            //for(size_t i =0; i < tempSegments.size(); i++)
            //{
                //qDebug() << tempSegments[i].getStartPoint().x << "," <<tempSegments[i].getStartPoint().y << " " << tempSegments[i].getEndPoint().x << "," <<tempSegments[i].getEndPoint().y;
            //}
            ObjectCandidate tempCandidate(Xstart, Ystart, Xend, Yend, validColours[0], tempSegments);
            candidates.push_back(tempCandidate);
            while (!tempUsedSegments.empty())
            {
                usedSegments[tempUsedSegments.back()] = true;
                tempUsedSegments.pop_back();
            }
        }
        else {
            //reset segments to not used
            while (!tempUsedSegments.empty())
            {
                horizontalsegments[tempUsedSegments.back()].isUsed = false;
                leftover.push_back( horizontalsegments[tempUsedSegments.back()] );
                tempUsedSegments.pop_back();
            }
        }

    }
    //qDebug() << "candidate size: " << candidates.size();
    return candidates;
}

void Vision::DetectLineOrRobotPoints(ClassifiedSection* scanArea, LineDetection* LineDetector)
{
    //qDebug() << "Forming Lines or Robot Points:" << endl;

    LineDetector->FindLineOrRobotPoints(scanArea, this);


    return;
}

void Vision::DetectLines(LineDetection* LineDetector)
{
    //qDebug() << "Forming Lines:" << endl;

    LineDetector->FormLines(AllFieldObjects, this, m_sensor_data);

    //qDebug() << "Detected: " <<  fieldLines.size() << " Lines, " << cornerPoints.size() << " Corners." <<endl;

    return;
}
void Vision::DetectLines(LineDetection* LineDetector, vector< ObjectCandidate >& candidates, vector< TransitionSegment >& leftover)
{
    //qDebug() << "Forming Lines:" << endl;

    LineDetector->FormLines(AllFieldObjects, this, m_sensor_data, candidates, leftover);

    #if DEBUG_VISION_VERBOSITY > 5
        debug << "\tDetected: " <<  LineDetector->fieldLines.size() << " Lines, " << LineDetector->cornerPoints.size() << " Corners." <<endl;
    #endif

    //qDebug() << "Detected: " <<  fieldLines.size() << " Lines, " << cornerPoints.size() << " Corners." <<endl;

    return;
}
Circle Vision::DetectBall(const std::vector<ObjectCandidate> &FO_Candidates)
{
    //debug<< "Vision::DetectBall" << endl;

    Ball BallFinding;

    //qDebug() << "Vision::DetectBall : Ball Class created" << endl;
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    //qDebug() << "Vision::DetectBall : getting Image sizes" << endl;
    //qDebug() << "Vision::DetectBall : Init Ball" << endl;

    Circle ball;
    ball.isDefined = false;
    if (FO_Candidates.size() <= 0)
    {
        return ball;
    }
    #if DEBUG_VISION_VERBOSITY > 6
        debug << "Vision::DetectBall : Find Ball" << endl;
    #endif
    ball = BallFinding.FindBall(FO_Candidates, AllFieldObjects, this, height, width);
    #if DEBUG_VISION_VERBOSITY > 6
    debug << "Vision::DetectBall : Finnised FO_Ball" << endl;
    #endif
    if(ball.isDefined)
    {
        #if DEBUG_VISION_VERBOSITY > 6
            debug<< "Vision::DetectBall : Update FO_Ball" << endl;
        #endif
        Vector2<float> positionAngle;
        Vector2<int> viewPosition;
        Vector2<int> sizeOnScreen;
        Vector3<float> sphericalError;
        Vector3<float> visualSphericalPosition;
        Vector3<float> transformedSphericalPosition;
        viewPosition.x = (int)round(ball.centreX);
        viewPosition.y = (int)round(ball.centreY);
        double ballDistanceFactor=EFFECTIVE_CAMERA_DISTANCE_IN_PIXELS()*ORANGE_BALL_DIAMETER;
        float BALL_OFFSET = 0.0;
        float distance = (float)(ballDistanceFactor/(2*ball.radius)+BALL_OFFSET);
        float bearing = (float)CalculateBearing(viewPosition.x);
        float elevation = (float)CalculateElevation(viewPosition.y);
        positionAngle.x = bearing;
        positionAngle.y = elevation;
        visualSphericalPosition[0] = distance;
        visualSphericalPosition[1] = bearing;
        visualSphericalPosition[2] = elevation;

        vector<float> ctvector;
        bool isOK = getSensorsData()->get(NUSensorsData::CameraTransform, ctvector);
        if(isOK == true)
        {
            Matrix cameraTransform = Matrix4x4fromVector(ctvector);
            transformedSphericalPosition = Kinematics::TransformPosition(cameraTransform,visualSphericalPosition);
        }
        //qDebug() << "Vision::DetectBall : Update FO_Ball" << (ball.radius*2);
        sizeOnScreen.x = int(ball.radius*2);
        sizeOnScreen.y = int(ball.radius*2);

        AllFieldObjects->mobileFieldObjects[FieldObjects::FO_BALL].UpdateVisualObject(transformedSphericalPosition,
                                                                                      sphericalError,
                                                                                      positionAngle,
                                                                                      viewPosition,
                                                                                      sizeOnScreen,
                                                                                      currentImage->GetTimestamp());
        //ballObject.UpdateVisualObject(sphericalPosition,sphericalError,viewPosition);
        //qDebug() << "Setting FieldObject:" << AllFieldObjects->mobileFieldObjects[FieldObjects::FO_BALL].isObjectVisible();

        #if DEBUG_VISION_VERBOSITY > 6
        debug    << "At: Distance: " << AllFieldObjects->mobileFieldObjects[FieldObjects::FO_BALL].measuredDistance()
                    << " Bearing: " << AllFieldObjects->mobileFieldObjects[FieldObjects::FO_BALL].measuredBearing()
                    << " Elevation: " << AllFieldObjects->mobileFieldObjects[FieldObjects::FO_BALL].measuredElevation() << endl;
        #endif

    }

    //qDebug() << "Vision::DetectBall : Finnished" << endl;
    return ball;

}

void Vision::DetectGoals(std::vector<ObjectCandidate>& FO_Candidates,std::vector<ObjectCandidate>& FO_AboveHorizonCandidates,std::vector< TransitionSegment > horizontalSegments)
{
    int width = currentImage->getWidth();
    int height = currentImage->getHeight();
    GoalDetection goalDetector;
    goalDetector.FindGoal(FO_Candidates, FO_AboveHorizonCandidates, AllFieldObjects, horizontalSegments, this,height,width);
    goalDetector.FindGoal(FO_Candidates, FO_AboveHorizonCandidates, AllFieldObjects, horizontalSegments, this,height,width);
    return;
}

void Vision::PostProcessGoals()
{
    GoalDetection goalDetector;
    goalDetector.PostProcessGoalPosts(AllFieldObjects);
    return;
}

void Vision::DetectRobots(std::vector < ObjectCandidate > &RobotCandidates)
{
    int MaxPercentageOfColour = 50;
    int MinPercentageOfColour = 1;
    for(unsigned int i = 0; i < RobotCandidates.size(); i++)
    {
        std::vector <TransitionSegment > segments = RobotCandidates[i].getSegments();
        int pinkSize = 0;
        int blueSize = 0;
        int whiteSize = 0;
        for(unsigned int j = 0; j < segments.size(); j++)
        {
            if(segments[j].getColour() == ClassIndex::pink || segments[j].getColour() == ClassIndex::pink_orange)
            {
                pinkSize = pinkSize + segments[j].getSize();
            }
            else if(segments[j].getColour() == ClassIndex::shadow_blue || segments[j].getColour() == ClassIndex::blue)
            {
                blueSize = blueSize + segments[j].getSize();
            }
            else if(segments[j].getColour() == ClassIndex::white )
            {
                whiteSize = whiteSize + segments[j].getSize();
                //qDebug() << "White Segments: " << segments[j].getSize() << segments[j].getStartPoint().x << segments[j].getStartPoint().y << segments[j].getEndPoint().x << segments[j].getEndPoint().y;
            }

        }
        //qDebug() << i <<": Segments: " << segments.size()<< "White: " <<  whiteSize << "  Blue: " << blueSize;
        if( (blueSize > pinkSize && whiteSize*MaxPercentageOfColour/100 > blueSize && blueSize > whiteSize*MinPercentageOfColour/100))// || RobotCandidates[i].getColour() == ClassIndex::blue || RobotCandidates[i].getColour() == ClassIndex::shadow_blue)
        {
            //qDebug() << i <<": Blue Robot: ";
            RobotCandidates[i].setColour(ClassIndex::shadow_blue);
            //MAKE a Mobile_FIELDOBJECT: BLUE ROBOT
            AmbiguousObject tempRobotObject(FieldObjects::FO_BLUE_ROBOT_UNKNOWN, "UNKNOWN BLUE ROBOT");
            //qDebug() << i <<": Blue Robot: make object";
            Vector2<int> bottomRight = RobotCandidates[i].getBottomRight();
            float cx = RobotCandidates[i].getCentreX();
            float cy = bottomRight.y;
            float bearing = CalculateBearing(cx);
            float elevation = CalculateElevation(cy);
            float distance = 0;
            //qDebug() << i <<": Blue Robot: get transform";
            vector<float> ctgvector;
            Vector3<float> measured(distance,bearing,elevation);
            Vector2<float> screenPositionAngle(bearing,elevation);
            bool isOK = getSensorsData()->get(NUSensorsData::CameraToGroundTransform, ctgvector);
            if(isOK == true)
            {
                Matrix camera2groundTransform = Matrix4x4fromVector(ctgvector);
                measured = Kinematics::DistanceToPoint(camera2groundTransform, bearing, elevation);

                #if DEBUG_VISION_VERBOSITY > 6
                    debug << "\t\tCalculated Distance to Point: " << distance<<endl;
                #endif
            }
            //qDebug() << i <<": Blue Robot: get things in order";
            Vector3<float> measuredError(0,0,0);
            Vector2<int> screenPosition(RobotCandidates[i].getCentreX(), RobotCandidates[i].getCentreY());
            Vector2<int> sizeOnScreen(RobotCandidates[i].width(), RobotCandidates[i].height());
            //qDebug() << i <<": Blue Robot: update object with vision data";
            tempRobotObject.UpdateVisualObject(measured,measuredError,screenPositionAngle,screenPosition,sizeOnScreen,m_timestamp);
            AllFieldObjects->ambiguousFieldObjects.push_back(tempRobotObject);
            //qDebug() << i <<": Blue Robot: object push to FO";
        }
        else if( (blueSize < pinkSize && whiteSize*MaxPercentageOfColour/100 > pinkSize &&  pinkSize > whiteSize*MinPercentageOfColour/100) ) //|| RobotCandidates[i].getColour() == ClassIndex::pink || RobotCandidates[i].getColour() == ClassIndex::pink_orange)
        {
            //qDebug() << i <<": Pink Robot: ";
            RobotCandidates[i].setColour(ClassIndex::pink);
            AmbiguousObject tempRobotObject(FieldObjects::FO_PINK_ROBOT_UNKNOWN, "UNKNOWN PINK ROBOT");
            //qDebug() << i <<": Pink Robot: make object";
            Vector2<int> bottomRight = RobotCandidates[i].getBottomRight();
            float cx = RobotCandidates[i].getCentreX();
            float cy = bottomRight.y;
            float bearing = CalculateBearing(cx);
            float elevation = CalculateElevation(cy);
            float distance = 0;
            //qDebug() << i <<": pink Robot: get transform";
            vector<float> ctgvector;
            Vector3<float> measured(distance,bearing,elevation);
            Vector2<float> screenPositionAngle(bearing,elevation);
            bool isOK = getSensorsData()->get(NUSensorsData::CameraToGroundTransform, ctgvector);
            if(isOK == true)
            {
                Matrix camera2groundTransform = Matrix4x4fromVector(ctgvector);
                measured = Kinematics::DistanceToPoint(camera2groundTransform, bearing, elevation);

                #if DEBUG_VISION_VERBOSITY > 6
                    debug << "\t\tCalculated Distance to Point: " << distance<<endl;
                #endif
            }
            Vector3<float> measuredError(0,0,0);
            Vector2<int> screenPosition(RobotCandidates[i].getCentreX(),RobotCandidates[i].getCentreY());
            Vector2<int> sizeOnScreen( RobotCandidates[i].width(), RobotCandidates[i].height());
            //qDebug() << i <<": pink Robot: update object with vision data";
            tempRobotObject.UpdateVisualObject(measured,measuredError,screenPositionAngle,screenPosition,sizeOnScreen,m_timestamp);
            AllFieldObjects->ambiguousFieldObjects.push_back(tempRobotObject);
            //qDebug() << i <<": pink Robot: object push to FO";
        }
    }


}

double Vision::CalculateBearing(double cx) const{
    //double FOVx = deg2rad(46.40f); //Taken from Old Globals
    double FOVx = m_camera_specs->m_horizontalFov;
    return atan( (currentImage->getWidth()/2-cx) / ( (currentImage->getWidth()/2) / (tan(FOVx/2.0)) ) );
}


double Vision::CalculateElevation(double cy) const{
    //double FOVy = deg2rad(34.80f); //Taken from DOCUMENTATION OF NAO
    double FOVy = m_camera_specs->m_verticalFov;
    return atan( (currentImage->getHeight()/2-cy) / ( (currentImage->getHeight()/2) / (tan(FOVy/2.0)) ) );
}

double Vision::EFFECTIVE_CAMERA_DISTANCE_IN_PIXELS()
{
    //double FOVx = deg2rad(46.40f); //Taken from DOCUMENTATION OF NAO
    double FOVx = m_camera_specs->m_horizontalFov;
    return (0.5*currentImage->getWidth())/(tan(0.5*FOVx));
}

/*! @brief Returns the number of frames dropped since the last call to this function
    @return the number of frames dropped
 */
int Vision::getNumFramesDropped()
{
    int framesdropped = numFramesDropped;
    numFramesDropped = 0;
    return framesdropped;
}

/*! @brief Returns the number of frames processed since the last call to this function
    @return the number of frames processed
 */
int Vision::getNumFramesProcessed()
{
    int framesprocessed = numFramesProcessed;
    numFramesProcessed = 0;
    return framesprocessed;
}

/*
QImage Vision::getEdgeFilter()
{
    return getEdgeFilter(0,0,currentImage->getWidth(), currentImage->getHeight());
}

QImage Vision::getEdgeFilter(int x, int y, int width, int height)
{
    return getEdgeFilter(currentImage->getSubImage(x,y,width, height, 1));
}

QImage Vision::getEdgeFilter(int x, int y, int width, int height, int decimation_spacing)
{
    return getEdgeFilter(currentImage->getSubImage(x,y,width, height, decimation_spacing));
}

QImage Vision::getEdgeFilter(QImage image)
{
    QImage* resultant = new QImage(image.width(), image.height(), QImage::Format_ARGB32);
    QImage* grayscale = new QImage(image.width(), image.height(), QImage::Format_ARGB32);
    float c = 0;
    unsigned char cP = 0;
    unsigned char cR = 0;
    unsigned int rgb = 0, gray = 0;

    //Precalculate Grayscale image
    for (int x = 1; x < image.width() - 1; x++)
        for (int y = 1; y < image.height() - 1; y++)
        {
            rgb = image.pixel(x,y);
            ColorModelConversions::rgb2gray(rgb, gray);
            grayscale->setPixel(x,y,gray);
        }

     //filter with edge detecting filter
     for (int x = 1; x < image.width() - 1; x++)
     {
         for (int y = 1; y < image.height() - 1; y++)
         {
             c = 0; cP = 0; cR = 0;

             c += 2*(grayscale->pixel(x-1,y-1) % 256); //a
             c += 2*(grayscale->pixel(x  ,y-1) % 256); //b
             c += 2*(grayscale->pixel(x-1,y  ) % 256); //d

             c -= 2*(grayscale->pixel(x+1,y  ) % 256); //f
             c -= 2*(grayscale->pixel(x  ,y+1) % 256); //h
             c -= 2*(grayscale->pixel(x+1,y+1) % 256); //i

             if      (c > 255) cR = 255;
             else if (c <   0) cR = 0;
             else                    cR = (unsigned char)(c);
             resultant->setPixel(x,y, (cR * 0x01010101));
         }
     }

    return *resultant;
}
*/
/*!
  @brief perform an fft to extract prominent features for further analysis
  */
/*
QImage Vision::getFFT()
{
    qDebug() << "getFFT()";
    return getFFT(0,0,currentImage->getWidth(), currentImage->getHeight(), 1);
}
QImage Vision::getFFT(int x, int y, int width, int height)
{
    return getFFT(currentImage->getSubImage(x,y,width, height, 1));
}
QImage Vision::getFFT(int x, int y, int width, int height, int decimation_spacing)
{
    qDebug() << "getFFT(int "<< x <<", int "<< y <<", int "<< width <<", int "<< height <<", int "<< decimation_spacing <<")";
    return getFFT(currentImage->getSubImage(x,y,width, height, decimation_spacing));
}

QImage Vision::getFFT(QImage image)
{
    qDebug() << "getFFT(QImage)";
    unsigned int rgb = 0, gray = 0;
    for (int x = 0; x < image.width(); x++){
        for (int y = 0; y < image.height(); y++){
            rgb = image.pixel(x,y);
            qDebug() << "("<<x<<","<<y<<")::"<<((rgb/16777216)%256)<<"|"<<((rgb/65536)%256)<<"|"<<((rgb/256)%256)<<"|"<<((rgb)%256);
            ColorModelConversions::rgb2gray(rgb, gray);
            qDebug() << "("<<x<<","<<y<<")::"<<((gray/16777216)%256)<<"|"<<((gray/65536)%256)<<"|"<<((gray/256)%256)<<"|"<<((gray)%256);
            image.setPixel(x, y, gray);
        }
    }
    qDebug() << "getFFT(QImage)end";

    return image;
}*/

bool Vision::isPixelOnScreen(int x, int y)
{
    if(x <= 0 || x >= currentImage->getWidth())
    {
        return false;
    }

    if(y <= 0 || y >= currentImage->getHeight())
    {
        return false;
    }
    else
        return true;
}

//OBSTACLE DETECTION

vector<AmbiguousObject> Vision::getObjectsFromCandidates(vector<ObjectCandidate> candidates)
{
    vector<AmbiguousObject> objectList;

    for(unsigned int i=0; i<candidates.size(); i++)
    {
        AmbiguousObject tempObject(FieldObjects::FO_OBSTACLE, "UNKNOWN OBSTACLE");

        Vector2<int> visual_centre(candidates.at(i).getCentreX(), candidates.at(i).getCentreY());
        Vector2<int> visual_dim(candidates.at(i).width(), candidates.at(i).height());
        Vector2<int> bottom_right = candidates.at(i).getBottomRight();

        //calculate position based on bottom centre of candidate
        float cx = visual_centre.x;
        float cy = bottom_right.y;

        float bearing = CalculateBearing(cx);
        float elevation = CalculateElevation(cy);
        float distance = 0;
        vector<float> ctgvector;
        Vector3<float> measured(distance,bearing,elevation);
        Vector2<float> screenPositionAngle(bearing,elevation);
        //bool isOK = getSensorsData()->get(NUSensorsData::CameraTransform, ctgvector);
        bool isOK = getSensorsData()->get(NUSensorsData::CameraToGroundTransform, ctgvector);
        if(isOK == true)
        {
            //Matrix cameraTransform = Matrix4x4fromVector(ctgvector);
            //measured = Kinematics::TransformPosition(cameraTransform, measured);
            Matrix camera2groundTransform = Matrix4x4fromVector(ctgvector);
            measured = Kinematics::DistanceToPoint(camera2groundTransform, bearing, elevation);

            if(cy == currentImage->getHeight()) {
                //Object seen at bottom of image, should look down for better view, however
                //currently just assume object very close
                measured.x = 15;
            }

            #if DEBUG_VISION_VERBOSITY > 6
                debug << "\t\tCalculated Distance to Point: " << distance << endl;
            #endif
        }
        //qDebug() << i << ": Obstacle: get things in order";
        Vector3<float> measuredError(0,0,0);
        //qDebug() << i << ": Obstacle: update object with vision data";
        tempObject.UpdateVisualObject(measured, measuredError, screenPositionAngle, visual_centre, visual_dim, m_timestamp);
        objectList.push_back(tempObject);

        #if DEBUG_VISION_VERBOSITY > 0
            debug << "Obst " << i << " bottom: (" << cx << ", " << cy << ") centre: (" <<
                     visual_centre.x << ", " << visual_centre.y << ") d2p: " <<
                     measured.x << "cm b2p:" << measured.y << "rad e2p:" << measured.z <<endl;
        #endif
    }

    return objectList;
}

vector<int> Vision::getVerticalDifferences(const vector< Vector2<int> >& prehull, const vector< Vector2<int> >& hull) const
{
    //returns a vector of pairwise vertical distance values from prehull to hull (in screen coordinates)
    vector<int> result;
    int next_diff;
    if(prehull.size() != hull.size()) {
        //non matching vectors - return empty result
        //debug
        #if DEBUG_VISION_VERBOSITY > 1
            debug << "Non matching vectors - no obstacle detection\n";
        #endif

        return result;
    }
    else {
        for(unsigned int i=0; i<hull.size(); i++) {
            //loop through the vectors in parallel
            next_diff = prehull.at(i).y - hull.at(i).y;  //prehull has larger y value (counts down from top)
            result.push_back(next_diff);
        }
        return result;
    }
}

vector<ObjectCandidate> Vision::getObstacleCandidates(const vector< Vector2<int> >& prehull, const vector< Vector2<int> >& hull,
                                                          int height_thresh, int width_min) const
{
    vector<ObjectCandidate> result;
    ObjectCandidate next_obst;
    Vector2<int> start, end, top_left, bottom_right;
    int width, height;

    vector<int> differences = getVerticalDifferences(prehull, hull);

    int consecutive_hull_breaks = 0;
    int lowest_point;

    for(unsigned int i=0; i<differences.size(); i++) {
        if(differences.at(i) >= height_thresh) {
            //there is a break
            if(consecutive_hull_breaks == 0) {
                //first scan of break
                start = prehull.at(i);
                end = prehull.at(i);
                lowest_point = start.y;
                height = lowest_point - hull.at(i).y;
            }
            else {
                //consecutive scan of break
                end = prehull.at(i);
                if(end.y > lowest_point) {
                    lowest_point = end.y; //keep track of minimum
                    height = lowest_point - hull.at(i).y;
                }
            }
            consecutive_hull_breaks++;
        }
        else {
            //possible end of break
            if(consecutive_hull_breaks >= width_min) {
                //break large enough to indicate obstacle
                //  x-position given by centre of break
                //  y-position given by lowest point in break
                width = start.x - end.x;
                //next_obst.x = width / 2; //centre of break
                //next_obst.y = lowest_point;

                top_left.x = start.x;
                top_left.y = lowest_point - height;
                bottom_right.x = end.x;
                bottom_right.y = lowest_point;

                next_obst.setTopLeft(top_left);
                next_obst.setBottomRight(bottom_right);

                result.push_back(next_obst);
            }
            //restart
            consecutive_hull_breaks = 0;
        }
    }
    return result;
}
