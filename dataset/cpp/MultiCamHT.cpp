#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <sys/timeb.h>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;

// Use the following enum and global constant to select whether a software or
// hardware trigger is used.
enum triggerType
{
	SOFTWARE,
	HARDWARE
};

int getMilliCount(){
        timeb tb; 
        ftime(&tb);
        int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
        return nCount;
}

int getMilliSpan(int nTimeStart){
        int nSpan = getMilliCount() - nTimeStart;
        if(nSpan < 0)
                nSpan += 0x100000 * 1000;
        return nSpan;
}

//const triggerType chosenTrigger = HARDWARE;
const triggerType chosenTrigger = SOFTWARE;



//
//
//
// Functions to Configure Trigger and Camera
//
//
//



// This function configures the camera to use a trigger. 
// 1. trigger is disabled before making any changes. source
// 2. if camera is primary then trigger source is set to software, else trigger source is set to hardware.
// 3. if camera is secondary, trigger overlap is set to readout.
// 4. trigger selector is set to frame start.
// 5. trigger is enabled
int ConfigureTrigger(INodeMap & nodeMap, bool isPrimary)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING TRIGGER ***" << endl << endl;
    
    try
    {
		///////////////////////////////////////////////////////
		// Disable Trigger on both primary and secondary camera
		// Leave trigger disabled on primary camera
        ///////////////////////////////////////////////////////
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }
		
        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }
		
        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());
        cout << "Trigger disabled. Configuring Trigger..." << endl;
		// Trigger disabled

		
		//////////////////////////////////////
		// Set line selector on primary camera
		//////////////////////////////////////	
        if (isPrimary == true)
        {
			// Get pointer to "Line Selector"
	        CEnumerationPtr ptrLineSelector = nodeMap.GetNode("LineSelector");
	        if (!IsAvailable(ptrLineSelector) || !IsWritable(ptrLineSelector))
	        {
	            cout << "Unable to acquire pointer to Line Selector for Primary Camera. Aborting..." << endl;
	            return -1;
	        }

	        // Set Line Selector to Line 0
            CEnumEntryPtr ptrLineSelectorLine0 = ptrLineSelector->GetEntryByName("Line0");
            if (!IsAvailable(ptrLineSelectorLine0) || !IsReadable(ptrLineSelectorLine0))
            {
                cout << "Unable to set Line Selector to Line 0. Aborting..." << endl;
                return -1;
            }

            ptrLineSelector->SetIntValue(ptrLineSelectorLine0->GetValue());
            cout << "Line Selector on Primary Camera set to Line 0" << endl;
    

#if 0      
            //Enable the 3.3V option i.e. make it true
            CBooleanPtr ptrV3_3 = nodeMap.GetNode("V3_3Enable");
            if (!IsAvailable(ptrV3_3) || !IsWritable(ptrV3_3))
            {
                cout << "Unable to retrieve 3.3 V enable value. Aborting..." << endl;
                return -1;
            }
            CBooleanPtr ptrV3_3Enable = ptrV3_3->GetEntryByName("true");

            if (!IsAvailable(ptrV3_3Enable) || !IsReadable(ptrV3_3Enable))
            {
                cout << "Unable to set voltage 3.3V to true. Aborting..." << endl;
                return -1;
            }

            ptrV3_3->SetValue(ptrV3_3Enable->GetValue());
            //TODO: print if the value got enabled.
            ptrV3_3->SetValue(true); 
#endif


 		}// End of configuring trigger for Primary Camera


		////////////////////////////////////////////////////////
		// Configure hardware trigger source on secondary camera
		////////////////////////////////////////////////////////
		else 
		{
			// Select trigger source
	        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
	        if (!IsAvailable(ptrTriggerSource) || !IsWritable(ptrTriggerSource))
	        {
	            cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
	            return -1;
	        }

			////////////////////////////////////////
            //Secondary camera TriggerSource = Line3
			////////////////////////////////////////
            CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line3");
            if (!IsAvailable(ptrTriggerSourceHardware) || !IsReadable(ptrTriggerSourceHardware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());
            cout << "Secondary Camera: TriggerSource = Line3." << endl;
			//TriggerSource set to Line3

	
			//////////////////////////////
            //Set TriggerOverlap = ReadOut
			//////////////////////////////
            CEnumerationPtr ptrTriggerOverlap = nodeMap.GetNode("TriggerOverlap");

            if (!IsAvailable(ptrTriggerOverlap) || !IsReadable(ptrTriggerOverlap))
            {
                cout << "Unable to set trigger overlap. Aborting..." << endl;
                return -1;
            }

            CEnumEntryPtr ptrTriggerOverlapValue = ptrTriggerOverlap->GetEntryByName("ReadOut");
            if (!IsAvailable(ptrTriggerOverlapValue) || !IsReadable(ptrTriggerOverlapValue))
            {
                cout << "Unable to grab overlap value. Aborting..." << endl;
                return -1;
            }
            ptrTriggerOverlap->SetIntValue(ptrTriggerOverlapValue->GetValue());
            cout << "Secondary Camera: TriggerOverlap = ReadOut" << endl << endl;
			//TriggerOverlap set to ReadOut

			///////////////////////////////////////////////////////////////////////
	        // Setting trigger selector to FrameBurst/Acquisition/FrameStart Mode
	        //////////////////////////////////////////////////////////////////////
	        CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");
	        if (!IsAvailable(ptrTriggerSelector) || !IsReadable(ptrTriggerSelector)) {
	            cout << "Unable to get the trigger selectr value. Abort....." << endl << endl;
	            return -1;
	        }

	        CEnumEntryPtr ptrTriggerSelectorSel = ptrTriggerSelector->GetEntryByName("FrameStart");
	        if (!IsAvailable(ptrTriggerSelectorSel) || !IsReadable(ptrTriggerSelectorSel))
	        {
	            cout << "Unable to selct trigger selector "
	                "(enum entry retrieval). Aborting..." << endl;
	            return -1;
	        }
	        ptrTriggerSelector->SetIntValue(ptrTriggerSelectorSel->GetValue());
	        cout << "Trigger selector: " << ptrTriggerSelector->GetIntValue() << endl;


	        //////////////////////////////////////////
	        // Enable Trigger only on secondary camera
	        //////////////////////////////////////////
	        CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
	        if (!IsAvailable(ptrTriggerModeOn) || !IsReadable(ptrTriggerModeOn))
	        {
	            cout << "Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
	            return -1;
	        }

	        ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());
	        cout << "Trigger turned on" << endl << endl;
			// Trigger enabled

        }// End of configuring trigger for secondary camera

    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
} 
// End of ConfigureTrigger




///////////////////////////////
// Function to configure Camera
///////////////////////////////
int ConfigureCamera(CameraPtr pCam, INodeMap & nodeMap) 
{

    float acquisitionFrameRate = 0, AcFrameRate = 0;
    int result = 0;

    try {
            /////////////////////////////////////
            // Set acquisition mode to continuous
            /////////////////////////////////////
            CEnumerationPtr ptrAcquisitionMode = pCam->GetNodeMap().GetNode("AcquisitionMode");
            if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
            {
                cout << "Unable to set acquisition mode to continuous" << endl;
                return -1;
            }

            CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
            if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
            {
                cout << "Unable to set acquisition mode to continuous" << endl;
                return -1;
            }

            ptrAcquisitionMode->SetIntValue(ptrAcquisitionModeContinuous->GetValue());
            cout << "Acquisition mode set to continuous..." << endl;
            // Acquisition mode set to continuous


            ///////////////////////////////////
            // Turn off automatic exposure mode
            ///////////////////////////////////
            CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
            if (!IsAvailable(ptrExposureAuto) || !IsWritable(ptrExposureAuto))
            {
                cout << "Unable to disable automatic exposure (node retrieval). Aborting..." << endl << endl;
                return -1;
            }
        
            CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
            if (!IsAvailable(ptrExposureAutoOff) || !IsReadable(ptrExposureAutoOff))
            {
                cout << "Unable to disable automatic exposure (enum entry retrieval). Aborting..." << endl << endl;
                return -1;
            }
        
            ptrExposureAuto->SetIntValue(ptrExposureAutoOff->GetValue());
            // Auto Exposure turned off
            

            /////////////////////////////////////////////////////////////////////
            // Set exposure time manually; exposure time recorded in microseconds
            /////////////////////////////////////////////////////////////////////
            CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
            if (!IsAvailable(ptrExposureTime) || !IsWritable(ptrExposureTime))
            {
                cout << "Unable to set exposure time. Aborting..." << endl << endl;
                return -1;
            }
        
            // Ensure desired exposure time does not exceed the maximum
            const double exposureTimeMax = ptrExposureTime->GetMax();
            double exposureTimeToSet = 5500.0;

            if (exposureTimeToSet > exposureTimeMax)
            {
                exposureTimeToSet = exposureTimeMax;
            }
        
            ptrExposureTime->SetValue(exposureTimeToSet);
        
            cout << "Exposure time set to " << exposureTimeToSet << " us..." << endl << endl;
            // Exposure set to 5500.0

#if 0
            // Setting up Acquisition frame rate --------------------------------------------//
            CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
            if (!IsAvailable(ptrAcquisitionFrameRate) || !IsReadable(ptrAcquisitionFrameRate)) {
                cout << "Unable to retrieve frame rate. Aborting..." << endl << endl;
                return -1;
            }

            const float AcFrameRateMax = ptrAcquisitionFrameRate->GetMax();
            float AcFrameRatetoSet = 170;

            if(AcFrameRatetoSet > AcFrameRateMax) {
                AcFrameRatetoSet = AcFrameRateMax;
            }

            ptrAcquisitionFrameRate->SetValue(AcFrameRatetoSet);

            cout << "Acquisition Frame Rate: " << AcFrameRatetoSet << " fps" << endl;
#endif

#if 0
            // Enableing the acquisition frame rate enable option
            CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap.GetNode("Acquisition0FrameRateEnable");
            if (!IsAvailable(ptrAcquisitionFrameRateEnable)
                    || !IsReadable(ptrAcquisitionFrameRateEnable))
            {
                cout << "Unable to enable the ac frame rate. Aborting..." << endl << endl;
                return -1;
            }

            ptrAcquisitionFrameRateEnable->SetValue("On");
#endif
            ///////////////////////////////////
            // Display acquisition frame rate
            //////////////////////////////////
            CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
            if (!IsAvailable(ptrAcquisitionFrameRate)
                    || !IsReadable(ptrAcquisitionFrameRate))
            {
                cout << "Unable to retrieve frame rate. Aborting..." << endl << endl;
                return -1;
            }

            acquisitionFrameRate = static_cast<float>(ptrAcquisitionFrameRate->GetValue());
            cout << "Acquisition Frame Rate: " << acquisitionFrameRate << endl << endl;

            /////////////////////////
            // Begin acquiring images
            /////////////////////////
            pCam->BeginAcquisition();

        } 
        catch (Spinnaker::Exception &e) 
        {
        cout << "Error: " << e.what() << endl;
        result = -1;
        }
    return result;
}


// Function to Turn Off trigger on camera
int ResetTrigger(INodeMap & nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn trigger mode back off
        //
        // *** NOTES ***
        // Once all images have been captured, turn trigger mode back off to
        // restore the camera to a clean state.
        //
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsAvailable(ptrTriggerMode) || !IsReadable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Non-fatal error..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsAvailable(ptrTriggerModeOff) || !IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode "
                "(enum entry retrieval). Non-fatal error..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

        cout << "Trigger mode disabled..." << endl << endl;
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}




//
//
//
// Functions to capture images
//
//
//





// This function retrieves a single image using the trigger. In this example,
// only a single image is captured and made available for acquisition - as such,
// attempting to acquire two images for a single trigger execution would cause
// the example to hang. This is different from other examples, whereby a
// constant stream of images are being captured and made available for image
// acquisition.

// Note: If the cameras are configured for hardware trigger, your program need not
// send trigger signals to either primary or secondary camera. The primary camera will
// automatically send trigger signals over GPIO cable and keep all the cameras synchronized
int GrabNextImageByTrigger(INodeMap & nodeMap, CameraPtr pCam)
{
    int result = 0;

    try
    {
		////////////////////////
		// Send software trigger
		////////////////////////
        if (chosenTrigger == SOFTWARE)
        {
            // Execute software trigger
            CCommandPtr ptrSoftwareTriggerCommand = nodeMap.GetNode("TriggerSoftware");
            if (!IsAvailable(ptrSoftwareTriggerCommand) || !IsWritable(ptrSoftwareTriggerCommand))
            {
                cout << "Unable to execute trigger. Aborting..." << endl;
                return -1;
            }

            ptrSoftwareTriggerCommand->Execute();
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}


/////////////////////////////////////////////////////
// Retrieve, convert, and save images for each camera
/////////////////////////////////////////////////////
int AcquireImages(CameraList camList)
{
    int result = 0;
    CameraPtr pCam = NULL, primaryCam = NULL;
    int imgTotal = 5000;

    try
    {
    
        // Uncomment this part to display FPS
#if 0
        int start = getMilliCount();
        vector<int> v_time;
#endif

        int start = getMilliCount();
        vector<int> v_time;

        unsigned int imgNum = 0;
        Mat src[camList.GetSize()];
        
        primaryCam = camList.GetByIndex(0);
        INodeMap & primaryNodeMap = primaryCam->GetNodeMap();

        cout << "Streaming Video" << endl;
       
        while(imgNum < imgTotal)
        {


			// Uncomment this part if you want to use software trigger
#if 0
			// Get user input
            cout << "Press the Enter key to initiate software trigger." << endl;
            getchar();

            // Retrieve the next image from the trigger
            result = result | GrabNextImageByTrigger(primaryNodeMap, primaryCam);

            if(result == -1) {
                cout << __LINE__ << imgNum << endl;
                continue;
            }
#endif


            for(int i = 0; i < camList.GetSize(); ++i)
            {	
                // Select camera
                pCam = camList.GetByIndex(i);
                try
                {
                	// Acquire Image from camera
                    ImagePtr pResultImage = pCam->GetNextImage();

                    if (pResultImage->IsIncomplete())
                    {
                        cout << "Image incomplete with image status " 
                            << pResultImage->GetImageStatus() << "..." << endl << endl;
                    }
                    else
                    {
                        // Convert image to BayerRG8
                        ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);

                        unsigned int rowBytes
                            = (int)convertedImage->GetImageSize()/convertedImage->GetHeight();

                        src[i] = Mat(convertedImage->GetHeight(),
                                convertedImage->GetWidth(), CV_8UC1, convertedImage->GetData(),
                                rowBytes);

                        resize(src[i], src[i], Size(640, 480), 0,0, INTER_LINEAR);

                        // Display captured image
                        cv::imshow("Camera-" + to_string(i), src[i]);

                        // Uncomment to save images
#if 0
                        char fileName[1000];
						
						// Create filename
						sprintf(fileName, "/home/umh-admin/Pictures/MultiCam/Cam%d-img%d.jpg", i, imgNum);

                        cv::imwrite(filename, src[i]);
#endif
                        cv::waitKey(1);

                    }
                    pResultImage->Release();
					
                }
                catch (Spinnaker::Exception &e)
                {
                    cout << "Error: " << e.what() << endl;
                    result = -1;
                }
            }

            // Uncomment to display FPS
            int timeElapsed = getMilliSpan(start);
            v_time.push_back(timeElapsed);

            if (v_time.size() > 10)
            {
                int t = timeElapsed-v_time[v_time.size()-10];
                double fps = 10000.0/t;
                cout << fps << endl;
            }

            ++imgNum;
        }

        //////////////////////////////////
        // End acquisition for each camera
        //////////////////////////////////
        for (int i = 0; i < camList.GetSize(); ++i)
        {
            // End acquisition
            camList.GetByIndex(i)->EndAcquisition();
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}




//
//
// Init Functions
//
//


// Function to initialize and deinitialize each camera
int RunMultipleCameras(CameraList camList)
{
    int result = 0;
    CameraPtr pCam = NULL;
    bool isPrimary = true;

    try
    {
        // Initialize camera and trigger for each camera
        for (int i = 0; i < camList.GetSize(); i++)
        {
            // Select camera
            pCam = camList.GetByIndex(i);

			// Print Camera Serial Number
			cout << "Initializing Camera: " << i << " SerialNum:" << pCam->GetUniqueID() << endl;

            // Initialize camera
            pCam->Init();

            // Retrieve GenICam nodemap
            INodeMap & nodeMap = camList.GetByIndex(i)->GetNodeMap();

            // Set camera with index 0 as Primary camera.
			// FUTURE: Update this part of code to set camera with specific serial number as Primary
            if(i>0)
                isPrimary = false;

            // Call function to configure trigger
            result = ConfigureTrigger(nodeMap, isPrimary);
            if (result < 0)
            {
				cout << "Error configuring trigger" << endl;
                return result;
            }

            // Call function to configure camera
            result = ConfigureCamera(pCam, nodeMap);
			if (result < 0)
            {
				cout << "Error configuring camera" << endl;
                return result;
            }
        }// End of initialization of trigger and camera
		

        // Acquire images on all cameras
        result = AcquireImages(camList);
		if (result < 0)
            {
				cout << "Error acquiring images from cameras" << endl;
                return result;
            }


        // Deinitialize each camera
        for (int i = 0; i < camList.GetSize(); i++)
        {
            // Select camera
            pCam = camList.GetByIndex(i);

            // Retrieve GenICam nodemap
            INodeMap & nodeMap = pCam->GetNodeMap();

            // Reset trigger
            result = result | ResetTrigger(nodeMap);

            // Deinitialize camera
            pCam->DeInit();
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}


// Init: Get conneceted cameras
int main(int /*argc*/, char** /*argv*/)
{
    int result = 0;

    // Print application build information
    cout << "Program build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

    // Retrieve list of cameras from the system
    CameraList camList = system->GetCameras();

    unsigned int numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl << endl;

    // Finish if there are no cameras
    if (numCameras == 0)
    {
        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();

        cout << "Not enough cameras!" << endl;
        cout << "Done! Press Enter to exit..." << endl;
        getchar();

        return -1;
    }

	//Acquire Images from Multiple Cameras
    result = RunMultipleCameras(camList);

    cout << "Closing Program. Doing Clean Up" << endl << endl;

    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return result;
}
