#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <future>

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


// Function Declarations
int ConfigureTrigger(INodeMap & nodeMap);
int ResetTrigger(INodeMap & nodeMap);
int GrabNextImageByTrigger(CameraList camList);
int ConfigureCamera(CameraPtr pCam, INodeMap & nodeMap);
void emptyImageBuffer(CameraList camList);
int RunMultipleCameras(CameraList camList);
int getMilliCount();
int getMilliSpan(int nTimeStart);


// Use the following enum and global constant to select whether a software or
// hardware trigger is used.
enum triggerType
{
	SOFTWARE,
	HARDWARE
};

// Select Trigger type
const triggerType chosenTrigger = SOFTWARE;


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


//
// Functions to Configure Trigger and Camera
//

// This function configures the camera to use a trigger. 
// 1. trigger is disabled before making any changes. source
// 2. configure all the cameras to work with software trigger
// 3. trigger selector is set to frame start.
// 4. trigger is enabled
int ConfigureTrigger(INodeMap & nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING SOFTWARE TRIGGER ***" << endl << endl;
    
    try
    {
		//////////////////
		// Disable Trigger
        //////////////////
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
		///////////////////
		// Trigger Disabled
        ///////////////////

	
        // Select trigger source
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (!IsAvailable(ptrTriggerSource) || !IsWritable(ptrTriggerSource))
        {
            cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }
		
        ////////////////////////////////
		//Set trigger source to Software
		////////////////////////////////
		
		CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");
		if (!IsAvailable(ptrTriggerSourceSoftware) || !IsReadable(ptrTriggerSourceSoftware))
		{
		    cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
		    return -1;
		}

		ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());
		//cout << "Trigger source for the camera set to Software" << endl;


		/////////////////
		// Enable Trigger
		/////////////////
		CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
		if (!IsAvailable(ptrTriggerModeOn) || !IsReadable(ptrTriggerModeOn))
		{
		    cout << "Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
		    return -1;
		}

		ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());
		cout << "Trigger turned on" << endl << endl;
		//////////////////
		// Trigger Enabled
		//////////////////
		
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
} 
// End of Configure Trigger




// Function to Turn Off trigger on camera
int ResetTrigger(INodeMap & nodeMap)
{
    int result = 0;

    try
    {
        /////////////////////////////
        // Turn trigger mode back off
		/////////////////////////////
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



///////////////////////////////
// Function to configure Camera
///////////////////////////////
int ConfigureCamera(CameraPtr pCam, INodeMap & nodeMap) 
{

    float acquisitionFrameRate = 0, AcFrameRate = 0;
    int result = 0;

    try	{
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



/////////////////////////
// GrabNextImageByTrigger
/////////////////////////
int GrabNextImageByTrigger(CameraList camList)
{
    int result = 0;

    try
    {
    	// Send software trigger command to each camera
	    for(int i=0; i<camList.GetSize(); i++)
	   	{
	   		INodeMap & nodeMap = camList.GetByIndex(i)->GetNodeMap();
	   		// Execute software trigger
	        CCommandPtr ptrSoftwareTriggerCommand = nodeMap.GetNode("TriggerSoftware");
	        if (!IsAvailable(ptrSoftwareTriggerCommand) || !IsWritable(ptrSoftwareTriggerCommand))
	        {
	            cout << "Unable to execute trigger. Aborting..." << endl;
	            return -1;
	        }

	        ///////////////////////////
	        // Execute Software Trigger
	        ///////////////////////////
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





////////////////
// AcquireImages
////////////////
void AcquireImages(CameraList camList)
{
    int result = 0;
    int counter = 0;
	vector<int> v_time;
    int imgCount = 0;
	int numImages = 100;
	int numCams = camList.GetSize();
	CameraPtr camPtr[numCams];
	ImagePtr pResultImage[numCams];
	int start = getMilliCount();

	// Extract cameras from camList
	for(int camNum=0; camNum<numCams; camNum++)
	{
		camPtr[camNum] = camList.GetByIndex(camNum);
	}
	
	// Acquire Images from all the cameras at the same time
    try
    {
        int start = getMilliCount();

		// Empty image buffer before capturing images
		//emptyImageBuffer(camList);
		// Buffer emptied
	
		cout << "Acquiring Images" << endl;

		for(int imgNum=0; imgNum<numImages; imgNum++)
        {
            // Retrieve the next image from the trigger
            result = result | GrabNextImageByTrigger(camList);

			// Acquire Images from each camera
			for(int camNum=0; camNum<numCams; camNum++)
			{
				pResultImage[camNum] = camPtr[camNum]->GetNextImage();

				// Check if acquired image is incomplete
				if (pResultImage[camNum]->IsIncomplete())
				{
		        	cout << "Image incomplete with image status " << pResultImage[camNum]->GetImageStatus() << "..." << endl << endl;
					camPtr[camNum]->EndAcquisition();	
				}
			}
			
			// Save acquired image from each camera
			for(int camNum=0; camNum<numCams; camNum++)
			{
				// Convert image to Mono8
				ImagePtr convertedImage = pResultImage[camNum]->Convert(PixelFormat_Mono8, HQ_LINEAR);

				// Add image to buffer
				unsigned int rowBytes = (int)convertedImage->GetImageSize()/convertedImage->GetHeight();

				Mat imgTemp = Mat(convertedImage->GetHeight(),
				                  convertedImage->GetWidth(), CV_8UC1, convertedImage->GetData(), rowBytes);

				// Generate unique filename
				char filename[1000];
				sprintf(filename, "/home/umh-admin/LabWork/MultiCamSystem/images/Cam%d-%d.jpg", camNum, imgNum);
				
				// Write image
				imwrite(filename, imgTemp);
    		
				// Release Image
		       	pResultImage[camNum]->Release();
		   	}// End of camera loop

		   	// Calculate FPS
			int timeElapsed = getMilliSpan(start);
			v_time.push_back(timeElapsed);

			if (v_time.size() > 10)
			{
				int t = timeElapsed-v_time[v_time.size()-10];
				double fps = 10000.0/t;
				cout << fps << endl;
			}
				
		}// End image loop


#if 0
		// End acquisiton for all the cameras
		for(int i=0; i<numCams; i++)
		{
			camPtr[i]->EndAcquisition();
		}	  
#endif


  	}// End of Try block
	catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    //return result;
	cout << endl <<"Finished Saving Images" <<endl;
}



/*
void SaveImages(int camNum, vector<Mat> &imageBuffer)
{
	cout << "Saving from Camera: " << camNum << endl;	
	
	int result = 0;
	int numImages = 1000;
	vector<int> v_time;
	int start = getMilliCount();
	int imgCount = 1;

	std::this_thread::sleep_for(std::chrono::seconds(2));

	try
	{

		while(imgCount <= numImages)
		{
			//while(imageBuffer.empty())
				//std::this_thread::sleep_for(std::chrono::nanoseconds(1));
				//this_thread::yield();
			
			
			char fileName[1000];
		
			// Create unique filename
			if(camNum == 1)
				sprintf(fileName, "/home/umh-admin/Downloads/spinnaker_1_0_0_295_amd64/bin/bufferTest/Cam1/%d.jpg", imgCount);
			else if(camNum == 2)
				sprintf(fileName, "/home/umh-admin/Downloads/spinnaker_1_0_0_295_amd64/bin/bufferTest/Cam2/%d.jpg", imgCount);


			// Pop front image from the buffer
			if(!imageBuffer.empty())
			{
				// Don't save. Wait till the next image is available
				m.lock();
				Mat imgTemp = imageBuffer.front();
				imageBuffer.erase(imageBuffer.begin());
				m.unlock();
				imwrite(fileName, imgTemp);
#if 0
				Mat imgTemp = bufferList[camNum][imgCount];
				cout << "Saving. Cam: " << camNum << " Image: " << imgCount << endl;
				imwrite(fileName, imgTemp);
#endif

				// Calculate FPS
				int timeElapsed = getMilliSpan(start);
				v_time.push_back(timeElapsed);

				if (v_time.size() > 10)
				{
					int t = timeElapsed-v_time[v_time.size()-10];
					double fps = 10000.0/t;
					cout << fps << endl;
				}

				++imgCount;

			}
			
		}
		cout << "Saved Images: " << numImages << endl;
	}catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }
	//return result;
}
*/



///////////////////
// emptyImageBuffer
///////////////////
void emptyImageBuffer(CameraList camList)
{
	int numCams = camList.GetSize();
	ImagePtr pResultImage;

	try
	{
		// Extract cameras from camList
		for(int camNum=0; camNum<numCams; camNum++)
		{
			cout << "Emptying buffer for camera: " << camNum << endl;
			// Get Camera Pointer
			CameraPtr camPtr = camList.GetByIndex(camNum);

			// Get next image
			pResultImage = camPtr->GetNextImage();

			while(pResultImage != NULL)
			{
				pResultImage = camPtr->GetNextImage(10);
				pResultImage->Release();
			}
			pResultImage->Release();
			
		}
	}
	catch(Spinnaker::Exception &e)
    {
        cout << "Camera buffer cleared"  << endl;
    }
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

            // Configure Software Trigger
			result = ConfigureTrigger(nodeMap);

            if (result < 0)
            {
				cout << "Error configuring trigger" << endl;
                return result;
            }

            // Configure Cameras
            result = ConfigureCamera(pCam, nodeMap);
			if (result < 0)
            {
				cout << "Error configuring camera" << endl;
                return result;
            }
			
        }// End of initialization of trigger and camera


        // Acquire and save images from each camera
        AcquireImages(camList);
		

        // Deinitialize each camera
		sleep(3);
		for (int i = 0; i < camList.GetSize(); i++)
		{			
		   	// Select camera
		    pCam = camList.GetByIndex(i);

			pCam->EndAcquisition();

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
	cout << "RunMultipleCameras Function has ended" << endl;
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

	//Configure cameras and create separate thread for each camera to acquire images
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
