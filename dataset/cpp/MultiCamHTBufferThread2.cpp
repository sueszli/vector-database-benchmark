#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
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


// Trigger Config
int ConfigureTrigger(INodeMap & nodeMap, bool isPrimary);
int ResetTrigger(INodeMap & nodeMap);
int GrabNextImageByTrigger(INodeMap & nodeMap, CameraPtr pCam);

// Camera Config
int ConfigureCamera(CameraPtr pCam, INodeMap & nodeMap);
void emptyImageBuffer(CameraPtr pCam);
int RunMultipleCameras(CameraList camList);


// Miscellaneous
int getMilliCount();
int getMilliSpan(int nTimeStart);


// Use the following enum and global constant to select whether a software or
// hardware trigger is used.
enum triggerType
{
	SOFTWARE,
	HARDWARE
};


//const triggerType chosenTrigger = HARDWARE;
const triggerType chosenTrigger = HARDWARE;


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
		// Trigger disabled


		
        // Select trigger source
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (!IsAvailable(ptrTriggerSource) || !IsWritable(ptrTriggerSource))
        {
            cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }
		
		/////////////////////////////////////////
		// Set software trigger on primary camera
		/////////////////////////////////////////	
        if (isPrimary == true)
        {
            // Set trigger mode to software
            CEnumEntryPtr ptrTriggerSourceSoftware = ptrTriggerSource->GetEntryByName("Software");
            if (!IsAvailable(ptrTriggerSourceSoftware) || !IsReadable(ptrTriggerSourceSoftware))
            {
                cout << "Unable to set trigger mode to Software. Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceSoftware->GetValue());
            cout << "Primary camera trigger source set to software." << endl;
			//TriggerSource set to Software
    	}     
 
            
		//Enabling Hardware Trigger on Primary Camera and 3.3V
#if 0
            // Set trigger mode to hardware for all primary cameras ('Line2')
            CEnumEntryPtr ptrTriggerSourceHardware = ptrTriggerSource->GetEntryByName("Line2");
            if (!IsAvailable(ptrTriggerSourceHardware) || !IsReadable(ptrTriggerSourceHardware))
            {
                cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
                return -1;
            }

            ptrTriggerSource->SetIntValue(ptrTriggerSourceHardware->GetValue());
            cout << "Trigger source for primary camera set to hardware Line2" << endl;

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


 		
		/////////////////////////////////////////////////
		// Configure hardware trigger on secondary camera
		/////////////////////////////////////////////////
		else 
		{
			///////////////////////
            //TriggerSource = Line3
			///////////////////////
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
        }

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
		// Trigger enabled
		
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

    //
    // Prepare each camera to acquire images
    //
    // *** NOTES ***
    // For pseudo-simultaneous streaming, each camera is prepared as if it
    // were just one, but in a loop. Notice that cameras are selected with
    // an index. We demonstrate pseduo-simultaneous streaming because true
    // simultaneous streaming would require multiple process or threads,
    // which is too complex for an example.
    //
    // Serial numbers are the only persistent objects we gather in this
    // example, which is why a vector is created.
    //

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
			// Set acquisition mode to continuous


		    //
			// Turn off automatic exposure mode
			//
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
		
			cout << "Automatic exposure disabled..." << endl;

			//
			// Set exposure time manually; exposure time recorded in microseconds
			//
			CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
			if (!IsAvailable(ptrExposureTime) || !IsWritable(ptrExposureTime))
			{
				cout << "Unable to set exposure time. Aborting..." << endl << endl;
				return -1;
			}
		
			// Ensure desired exposure time does not exceed the maximum
			const double exposureTimeMax = ptrExposureTime->GetMax();
			double exposureTimeToSet = 2000.0;

			if (exposureTimeToSet > exposureTimeMax)
			{
				exposureTimeToSet = exposureTimeMax;
			}
		
			ptrExposureTime->SetValue(exposureTimeToSet);
		
			cout << "Exposure time set to " << exposureTimeToSet << " us..." << endl << endl;

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

		    // Retrieve Acquisition frame rate ------------------------------------//
		    CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
		    if (!IsAvailable(ptrAcquisitionFrameRate)
		            || !IsReadable(ptrAcquisitionFrameRate))
		    {
		        cout << "Unable to retrieve frame rate. Aborting..." << endl << endl;
		        return -1;
		    }

		    acquisitionFrameRate = static_cast<float>(ptrAcquisitionFrameRate->GetValue());
		    cout << "Acquisition Frame Rate: " << acquisitionFrameRate << endl << endl;

		    // Begin acquiring images
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
// GrabNextImgaeByTrigger
/////////////////////////

// This function retrieves a single image using the trigger. In this example,
// only a single image is captured and made available for acquisition - as such,
// attempting to acquire two images for a single trigger execution would cause
// the example to hang. This is different from other examples, whereby a
// constant stream of images are being captured and made available for image
// acquisition.
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





////////////////
// AcquireImages
////////////////
//
// This function  acquires images from all the initialized cameras
//
void AcquireImages(CameraPtr pCam, vector<Mat> &imageBuffer)
{
    int result = 0;
    int counter = 0;
	vector<int> v_time;
    int imgCount = 0;
	int numImages = 1000;


    try
    {
        int start = getMilliCount();

		// Empty image buffer before capturing images
		// emptyImageBuffer(pCam);
		// Buffer emptied
	
		cout << "Acquiring Images... " <<endl;

		for(int i=0; i<numImages; i++)
        {
            // Retrieve the next image from the trigger
            //result = result | GrabNextImageByTrigger(primaryNodeMap, primaryCam);
			
            if(result == -1) {
                cout << __LINE__ << i << endl;
                continue;
            }
			ImagePtr pResultImage = pCam->GetNextImage();

			if (pResultImage->IsIncomplete())
			{
               cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl << endl;
			}
			else
            {
		        // Convert image to BayerRG8
		        ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, DIRECTIONAL_FILTER);
				//ImagePtr convertedImage = pResultImage;

				// Add image to buffer
				// unsigned int rowBytes = (int)convertedImage->GetImageSize()/convertedImage->GetHeight();

				Mat imgTemp = Mat(convertedImage->GetHeight(),
		                      convertedImage->GetWidth(), CV_8UC(1), convertedImage->GetData());
				imageBuffer[i] = imgTemp.clone();

#if 0
				if(imgCount == 0)
					imageBuffer.pop_front();
				else
				{	
					cv::imshow("Camera-1", imageBuffer.front());
					imageBuffer.pop_front();
				}
#endif

                        
#if 0			
				// Calculate FPS
		   		int timeElapsed = getMilliSpan(start);
		        v_time.push_back(timeElapsed);

		        if (v_time.size() > 10)
		        {
		        	int t = timeElapsed-v_time[v_time.size()-10];
		            double fps = 10000.0/t;
		            cout << fps << endl;
		        }
#endif
				//cv::waitKey(1);

       		}

			// Release Image
            pResultImage->Release();
      	}

		// End acquisiton for the camera	  
		pCam->EndAcquisition();
        
  	}// End of Try block
	catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    //return result;
	cout << endl <<"Finished Acquiring Images... " <<endl;
}




void SaveImages(int camNum, vector<Mat> &imageBuffer)
{
	int result = 0;
	int numImages = 1000;
	vector<int> v_time;
	int start = getMilliCount();

	std::this_thread::sleep_for(std::chrono::seconds(2));
	cout << "Saving Images... " <<endl;

	try
	{

		for(int i=0; i<numImages; i++)
		{
			char fileName[1000];
		
			// Create unique filename
			if(camNum == 1)
				sprintf(fileName, "/home/umh-admin/Downloads/spinnaker_1_0_0_295_amd64/bin/bufferTest/Cam1/%d.jpg", i);
			else if(camNum == 2)
				sprintf(fileName, "/home/umh-admin/Downloads/spinnaker_1_0_0_295_amd64/bin/bufferTest/Cam2/%d.jpg", i);

			// Pop front image from the buffer
			Mat imgTemp = imageBuffer[i];

			imwrite(fileName, imgTemp);

			// Calculate FPS
		   	int timeElapsed = getMilliSpan(start);
		    v_time.push_back(timeElapsed);

		    if (v_time.size() > 10)
		    {
		    	int t = timeElapsed-v_time[v_time.size()-10];
		       	double fps = 10000.0/t;
		       	cout << fps << endl;
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



#if 0
void AcquireImages(vector<int> &imageBuffer)
{	
	for(int i=0; i<4000; i++)
	{
		cout << "Acquiring Image: " << i << endl;
		imageBuffer[i] = i;
	}	
	cout << "Finished acquiring images" << endl;
}


void SaveImages(vector<int> &imageBuffer)
{
	cout << endl << "Save Image thread sleeping for 5 seconds" << endl;
    //std::this_thread::sleep_for(std::chrono::seconds(1));

	for(int i=1; i<4000; i++)
	{
		cout << "Saved Image: " << imageBuffer[i] << endl;	
	}
	cout << "Finished saving images" << endl;
}

#endif


///////////////////
// emptyImageBuffer
///////////////////

//
// Function to clear buffer before acquiring images
//
void emptyImageBuffer(CameraPtr pCam)
{	
		try
		{
			// Get next image
			ImagePtr pResultImage = pCam->GetNextImage();

			while(pResultImage != NULL)
			{
				pResultImage = pCam->GetNextImage(1000);
				pResultImage->Release();

			}
			pResultImage->Release();
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
    CameraPtr pCam1 = NULL;
	CameraPtr pCam2 = NULL;
    bool isPrimary = false;

	// Vector of Buffers. Each buffer is also a vector
	//vector<vector<int>*> bufferList;
		
	
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

            result = ConfigureTrigger(nodeMap, isPrimary);
            if (result < 0)
            {
				cout << "Error configuring trigger" << endl;
                return result;
            }

            result = ConfigureCamera(pCam, nodeMap);
			if (result < 0)
            {
				cout << "Error configuring camera" << endl;
                return result;
            }

			// Create buffer to store images
			//int bufferSize = 1000;
			//vector<int> *buffer = new vector<int> (4000);
			//bufferList.push_back(buffer);
			
        }// End of initialization of trigger and camera
		
		vector<Mat> imageBuffer1 (5000);
		vector<Mat> imageBuffer2 (5000);

		cout << "Testing " << __LINE__ << endl;
		
		pCam1 = camList.GetByIndex(0);
		//pCam2 = camList.GetByIndex(1);

		auto f1 = std::async(std::launch::async, AcquireImages, pCam1, std::ref(imageBuffer1));
		auto f2 = std::async(std::launch::async, SaveImages, 1, std::ref(imageBuffer1));
		auto f3 = std::async(std::launch::async, AcquireImages, pCam2, std::ref(imageBuffer2));
		auto f4 = std::async(std::launch::async, SaveImages, 2, std::ref(imageBuffer2));
		
#if 0
		// For each camera, create a separate thread to acquire images and store in corresponding buffer
        for(int i=0; i<camList.GetSize(); i++)
		{
			//pCam = camList.GetByIndex(i);			
			//auto ret = std::async(std::launch::async, AcquireImages, /*pCam, */std::ref(*bufferList[i]));
			auto f = std::async(std::launch::async, AcquireImages, std::ref(imageBuffer));
			
			if (result < 0)
            {
				cout << "Error acquiring images from cameras" << endl;
                return result;
            }

		}

		
		// Save images from buffers
		for(int i=0; i<camList.GetSize(); i++)
		{			
			//auto ret2 = std::async(std::launch::async, saveImages, std::ref(*bufferList[i]));
			auto f2 = std::async(std::launch::async, SaveImages, std::ref(imageBuffer));
			
			if (result < 0)
            {
				cout << "Error saving images from cameras" << endl;
                return result;
            }

		}
#endif

        // Deinitialize each camera
        for (int i = 0; i < camList.GetSize(); i++)
        {
			// Delete buffer associated with the camera
			//delete bufferList[i];
			
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
