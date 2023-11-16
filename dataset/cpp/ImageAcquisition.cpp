
#include "../headers/ImageAcquisition.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;


// Functions
int GrabNextImageByTrigger(INodeMap & nodeMap, CameraPtr pCam);
int AcquireImages(CameraList camList);
void emptyImageBuffer(CameraList camList);

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
int AcquireImages(CameraList camList)
{
    int result = 0;
    CameraPtr pCam = NULL, primaryCam = NULL;
    int counter = 0;

    try
    {
        char key = 0;


        int start = getMilliCount();
        vector<int> v_time;

        unsigned int imageCnt = 0;
        //Mat src[2];
        
        primaryCam = camList.GetByIndex(0);
        INodeMap & primaryNodeMap = primaryCam->GetNodeMap();

		// Empty image buffer before capturing images
		//emptyImageBuffer(camList);
		// Buffer emptied

        while(imageCnt < 1000)
        {
			// Get user input
            //cout << o"Press the Enter key to initiate software trigger." << endl;
            //getchar();

            // Retrieve the next image from the trigger
            //result = result | GrabNextImageByTrigger(primaryNodeMap, primaryCam);
			
            if(result == -1) {
                cout << __LINE__ << imageCnt << endl;
                continue;
            }
        
        
			int k = 0;    
            for(int i = 0; i < camList.GetSize(); ++i)
            {	
                // Select camera
                pCam = camList.GetByIndex(i);
                try
                {
                    //ImagePtr pResultImage = pCam->GetNextImage();

                    //cout << "Grabbed image from Camera: " << i << endl;

                    //if (pResultImage->IsIncomplete())
                    //{
                        //cout << "Image incomplete with image status " 
                            //<< pResultImage->GetImageStatus() << "..." << endl << endl;
                    //}
                    //else
                    {
                        // Convert image to BayerRG8
                        //ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);

						char fileName[1000];
						// Create a unique filename
						sprintf(fileName, "/home/umh-admin/Downloads/"
                                "spinnaker_1_0_0_295_amd64/bin/trigger_test/%d/%d.jpg", i+1, imageCnt);

						// Save image with unique filename
						//convertedImage->Save(fileName);
						//pResultImage->Save(fileName);
						


#if 0
                        unsigned int rowBytes
                            = (int)convertedImage->GetImageSize()/convertedImage->GetHeight();

                        src[i] = Mat(convertedImage->GetHeight(),
                                convertedImage->GetWidth(), CV_8UC1, convertedImage->GetData(),
                                rowBytes);

                        resize(src[i], src[i], Size(640, 480), 0,0, INTER_LINEAR);

                        cv::imshow("Camera-" + to_string(i), src[i]);

                        key = cv::waitKey(1);
#endif

#if 0
                        long int sysTime = time(0);

                        char temp[1000];
                        sprintf(temp, "/home/umh-admin/Downloads/"
                                "spinnaker_1_0_0_295_amd64/bin/trigger_test/%d/"
                                "%d--%Ld--%d.jpg", i+1, i+1, sysTime, imageCnt);
                        imwrite(temp, src);
#endif
                        int timeElapsed = getMilliSpan(start);
                        v_time.push_back(timeElapsed);

                        if (v_time.size() > 10)
                        {
                            int t = timeElapsed-v_time[v_time.size()-10];
                            double fps = 10000.0/t;
                            cout << fps << endl;
                        }
                        ++counter;
                    }
                    //pResultImage->Release();
					
                }
                catch (Spinnaker::Exception &e)
                {
                    cout << "Error: " << e.what() << endl;
                    result = -1;
                }
            }
            ++imageCnt;
        }
#if 0
        //Display Stats
        int milliSecondsElapsed = getMilliSpan(start);
        cout << "Capture Time in milliseconds: " << milliSecondsElapsed << "." << endl;
        //cout << "Images saved: " << counter << endl;
        //cout << "Images saved per second: " << (counter*1000)/milliSecondsElapsed << endl;
        cout << "Calculated FPS: " << (counter*1000)/milliSecondsElapsed << endl;
#endif

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


///////////////////
// emptyImageBuffer
///////////////////

//
// Function to clear buffer before acquiring images
//
void emptyImageBuffer(CameraList camList)
{
	for(int i = 0; i < camList.GetSize(); ++i)
	{
		// Select camera
        CameraPtr pCam = camList.GetByIndex(i);
		
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
        	cout << "Camera: " << i << " buffer cleared"  << endl;
        }
		
	}
	
}
