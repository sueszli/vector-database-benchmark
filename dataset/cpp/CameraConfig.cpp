// C++
#include <iostream>
#include <sstream>

// Spinnaker
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;

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
