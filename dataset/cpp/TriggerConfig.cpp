#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <sstream>

// Spinnaker
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#endif

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;


// Functions
int ConfigureTrigger(INodeMap & nodeMap, bool isPrimary);
int ResetTrigger(INodeMap & nodeMap);

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
