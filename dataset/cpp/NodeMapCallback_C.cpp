//=============================================================================
// Copyright © 2015 Point Grey Research, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of 
// Point Grey Research, Inc. ("Confidential Information"). You shall not
// disclose such Confidential Information and shall use it only in 
// accordance with the terms of the "License Agreement" that you 
// entered into with PGR in connection with this software.
//
// UNLESS OTHERWISE SET OUT IN THE LICENSE AGREEMENT, THIS SOFTWARE IS 
// PROVIDED ON AN “AS-IS” BASIS AND POINT GREY RESEARCH INC. MAKES NO 
// REPRESENTATIONS OR WARRANTIES ABOUT THE SOFTWARE, EITHER EXPRESS 
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OR 
// CONDITIONS OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
// NON-INFRINGEMENT. POINT GREY RESEARCH INC. SHALL NOT BE LIABLE FOR ANY 
// DAMAGES, INCLUDING BUT NOT LIMITED TO ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, PUNITIVE, OR CONSEQUENTIAL DAMAGES, OR ANY LOSS OF PROFITS, 
// REVENUE, DATA OR DATA USE, ARISING OUT OF OR IN CONNECTION WITH THIS 
// SOFTWARE OR OTHERWISE SUFFERED BY YOU AS A RESULT OF USING, MODIFYING 
// OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

/**
 *	@example NodeMapCallback_C.cpp
 *
 *	@brief NodeMapCallback_C.cpp shows how to use nodemap callbacks. It
 *	relies on information provided in the Enumeration_C, Acquisition_C, and
 *	NodeMapInfo_C examples. As callbacks are very similar to events, it may be 
 *	a good idea to explore this example prior to tackling the events examples.
 *
 *	This example focuses on creating, registering, using, and unregistering
 *	callbacks. A callback requires a certain function signature, which allows 
 *	it to be registered to and access a node. Events, while slightly more 
 *	complex, follow this same pattern.
 *
 *	Once comfortable with NodeMapCallback_C, we suggest checking out any of the
 *	events examples: DeviceEvents_C, EnumerationEvents_C, ImageEvents_C, or 
 *	Logging_C.
 */

#include "SpinnakerC.h"
#include "stdio.h"
#include "string.h"

// Compiler warning C4996 suppressed due to deprecated strcpy() and sprintf() 
// functions on Windows platform.
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64 
	#pragma warning(disable : 4996)
#endif

// This macro helps with C-strings.
#define MAX_BUFF_LEN 256

// This is the first of two callback functions. Notice the function signature.
// This callback function will be registered to the height node.
void onHeightNodeUpdate(spinNodeHandle hNode) 
{
	spinError err = SPINNAKER_ERR_SUCCESS;
	int64_t height = 0;

	err = spinIntegerGetValue(hNode, &height);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve height. Non-fatal error %d...\n\n", err);
		return;
	}

	printf("Height callback message:\n");
	printf("\tLook! Height changed to %d...\n\n", (int)height);
}

// This is the second of two callback functions. Notice that despite different
// names, everything else is exactly the same as the first. This callback 
// function will be registered to the gain node.
void onGainNodeUpdate(spinNodeHandle hNode)
{
	spinError err = SPINNAKER_ERR_SUCCESS;
	double gain = 0.0;

	err = spinFloatGetValue(hNode, &gain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve gain. Non-fatal error %d...\n\n", err);
		return;
	}

	printf("Gain callback message:\n");
	printf("\tLook now! Gain changed to %f...\n\n", gain);
}

// This function prepares the example by disabling automatic gain, creating two 
// callbacks, and registering them to their respective nodes.
spinError ConfigureCallbacks(spinNodeMapHandle hNodeMap, spinNodeCallbackHandle* callbackHeight, spinNodeCallbackHandle* callbackGain)
{
	spinError err = SPINNAKER_ERR_SUCCESS;

	printf("\n\n*** CALLBACKS CONFIGURATION ***\n\n");

	//
	// Turn off automatic gain
	//
	// *** NOTES ***
	// Automatic gain prevents the manual configuration of gain and needs to
	// be turned off for this example.
	//
	// *** LATER ***
	// Automatic gain is turned off at the end of the example in order to
	// restore the camera to its default state.
	//
	spinNodeHandle hGainAuto = NULL;
	spinNodeHandle hGainAutoOff = NULL;
	int64_t gainAutoOff = 0;

	err = spinNodeMapGetNode(hNodeMap, "GainAuto", &hGainAuto);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to disable automatic gain (node retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinEnumerationGetEntryByName(hGainAuto, "Off", &hGainAutoOff);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to disable automatic gain (enum entry retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinEnumerationEntryGetIntValue(hGainAutoOff, &gainAutoOff);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to disable automatic gain (enum entry int value retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinEnumerationSetIntValue(hGainAuto, gainAutoOff);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to disable automatic gain (enum entry setting). Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Automatic gain disabled...\n");

	//
	// Register callback to height node
	//
	// *** NOTES ***
	// Callbacks need to be registered to nodes, which should be writable 
	// if the callback is to ever be triggered. Notice that callback 
	// registration a handle - this handle is important at the end of the
	// example for deregistration.
	//
	// *** LATER ***
	// Each callback needs to be unregistered individually before releasing
	// the system or an exception will be thrown.
	//
	spinNodeHandle hHeight = NULL;

	err = spinNodeMapGetNode(hNodeMap, "Height", &hHeight);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register height callback (node retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinNodeRegisterCallback(hHeight, onHeightNodeUpdate, callbackHeight);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register height callback (callback registration). Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Height callback registered...\n");

	//
	// Register callback to gain node
	//
	// *** NOTES ***
	// Depending on the specific goal of the function, it can be important
	// to notice the node type that a callback is registered to. Notice in
	// the callback functions above that the callback registered to height 
	// casts its node as an integer whereas the callback registered to gain
	// casts as a float.
	//
	// *** LATER ***
	// Each callback needs to be unregistered individually before releasing
	// the system or an exception will be thrown.
	//
	spinNodeHandle hGain = NULL;

	err = spinNodeMapGetNode(hNodeMap, "Gain", &hGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register gain callback (callback registration). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinNodeRegisterCallback(hGain, onGainNodeUpdate, callbackGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register gain callback (callback registration). Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Gain callback registered...\n\n");

	return err;
}

// This function demonstrates the triggering of the nodemap callbacks. First it 
// changes height, which executes the callback registered to the height node, and
// then it changes gain, which executes the callback registered to the gain node.
spinError ChangeHeightAndGain(spinNodeMapHandle hNodeMap)
{
	spinError err = SPINNAKER_ERR_SUCCESS;

	printf("\n*** CHANGING HEIGHT & GAIN ***\n\n");

	//
	// Change height to trigger height callback
	//
	// *** NOTES ***
	// Notice that changing the height only triggers the callback function
	// registered to the height node.
	//
	spinNodeHandle hHeight = NULL;
	int64_t heightToSet = 0;
	int64_t heightMax = 0;

	err = spinNodeMapGetNode(hNodeMap, "Height", &hHeight);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to change height (node retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinIntegerGetMax(hHeight, &heightMax);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to change height (max retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	heightToSet = heightMax;

	printf("Regular function message:\n");
	printf("\tHeight about to be set to %d...\n\n", (int)heightToSet);

	err = spinIntegerSetValue(hHeight, heightToSet);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to change height (value setting). Aborting with error %d...\n\n", err);
		return err;
	}

	//
	// Change gain to trigger gain callback
	//
	// *** NOTES ***
	// The same is true of changing the gain node; changing a node will 
	// only ever trigger the callback function (or functions) currently
	// registered to it.
	//
	spinNodeHandle hGain = NULL;
	double gainToSet = 0.0;
	double gainMax = 0.0;

	err = spinNodeMapGetNode(hNodeMap, "Gain", &hGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register gain callback (callback registration). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinFloatGetMax(hGain, &gainMax);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to change gain (max retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	gainToSet = gainMax / 2.0;

	printf("Regular function message:\n");
	printf("\tGain about to be set to %f...\n\n", gainToSet);

	err = spinFloatSetValue(hGain, gainToSet);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to change gain (value setting). Aborting with error %d...\n\n", err);
		return err;
	}

	return err;
}

// This function cleans up the example by deregistering the callbacks and 
// turning automatic gain back on.
spinError ResetCallbacks(spinNodeMapHandle hNodeMap, spinNodeCallbackHandle callbackHeight, spinNodeCallbackHandle callbackGain)
{
	spinError err = SPINNAKER_ERR_SUCCESS;

	//
	// Deregister height callback
	//
	// *** NOTES ***
	// It is important to deregister each callback function from each node 
	// that it is registered to.
	//
	spinNodeHandle hHeight = NULL;

	err = spinNodeMapGetNode(hNodeMap, "Height", &hHeight);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register height callback (node retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinNodeDeregisterCallback(hHeight, callbackHeight);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to deregister height callback (callback deregistration). Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Height callback deregistered...\n");

	//
	// Deregister gain callback
	//
	// *** NOTES ***
	// It is important to deregister each callback function from each node 
	// that it is registered to.
	//
	spinNodeHandle hGain = NULL;

	err = spinNodeMapGetNode(hNodeMap, "Gain", &hGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to register gain callback (callback registration). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinNodeDeregisterCallback(hGain, callbackGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to deregister gain callback (callback deregistration). Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Gain callback deregistered...\n");

	//
	// Turn automatic gain back on
	//
	// *** NOTES ***
	// Automatic gain is turned on in order to return the camera to its
	// default state.
	//
	spinNodeHandle hGainAuto = NULL;
	spinNodeHandle hGainAutoContinuous = NULL;
	int64_t gainAutoContinuous = 0;

	err = spinNodeMapGetNode(hNodeMap, "GainAuto", &hGainAuto);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to disable automatic gain (node retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinNodeMapGetNode(hNodeMap, "GainAuto", &hGainAuto);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to enable automatic gain (node retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinEnumerationGetEntryByName(hGainAuto, "Continuous", &hGainAutoContinuous);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to enable automatic gain (enum entry retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinEnumerationEntryGetIntValue(hGainAutoContinuous, &gainAutoContinuous);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to enable automatic gain (enum entry int value retrieval). Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinEnumerationSetIntValue(hGainAuto, gainAutoContinuous);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to enable automatic gain (enum entry setting). Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Automatic gain turned back on...\n\n");

	return err;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo_C example for more in-depth comments on
// printing device information from the nodemap.
spinError PrintDeviceInfo(spinNodeMapHandle hNodeMap)
{
	spinError err = SPINNAKER_ERR_SUCCESS;
	unsigned int i = 0;

	printf("\n*** DEVICE INFORMATION ***\n\n");

	// Retrieve device information category node
	spinNodeHandle hDeviceInformation = NULL;

	err = spinNodeMapGetNode(hNodeMap, "DeviceInformation", &hDeviceInformation);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve node (non-fatal error %d)...\n\n", err);
		return err;
	}

	// Retrieve number of nodes within device information node
	size_t numFeatures = 0;

	err = spinCategoryGetNumFeatures(hDeviceInformation, &numFeatures);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve number of nodes (non-fatal error %d)...\n\n", err);
		return err;
	}

	// Iterate through nodes and print information
	for (i = 0; i < numFeatures; i++)
	{
		spinNodeHandle hFeatureNode = NULL;

		err = spinCategoryGetFeatureByIndex(hDeviceInformation, i, &hFeatureNode);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			printf("Unable to retrieve node. Non-fatal error %d...\n\n", err);
			continue;
		}

		spinNodeType featureType = UnknownNode;

		err = spinNodeGetType(hFeatureNode, &featureType);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			printf("Unable to retrieve node type. Non-fatal error %d...\n\n", err);
			continue;
		}

		char featureName[MAX_BUFF_LEN];
		size_t lenFeatureName = MAX_BUFF_LEN;
		char featureValue[MAX_BUFF_LEN];
		size_t lenFeatureValue = MAX_BUFF_LEN;

		err = spinNodeGetName(hFeatureNode, featureName, &lenFeatureName);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			strcpy(featureName, "Unknown name");
		}

		err = spinNodeToString(hFeatureNode, featureValue, &lenFeatureValue);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			strcpy(featureValue, "Unknown value");
		}

		printf("%s: %s\n", featureName, featureValue);
	}

	return err;
}

// This function acts as the body of the example; please see NodeMapInfo_C 
// example for more in-depth comments on setting up cameras.
spinError RunSingleCamera(spinCamera hCam)
{
	spinError err = SPINNAKER_ERR_SUCCESS;

	// Retrieve TL device nodemap and print device information
	spinNodeMapHandle hNodeMapTLDevice = NULL;

	err = spinCameraGetTLDeviceNodeMap(hCam, &hNodeMapTLDevice);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve TL device nodemap (non-fatal error %d)...\n\n", err);
	}
	else
	{
		err = PrintDeviceInfo(hNodeMapTLDevice);
	}

	// Initialize camera
	err = spinCameraInit(hCam);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to initialize camera. Aborting with error %d...\n\n", err);
		return err;
	}

	// Retrieve GenICam nodemap
	spinNodeMapHandle hNodeMap = NULL;

	err = spinCameraGetNodeMap(hCam, &hNodeMap);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve GenICam nodemap. Aborting with error %d...\n\n", err);
		return err;
	}

	// Configure callbacks
	spinNodeCallbackHandle callbackHeight = NULL;
	spinNodeCallbackHandle callbackGain = NULL;

	err = ConfigureCallbacks(hNodeMap, &callbackHeight, &callbackGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		return err;
	}

	// Change height and gain to trigger callbacks
	err = ChangeHeightAndGain(hNodeMap);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		return err;
	}

	// Reset callbacks
	err = ResetCallbacks(hNodeMap, callbackHeight, callbackGain);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		return err;
	}

	// Deinitialize camera
	err = spinCameraDeInit(hCam);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to deinitialize camera. Non-fatal error %d...\n\n", err);
		return err;
	}

	return err;
}

// Example entry point; please see Enumeration_C example for more in-depth
// comments on preparing and cleaning up the system.
int main(/*int argc, char** argv*/)
{
	spinError errReturn = SPINNAKER_ERR_SUCCESS;
	spinError err = SPINNAKER_ERR_SUCCESS;
	unsigned int i = 0;

	// Print application build information
	printf("Application build date: %s %s \n\n", __DATE__, __TIME__);

	// Retrieve singleton reference to system object
	spinSystem hSystem = NULL;

	err = spinSystemGetInstance(&hSystem);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve system instance. Aborting with error %d...\n\n", err);
		return err;
	}

	// Retrieve list of cameras from the system
	spinCameraList hCameraList = NULL;

	err = spinCameraListCreateEmpty(&hCameraList);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to create camera list. Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinSystemGetCameras(hSystem, hCameraList);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve camera list. Aborting with error %d...\n\n", err);
		return err;
	}

	// Retrieve number of cameras
	size_t numCameras = 0;

	err = spinCameraListGetSize(hCameraList, &numCameras);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to retrieve number of cameras. Aborting with error %d...\n\n", err);
		return err;
	}

	printf("Number of cameras detected: %u\n\n", (unsigned int)numCameras);

	// Finish if there are no cameras
	if (numCameras == 0)
	{
		// Clear and destroy camera list before releasing system
		err = spinCameraListClear(hCameraList);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
			return err;
		}

		err = spinCameraListDestroy(hCameraList);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
			return err;
		}

		// Release system
		err = spinSystemReleaseInstance(hSystem);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			printf("Unable to release system instance. Aborting with error %d...\n\n", err);
			return err;
		}

		printf("Not enough cameras!\n");
		printf("Done! Press Enter to exit...\n");
		getchar();

		return -1;
	}

	// Run example on each camera
	for (i = 0; i < numCameras; i++)
	{
		printf("\nRunning example for camera %d...\n", i);

		// Select camera
		spinCamera hCamera = NULL;

		err = spinCameraListGet(hCameraList, i, &hCamera);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			printf("Unable to retrieve camera from list. Aborting with error %d...\n\n", err);
			errReturn = err;
		}
		else
		{
			// Run example
			err = RunSingleCamera(hCamera);
			if (err != SPINNAKER_ERR_SUCCESS)
			{
				errReturn = err;
			}
		}

		// Release camera
		err = spinCameraRelease(hCamera);
		if (err != SPINNAKER_ERR_SUCCESS)
		{
			errReturn = err;
		}

		printf("Camera %d example complete...\n\n", i);
	}

	// Clear and destroy camera list before releasing system
	err = spinCameraListClear(hCameraList);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
		return err;
	}

	err = spinCameraListDestroy(hCameraList);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
		return err;
	}

	// Release system
	err = spinSystemReleaseInstance(hSystem);
	if (err != SPINNAKER_ERR_SUCCESS)
	{
		printf("Unable to release system instance. Aborting with error %d...\n\n", err);
		return err;
	}

	printf("\nDone! Press Enter to exit...\n");
	getchar();

	return errReturn;
}
