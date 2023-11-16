/*! @file NUData.cpp
    @brief Implementation of an abstract NUData class
    @author Jason Kulk

    @author Jason Kulk

  Copyright (c) 2009, 2010 Jason Kulk

    This file is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with NUbot.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "NUData.h"
#include <algorithm>
#include <cctype>

#include "debug.h"
#include "debugverbositynuactionators.h"
#include "debugverbositynusensors.h"
#include <iostream>

/*! @brief Streaming operator for id_t* objects. Copies the object at the pointer location.
    @param output The output stream on which to write.
    @param id The object to be written to the stream.
    @return The output stream with the object written to it.
 */
std::ostream& operator << (std::ostream& output, const NUData::id_t* id)
{
    output << id->Id << " ";
    output << id->Name << " ";
    return output;
}

/*! @brief Streaming operator for id_t* objects. Reads the object to a new object.
    @param input The input stream containing the data.
    @param id The pointer where the new object will be written.
    @return The input stream post read.
 */
std::istream& operator >> (std::istream& input, NUData::id_t* id)
{
    //id = new NUData::id_t();
    input >> id->Id;
    input >> id->Name;
    return input;
}

int curr_id = 0;
std::vector<NUData::id_t*> NUData::m_common_ids;
const NUData::id_t NUData::All(curr_id++, "All", NUData::m_common_ids);							//0
const NUData::id_t NUData::Head(curr_id++, "Head", NUData::m_common_ids);
const NUData::id_t NUData::Body(curr_id++, "Body", NUData::m_common_ids);
const NUData::id_t NUData::LArm(curr_id++, "LArm", NUData::m_common_ids);
const NUData::id_t NUData::RArm(curr_id++, "RArm", NUData::m_common_ids);
const NUData::id_t NUData::LHand(curr_id++, "LHand", NUData::m_common_ids);
const NUData::id_t NUData::RHand(curr_id++, "RHand", NUData::m_common_ids);
const NUData::id_t NUData::Torso(curr_id++, "Torso", NUData::m_common_ids);
const NUData::id_t NUData::LLeg(curr_id++, "LLeg", NUData::m_common_ids);
const NUData::id_t NUData::RLeg(curr_id++, "RLeg", NUData::m_common_ids);
const NUData::id_t NUData::LFoot(curr_id++, "LFoot", NUData::m_common_ids);
const NUData::id_t NUData::RFoot(curr_id++, "RFoot", NUData::m_common_ids);						//11
const NUData::id_t NUData::NumCommonGroupIds(curr_id++, "NumCommonGroupIds", NUData::m_common_ids);

const NUData::id_t NUData::HeadRoll(curr_id++, "HeadRoll", NUData::m_common_ids);				//13
const NUData::id_t NUData::HeadPitch(curr_id++, "HeadPitch", NUData::m_common_ids);
const NUData::id_t NUData::HeadYaw(curr_id++, "HeadYaw", NUData::m_common_ids);
const NUData::id_t NUData::NeckRoll(curr_id++, "NeckRoll", NUData::m_common_ids);
const NUData::id_t NUData::NeckPitch(curr_id++, "NeckPitch", NUData::m_common_ids);
const NUData::id_t NUData::NeckYaw(curr_id++, "NeckYaw", NUData::m_common_ids);
const NUData::id_t NUData::LShoulderRoll(curr_id++, "LShoulderRoll", NUData::m_common_ids);
const NUData::id_t NUData::LShoulderPitch(curr_id++, "LShoulderPitch", NUData::m_common_ids);
const NUData::id_t NUData::LShoulderYaw(curr_id++, "LShoulderYaw", NUData::m_common_ids);
const NUData::id_t NUData::LElbowRoll(curr_id++, "LElbowRoll", NUData::m_common_ids);
const NUData::id_t NUData::LElbowPitch(curr_id++, "LElbowPitch", NUData::m_common_ids);
const NUData::id_t NUData::LElbowYaw(curr_id++, "LElbowYaw", NUData::m_common_ids);
const NUData::id_t NUData::RShoulderRoll(curr_id++, "RShoulderRoll", NUData::m_common_ids);
const NUData::id_t NUData::RShoulderPitch(curr_id++, "RShoulderPitch", NUData::m_common_ids);
const NUData::id_t NUData::RShoulderYaw(curr_id++, "RShoulderYaw", NUData::m_common_ids);
const NUData::id_t NUData::RElbowRoll(curr_id++, "RElbowRoll", NUData::m_common_ids);
const NUData::id_t NUData::RElbowPitch(curr_id++, "RElbowPitch", NUData::m_common_ids);
const NUData::id_t NUData::RElbowYaw(curr_id++, "RElbowYaw", NUData::m_common_ids);
const NUData::id_t NUData::TorsoRoll(curr_id++, "TorsoRoll", NUData::m_common_ids);
const NUData::id_t NUData::TorsoPitch(curr_id++, "TorsoPitch", NUData::m_common_ids);
const NUData::id_t NUData::TorsoYaw(curr_id++, "TorsoYaw", NUData::m_common_ids);
const NUData::id_t NUData::LHipRoll(curr_id++, "LHipRoll", NUData::m_common_ids);
const NUData::id_t NUData::LHipPitch(curr_id++, "LHipPitch", NUData::m_common_ids);
const NUData::id_t NUData::LHipYawPitch(curr_id++, "LHipYawPitch", NUData::m_common_ids);
const NUData::id_t NUData::LHipYaw(curr_id++, "LHipYaw", NUData::m_common_ids);
const NUData::id_t NUData::LKneePitch(curr_id++, "LKneePitch", NUData::m_common_ids);
const NUData::id_t NUData::LAnkleRoll(curr_id++, "LAnkleRoll", NUData::m_common_ids);
const NUData::id_t NUData::LAnklePitch(curr_id++, "LAnklePitch", NUData::m_common_ids);
const NUData::id_t NUData::RHipRoll(curr_id++, "RHipRoll", NUData::m_common_ids);
const NUData::id_t NUData::RHipPitch(curr_id++, "RHipPitch", NUData::m_common_ids);
const NUData::id_t NUData::RHipYawPitch(curr_id++, "RHipYawPitch", NUData::m_common_ids);
const NUData::id_t NUData::RHipYaw(curr_id++, "RHipYaw", NUData::m_common_ids);
const NUData::id_t NUData::RKneePitch(curr_id++, "RKneePitch", NUData::m_common_ids);
const NUData::id_t NUData::RAnkleRoll(curr_id++, "RAnkleRoll", NUData::m_common_ids);
const NUData::id_t NUData::RAnklePitch(curr_id++, "RAnklePitch", NUData::m_common_ids);
const NUData::id_t NUData::NumJointIds(curr_id++, "NumJointIds", NUData::m_common_ids);			//48

const NUData::id_t NUData::NumCommonIds(curr_id++, "NumCommonIds", NUData::m_common_ids);		//49 Remember that m_num_common_ids needs to be manually set to this value

void NUData::addDevices(const std::vector<std::string>& hardwarenames)
{
    std::vector<std::string> names = standardiseNames(hardwarenames);
    std::vector<id_t*>& ids = m_ids_copy;

    for (size_t i=0; i<names.size(); i++)
    {	// for each name compare it to the name of every id
        for (size_t j=NumCommonGroupIds.Id+1; j<ids.size(); j++)
        {
            id_t& id = *(ids[j]);
            if (id == names[i])
            {   // if the name matches the id, then add the actionator to m_available_ids and update the map
                #if DEBUG_NUSENSORS_VERBOSITY > 4 or DEBUG_NUACTIONATORS_VERBOSITY > 4
                    debug << id.Name << " == " << names[i] << std::endl;
                #endif
                if (find(m_available_ids.begin(), m_available_ids.end(), id.Id) == m_available_ids.end())
                    m_available_ids.push_back(id.Id);
                if (find(m_id_to_indices[id.Id].begin(), m_id_to_indices[id.Id].end(), id.Id) == m_id_to_indices[id.Id].end())
                    m_id_to_indices[id.Id].push_back(id.Id);
                break;
            }
        }
    }

    for (size_t i=0; i<ids.size(); i++)
    {	// fill in the groups
        for (size_t j=0; j<ids.size(); j++)
        {
            if (not m_id_to_indices[j].empty() and belongsToGroup(*ids[j], *ids[i]))
            {
                if (find(m_id_to_indices[i].begin(), m_id_to_indices[i].end(), j) == m_id_to_indices[i].end())
                    m_id_to_indices[i].push_back(j);
            }
        }
    }

    #if DEBUG_NUACTIONATORS_VERBOSITY > 0 or DEBUG_NUSENSORS_VERBOSITY > 0
        debug << "NUData::addDevices:" << std::endl;
        printMap(debug);
    #endif
}

/*! @brief Returns a vector containing the standardised versions of the vector containing hardware names
    @param hardwarenames a list of hardwarenames
    @return a vector with the simplified names
 */
std::vector<std::string> NUData::standardiseNames(const std::vector<std::string>& hardwarenames)
{
    std::vector<std::string> simplenames;
    for (size_t i=0; i<hardwarenames.size(); i++)
    {
        std::string simplename = getStandardName(hardwarenames[i]);
        if (simplenames.empty())
            simplenames.push_back(simplename);
        else if (simplename.compare(simplenames.back()) != 0)
            simplenames.push_back(simplename);
    }
    return simplenames;
}

/*! @brief Returns a simplified version of the hardwarename, formatting is removed.
    @param hardwarename the string to simplify
    @return the simplename
*/
std::string NUData::getStandardName(const std::string& hardwarename)
{
    std::string simplename, currentletter;
    // compare each letter to a space, an underscore, a forward slash, a backward slash and a period
    for (size_t j=0; j<hardwarename.size(); j++)
    {
        currentletter = hardwarename.substr(j, 1);
        if (currentletter.compare(std::string(" ")) != 0 && currentletter.compare(std::string("_")) != 0 && currentletter.compare(std::string("/")) != 0 && currentletter.compare(std::string("\\")) != 0 && currentletter.compare(std::string(".")) != 0)
            simplename += currentletter[0];
    }

    // Replace "Left"/"Right" with L/R and move to front of name
    size_t Left = simplename.find("Left");
    size_t Right = simplename.find("Right");
    if (Left != std::string::npos)
    {
        simplename.erase(Left, 4);
        simplename.insert(0, "L");
    }
    if (Right != std::string::npos)
    {
        simplename.erase(Right, 5);
        simplename.insert(0, "R");
    }

    // Replace plurals (ears, eyes)
    size_t Ears = simplename.find("Ears");
    size_t Eyes = simplename.find("Eyes");
    if (Ears != std::string::npos)
        simplename.replace(Ears, 4, "Ear");
    if (Eyes != std::string::npos)
        simplename.replace(Ears, 4, "Eye");

    // Replace ChestBoard with Chest
    size_t ChestBoard = simplename.find("ChestBoard");
    if (ChestBoard != std::string::npos)
        simplename.replace(ChestBoard, 10, "Chest");

    // Replace LFace with LEye and RFace with REye
    size_t LFace = simplename.find("LFace");
    size_t RFace = simplename.find("RFace");
    if (LFace != std::string::npos)
        simplename.replace(LFace, 5, "LEye");
    if (RFace != std::string::npos)
        simplename.replace(RFace, 5, "REye");

    // Remove colours
    size_t Red = simplename.find("Red");
    if (Red != std::string::npos)
        simplename.erase(Red, 3);
    size_t Green = simplename.find("Green");
    if (Green != std::string::npos)
        simplename.erase(Green, 5);
    size_t Blue = simplename.find("Blue");
    if (Blue != std::string::npos)
        simplename.erase(Blue, 4);

    // Remove everything after a number
    int index = -1;
    for (size_t i=simplename.size()-1; i>0; i--)
    {
        if (isdigit(simplename[i]) and not isdigit(simplename[i-1]))
        {
            index = i;
            break;
        }
    }
    if (index >= 0)
        simplename.erase(index);

    return simplename;
}

/*! @brief Returns true if member belongs to group
    @param member the single id
    @param group the group id
    @return true if member belongs to group
 */
bool NUData::belongsToGroup(const id_t& member, const id_t& group)
{
    return t_belongsToGroup<id_t>(member, group);
}

/*! @brief Returns true if member belongs to group
    @param member the name of a single id. The name is case sensitive.
    @param group the group id
    @return true if member belongs to group
 */
bool NUData::belongsToGroup(const std::string& name, const id_t& group)
{
    return t_belongsToGroup<std::string>(name, group);
}

/*! @brief A templated function to determine whether a member belongs to a particular group
    @param member the single sensornator
    @param group the group of sensornators you want to see if member belongs to
    @return true if member belongs to group, false otherwise
 */
template<typename T> bool NUData::t_belongsToGroup(const T& member, const id_t& group)
{
    if (group == All)
    {
        for (int i=NumCommonGroupIds.Id+1; i<NumCommonIds.Id; i++)
            if (*m_common_ids[i] == member)
                return true;
        return false;
    }
    else if (group == Head)
    {
        if (HeadRoll == member or HeadPitch == member or HeadYaw == member or NeckRoll == member or NeckPitch == member or NeckYaw == member)
            return true;
        else
            return false;
    }
    else if (group == Body)
    {
        if (belongsToGroup(member, All) and not belongsToGroup(member, Head))
            return true;
        else
            return false;
    }
    else if (group == LArm)
    {
        if (LShoulderRoll == member or LShoulderPitch == member or LShoulderYaw == member or LElbowRoll == member or LElbowPitch == member or LElbowYaw == member)
            return true;
        else
            return false;
    }
    else if (group == RArm)
    {
        if (RShoulderRoll == member or RShoulderPitch == member or RShoulderYaw == member or RElbowRoll == member or RElbowPitch == member or RElbowYaw == member)
            return true;
        else
            return false;
    }
    else if (group == Torso)
    {
        if (TorsoRoll == member or TorsoPitch == member or TorsoYaw == member)
            return true;
        else
            return false;
    }
    else if (group == LLeg)
    {
        if (LHipRoll == member or LHipPitch == member or LHipYaw == member or LHipYawPitch == member or LKneePitch == member or LAnkleRoll == member or LAnklePitch == member)
            return true;
        else
            return false;
    }
    else if (group == RLeg)
    {
        if (RHipRoll == member or RHipPitch == member or RHipYaw == member or RHipYawPitch == member or RKneePitch == member or RAnkleRoll == member or RAnklePitch == member)
            return true;
        else
            return false;
    }
    else
        return false;
}

/*! @brief Returns a list of indices into m_sensors/m_actionators so that
           the sensors/actionators under id can be accessed.
    @param id the id of the sensor/actionator(s) to get the indicies for
 */
const std::vector<int>& NUData::mapIdToIndices(const id_t& id) const
{
    return m_id_to_indices[id.Id];
}

std::vector<NUData::id_t*> NUData::mapIdToIds(const id_t& id)
{
    const std::vector<int>& indicies = mapIdToIndices(id);
    std::vector<id_t*> ids;
    ids.reserve(indicies.size());
    for (size_t i=0; i<indicies.size(); i++)
        ids.push_back(m_ids_copy[indicies[i]]);
    return ids;
}

void NUData::printMap(std::ostream& output)
{
    for (size_t j=0; j<m_id_to_indices.size(); j++)
    {
        output << m_ids_copy[j]->Name << "->[";
        for (size_t k=0; k<m_id_to_indices[j].size(); k++)
            output << m_ids_copy[m_id_to_indices[j][k]]->Name << " ";
        output << "]" << std::endl;
    }
}

NUData::id_t* NUData::getId(const std::string& name)
{
    for (size_t j=0; j<m_id_to_indices.size(); j++)
    {
        if(name == m_ids_copy[j]->Name)
        {
            return m_ids_copy[j];
        }
    }
    debug << "NUData::getId(): Name not found: " << name << std::endl;
    errorlog << "NUData::getId(): Name not found: " << name << std::endl;
    return NULL;
}
