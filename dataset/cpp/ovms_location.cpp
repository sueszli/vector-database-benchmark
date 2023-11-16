/*
;    Project:       Open Vehicle Monitor System
;    Date:          14th March 2017
;
;    Changes:
;    1.0  Initial release
;
;    (C) 2011       Michael Stegen / Stegen Electronics
;    (C) 2011-2017  Mark Webb-Johnson
;    (C) 2011        Sonny Chen @ EPRO/DX
;
; Permission is hereby granted, free of charge, to any person obtaining a copy
; of this software and associated documentation files (the "Software"), to deal
; in the Software without restriction, including without limitation the rights
; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
; copies of the Software, and to permit persons to whom the Software is
; furnished to do so, subject to the following conditions:
;
; The above copyright notice and this permission notice shall be included in
; all copies or substantial portions of the Software.
;
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
; THE SOFTWARE.
*/

#include "ovms_log.h"
static const char *TAG = "location";

#include "ovms_location.h"
#include "ovms_config.h"
#include "ovms_events.h"
#include "ovms_script.h"
#include "ovms_notify.h"
#include "ovms_command.h"
#include "vehicle.h"
#include "metrics_standard.h"
#include <math.h>

const char *LOCATIONS_PARAM = "locations";
#define LOCATION_DEFRADIUS 100

#define LOCATION_R 6371
#define LOCATION_TO_RAD (3.1415926536 / 180)

// Calculate haversine distance in meters
double OvmsLocationDistance(double th1, double ph1, double th2, double ph2)
  {
  double dx, dy, dz;

  ph1 -= ph2;
  ph1 *= LOCATION_TO_RAD, th1 *= LOCATION_TO_RAD, th2 *= LOCATION_TO_RAD;

  dz = sin(th1) - sin(th2);
  dx = cos(ph1) * cos(th1) - cos(th2);
  dy = sin(ph1) * cos(th1);
  return (asin(sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * LOCATION_R)*1000.0;
  }

OvmsLocationAction::OvmsLocationAction(bool enter, enum LocationAction action, const char* params, int len)
  : m_enter(enter), m_action(action), m_params(params, len) {}

const char* OvmsLocationAction::ActionString()
  {
  switch (m_action)
    {
    case HOMELINK: return "homelink";
    case ACC: return "acc";
    case NOTIFY: return "notify";
    default: return "INVALID";
    }
  }

void OvmsLocationAction::Execute(bool enter)
  {
  if (m_enter != enter)
    return;
  if (m_action == HOMELINK)
    {
    int homelink = m_params.at(0) - '0';
    int durationms = 1000;
    if (m_params.length() > 1)
      durationms = atoi(m_params.substr(2).c_str());
    OvmsVehicle* currentvehicle = MyVehicleFactory.m_currentvehicle;
    if (currentvehicle)
      {
      if (!StandardMetrics.ms_v_env_on->AsBool())
        ESP_LOGI(TAG, "Not activating Homelink #%d because parked",homelink);
      else
        switch(currentvehicle->CommandHomelink(homelink-1, durationms))
          {
          case OvmsVehicle::Success:
            ESP_LOGI(TAG, "Homelink #%d activated for %dms",homelink,durationms);
            break;
          case OvmsVehicle::Fail:
            ESP_LOGE(TAG, "Could not activate homelink #%d",homelink);
            break;
          default:
            ESP_LOGE(TAG, "Vehicle does not implement homelink");
            break;
          }
      }
    }
  else if (m_action == NOTIFY)
    {
    MyNotify.NotifyString("info", enter ? "location.enter" : "location.leave", m_params.c_str());
    }
  }

ActionList::~ActionList()
  {
  clear();
  }

void ActionList::clear()
  {
  for (iterator i = begin(); i != end(); ++i)
    delete *i;
  std::list<OvmsLocationAction*>::clear();
  }

ActionList::iterator ActionList::erase(iterator pos)
  {
  delete *pos;
  return std::list<OvmsLocationAction*>::erase(pos);
  }

OvmsLocation::OvmsLocation(const std::string& name)
  {
  m_name = name;
  m_inlocation = false;
  }

OvmsLocation::~OvmsLocation()
  {
  }

bool OvmsLocation::IsInLocation(float latitude, float longitude)
  {
  // This should check if we are in the location
  double dist = OvmsLocationDistance((double)latitude,(double)longitude,(double)m_latitude,(double)m_longitude);
  std::string event;

  // ESP_LOGI(TAG, "Location %s is %0.1fm distant",m_name.c_str(),dist);

  if (fabs(dist) <= m_radius)
    {
    // We are in the location
    if (!m_inlocation)
      {
      m_inlocation = true;
      StandardMetrics.ms_v_pos_location->SetValue(m_name);
      if (StandardMetrics.ms_v_env_on->AsBool())
        {
        event = std::string("location.enter.");
        event.append(m_name);
        MyEvents.SignalEvent(event.c_str(), (void*)m_name.c_str(), m_name.size()+1);
        }
      for (ActionList::iterator it = m_actions.begin(); it != m_actions.end(); ++it)
        (*it)->Execute(true);
      }
    }
  else
    {
    // We are out of the location
    if (m_inlocation)
      {
      m_inlocation = false;
      StandardMetrics.ms_v_pos_location->SetValue("");
      if (StandardMetrics.ms_v_env_on->AsBool())
        {
        event = std::string("location.leave.");
        event.append(m_name);
        MyEvents.SignalEvent(event.c_str(), (void*)m_name.c_str(), m_name.size()+1);
        }
      for (ActionList::iterator it = m_actions.begin(); it != m_actions.end(); ++it)
        (*it)->Execute(false);
      }
    }

  // Return current status
  return m_inlocation;
  }

bool OvmsLocation::Parse(const std::string& value)
  {
  const char *p = value.c_str();
  char next = 0;
  int len = 0;
  m_actions.clear();
  m_radius = LOCATION_DEFRADIUS;
  int num = sscanf(p, "%f , %f %n%1c", &m_latitude, &m_longitude, &len, &next);
  if (num < 2)
    return false;
  if (num < 3)
    return true;
  p += len;
  if (next == ',')
    {
    num = sscanf(p, ", %u %n%1c", &m_radius, &len, &next);
    if (num < 1)
      return false;
    if (num < 2)
      return true;
    p += len;
    }
  if (next != ';')
    return false;
  while (*p == ';')
    {
    bool enter;
    enum LocationAction action;
    p += strspn(++p, " ");
    if (strncasecmp(p, "enter:", 6) == 0)
      enter = true;
    else if (strncasecmp(p, "leave:", 6) == 0)
      enter = false;
    else
      return false;
    p += 6;
    const char* act = p;
    int alen = strcspn(act, ":");
    p += alen;
    if (*p++ != ':')
      return false;
    len = strcspn(p, ";");
    if (alen == 8 && strncasecmp(act, "homelink", alen) == 0)
      {
      if (len < 1 || *p < '1' || *p > '3')
        {
        ESP_LOGE(TAG, "homelink parameter must be 1, 2. or 3");
        return false;
        }
      if (len > 1)
        {
        int durationms;
        int slen;
        int n = sscanf(p+1, ",%d%n", &durationms, &slen);
        if (durationms < 100 || n != 1 || slen != len-1)
          {
          ESP_LOGE(TAG, "Minimum homelink timer duration 100ms");
          return false;
          }
        }
      action = HOMELINK;
      }
    else if (alen == 3 && strncasecmp(act, "acc", alen) == 0)
      {
      if (!enter)
        {
        ESP_LOGE(TAG, "Location action ACC not valid on leave");
        return false;
        }
      action = ACC;
      }
    else if (alen == 6 && strncasecmp(act, "notify", alen) == 0)
      action =  NOTIFY;
    else
      {
      ESP_LOGE(TAG, "Unrecognized action keyword");
      return false;
      }
    m_actions.push_back(new OvmsLocationAction(enter, action, p, len));
    p += len;
    }
  return true;
  }

void OvmsLocation::Store(std::string& buf)
  {
  char val[32];
  snprintf(val,sizeof(val),"%0.6f,%0.6f,%d", m_latitude, m_longitude, m_radius);
  buf = val;
  for (ActionList::iterator it = m_actions.begin(); it != m_actions.end(); ++it)
    {
    OvmsLocationAction* ola = *it;
    buf.append(1, ';');
    buf.append(ola->m_enter ? "enter" : "leave");
    buf.append(1, ':');
    buf.append(ola->ActionString());
    buf.append(1, ':');
    buf.append(ola->m_params);
    }
  MyConfig.SetParamValue(LOCATIONS_PARAM, m_name, buf.c_str());
  }

void OvmsLocation::Render(std::string& buf)
  {
  metric_unit_t user_length = OvmsMetricGetUserUnit(GrpDistanceShort, Meters);

  buf = string_format("%0.6f,%0.6f (%d%s)",
      m_latitude, m_longitude, UnitConvert(Meters, user_length, m_radius), OvmsMetricUnitLabel(user_length));
  bool first = true;
  for (ActionList::iterator it = m_actions.begin(); it != m_actions.end(); ++it)
    {
    OvmsLocationAction* ola = *it;
    if (!ola->m_enter)
      continue;
    if (first)
      {
      buf.append("\n  Enter Actions: ");
      first = false;
      }
    else
      buf.append("; ");
    buf.append(ola->ActionString());
    buf.append(1, ' ');
    if (ola->m_action == HOMELINK)
      {
      buf.append(1, ola->m_params.at(0));
      if (ola->m_params.length() > 1)
        {
        buf.append(" (");
        buf.append(ola->m_params.substr(2));
        buf.append("ms)");
        }
      }
    else
      buf.append(ola->m_params);
    }
  first = true;
  for (ActionList::iterator it = m_actions.begin(); it != m_actions.end(); ++it)
    {
    OvmsLocationAction* ola = *it;
    if (ola->m_enter)
      continue;
    if (first)
      {
      buf.append("\n  Leave Actions: ");
      first = false;
      }
    else
      buf.append("; ");
    buf.append(ola->ActionString());
    buf.append(1, ' ');
    buf.append(ola->m_params);
    }
  }

void location_list(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  OvmsConfigParam* p = MyConfig.CachedParam(LOCATIONS_PARAM);
  if (p == NULL) return;

  // List all the locations that have been parsed as valid
  for (LocationMap::iterator it=MyLocations.m_locations.begin(); it!=MyLocations.m_locations.end(); ++it)
    {
    const std::string& name = it->first;
    OvmsLocation* loc = it->second;
    std::string value;
    loc->Render(value);
    writer->printf("%s%s: %s\n", loc->m_inlocation ? "*" : "", name.c_str(), value.c_str());
    }

  // List any entries in the location config that are not valid
  for (ConfigParamMap::iterator it=p->m_map.begin(); it!=p->m_map.end(); ++it)
    {
    const std::string& name = it->first;
    const std::string& value = it->second;
    auto k = MyLocations.m_locations.find(name);
    if (k == MyLocations.m_locations.end())
      {
      writer->printf("Location %s is invalid: %s\n", name.c_str(), value.c_str());
      }
    }
  writer->puts("NOTE: ACC actions are not implemented yet!");       // XXX IMPLEMENT AND REMOVE THIS!
  }

int location_set_validate(OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv, bool complete)
  {
  if (argc == 5)
    {
    return OvmsMetricUnit_Validate(writer, argc, argv[4], complete, GrpDistanceShort);
    }
  return -1;
  }

void location_set(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  const char *name = argv[0];
  float latitude, longitude;
  int radius = LOCATION_DEFRADIUS;
  int base_value = radius;
  metric_unit_t user_length = Meters;

  if (strcmp(name, "?") == 0)
    {
    writer->printf("Error: ? is not a valid name\n");
    return;
    }
  if (argc >= 3)
    {
    latitude = atof(argv[1]);
    longitude = atof(argv[2]);
    }
  else
    {
    latitude = MyLocations.m_latitude;
    longitude = MyLocations.m_longitude;
    }

  if (argc > 3)
    {
    radius = atoi(argv[3]);
    base_value = radius;
    if (argc < 5)
      user_length = OvmsMetricGetUserUnit(GrpDistanceShort, Meters);
    else
      {
      user_length = OvmsMetricUnitFromName(argv[4]);
      if (user_length == UnitNotFound)
        {
        writer->printf("Error: Invalid Metric %s\n", argv[4]);
        return;
        }
      user_length = OvmsMetricCheckUnit(Meters, user_length);
      if (user_length == UnitNotFound)
        {
        writer->printf("Error: Metric %s is not a length unit\n", argv[4]);
        return;
        }
      }

    radius = UnitConvert(user_length, Meters, radius);
    }

  std::string val = string_format("%0.6f,%0.6f,%d",latitude,longitude,radius);
  MyConfig.SetParamValue(LOCATIONS_PARAM,name,val.c_str());
  if (user_length == Meters)
    writer->printf("Location defined with radius of %dm\n", base_value);
  else
    writer->printf("Location defined with radius of %d%s = %dm\n", base_value, OvmsMetricUnitLabel(user_length), radius);
  }

void location_radius(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  const char *name = argv[0];
  OvmsLocation* const* locp = MyLocations.m_locations.FindUniquePrefix(name);

  if (locp == NULL)
    {
    writer->printf("Error: No location %s defined\n",name);
    return;
    }
  metric_unit_t user_length;
  if (argc < 3)
    user_length = OvmsMetricGetUserUnit(GrpDistanceShort, Meters);
  else
    {
    user_length = OvmsMetricUnitFromName(argv[2]);
    if (user_length == UnitNotFound)
      {
      writer->printf("Error: Invalid Metric %s\n", argv[2]);
      return;
      }
    user_length = OvmsMetricCheckUnit(Meters, user_length);
    if (user_length == UnitNotFound)
      {
      writer->printf("Error: Metric %s is not a length unit\n", argv[2]);
      return;
      }
    }

  std::string buf;
  OvmsLocation* loc = *locp;
  int base_value = atoi(argv[1]);
  int radius_m = UnitConvert(user_length, Meters, base_value);
  loc->m_radius = radius_m;
  loc->Store(buf);
  if (user_length == Meters)
    writer->printf("Location radius set to %dm\n", base_value);
  else
    writer->printf("Location radius set to %d%s = %dm\n", base_value, OvmsMetricUnitLabel(user_length), radius_m);
  }

void location_rm(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  const char *name = argv[0];
  OvmsLocation* const* locp = MyLocations.m_locations.FindUniquePrefix(name);

  if (locp == NULL)
    {
    writer->printf("Error: No location %s defined\n",name);
    return;
    }

  MyConfig.DeleteInstance(LOCATIONS_PARAM,(*locp)->m_name);
  writer->puts("Location removed");
  }

void location_status(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  int n;
  writer->printf("Currently at %0.6f,%0.6f ",MyLocations.m_latitude,MyLocations.m_longitude);
  writer->printf("(%sGPS lock", MyLocations.m_gpslock ? "" : "without ");
  n = StandardMetrics.ms_v_pos_satcount->AsInt();
  if (n > 0)
    {
    writer->printf(", %d satellite%s, %s", n, n == 1 ? "" : "s",
      MyLocations.m_gpsgood ? "reliable" : "unreliable");
    }
  writer->puts(")");

  if ((MyLocations.m_park_latitude != 0) || (MyLocations.m_park_longitude != 0))
    writer->printf("Vehicle is parked%s %0.6f,%0.6f\n",
      MyLocations.m_park_invalid ? ", last known coordinates:" : " at",
      MyLocations.m_park_latitude,
      MyLocations.m_park_longitude);
  n = MyLocations.m_locations.size();
  writer->printf("There %s %d location%s defined\n",
    n == 1 ? "is" : "are", n, n == 1 ? "" : "s");

  bool found = false;
  for (LocationMap::iterator it=MyLocations.m_locations.begin(); it!=MyLocations.m_locations.end(); ++it)
    {
    if (it->second->m_inlocation)
      {
      if (!found) writer->printf("Active locations:");
      writer->printf(" %s",it->second->m_name.c_str());
      found = true;
      }
    }
  if (found)
    writer->puts("");
  else
    writer->puts("No active locations");
  }

int location_validate(OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv, bool complete)
  {
  if (argc == 1)
    return MyLocations.m_locations.Validate(writer, argc, argv[0], complete);
  return -1;
  }

int location_radius_validate(OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv, bool complete)
  {
  switch (argc)
    {
    case 1: return MyLocations.m_locations.Validate(writer, argc, argv[0], complete);
    case 3: return OvmsMetricUnit_Validate(writer, argc, argv[2], complete, GrpDistanceShort);
    }
  return -1;
  }

void location_action(int verbosity, OvmsWriter* writer, enum LocationAction act, std::string& params)
  {
  const char* const* rargv = writer->GetArgv();
  int remove = *rargv[2] == 'r' ? 1 : 0;
  bool enter = *rargv[2+remove] == 'e';
  const char* name = rargv[3+remove];
  OvmsLocation* const* locp = MyLocations.m_locations.FindUniquePrefix(name);
  if (locp == NULL)
    {
    writer->printf("Error: No location %s defined\n",name);
    return;
    }
  OvmsLocation* loc = *locp;
  if (!remove)
    {
    loc->m_actions.push_back(new OvmsLocationAction(enter, act, params.c_str(), params.length()));
    loc->Store(params);
    writer->puts("Location action set");
    if (act == ACC)
      writer->puts("NOTE: ACC actions are not implemented yet!");       // XXX IMPLEMENT AND REMOVE THIS!
    }
  else
    {
    int removed = 0;
    for (ActionList::iterator it = loc->m_actions.begin(); it != loc->m_actions.end(); )
      {
      OvmsLocationAction* ola = *it;
      if (ola->m_enter != enter)
        {
        ++it;
        continue;
        }
      if (act != INVALID)
        {
        if (ola->m_action != act)
          {
          ++it;
          continue;
          }
        if (!params.empty() && ola->m_params != params)
          {
          ++it;
          continue;
          }
        }
      ++removed;
      it = loc->m_actions.erase(it);
      if (it == loc->m_actions.end())
        break;
      }
    if (!removed)
      {
      writer->printf("Error: No matching action found in location %s\n",name);
      return;
      }
    loc->Store(params);
    writer->printf("%d action%s removed from location %s\n", removed, removed > 1 ? "s" : "", name);
    }
  }

void location_homelink(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  std::string params = cmd->GetName();
  if (argc == 1)
    {
    int durationms;
    int slen;
    int n = sscanf(argv[0], "%u%n", &durationms, &slen);
    if (durationms < 100 || n != 1 || slen != strlen(argv[0]))
      {
      writer->puts("Error: Minimum homelink timer duration 100ms");
      return;
      }
    params.append(1, ',');
    params.append(argv[0]);
    }
  enum LocationAction act = HOMELINK;
  location_action(verbosity, writer, act, params);
  }

void location_homelink_any(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  std::string params;
  enum LocationAction act = HOMELINK;
  location_action(verbosity, writer, act, params);
  }

void location_acc(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  std::string params = argc ? argv[0] : "";
  enum LocationAction act = ACC;
  location_action(verbosity, writer, act, params);
  }

void location_notify(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  std::string params;
  for (int i = 0; i < argc; ++i)
    {
    params.append(argv[i]);
    if (i < argc-1)
      params.append(1, ' ');
    }
  if (params.find(';') != string::npos)
    {
    writer->puts("Error: Notify text cannot include semicolon");
    return;
    }
  enum LocationAction act = NOTIFY;
  location_action(verbosity, writer, act, params);
  }

void location_all(int verbosity, OvmsWriter* writer, OvmsCommand* cmd, int argc, const char* const* argv)
  {
  std::string params;
  enum LocationAction act = INVALID;
  location_action(verbosity, writer, act, params);
  }

#ifdef CONFIG_OVMS_SC_JAVASCRIPT_DUKTAPE

static duk_ret_t DukOvmsLocationStatus(duk_context *ctx)
  {
  const char *mn = duk_to_string(ctx,0);
  OvmsLocation* const* locp = MyLocations.m_locations.FindUniquePrefix(mn);
  if (locp && *locp)
    {
    duk_push_boolean(ctx, (*locp)->m_inlocation);
    return 1;  /* one return value */
    }
  else
    return 0;
  }

#endif //#ifdef CONFIG_OVMS_SC_JAVASCRIPT_DUKTAPE

OvmsLocations MyLocations __attribute__ ((init_priority (1900)));

OvmsLocations::OvmsLocations()
  {
  ESP_LOGI(TAG, "Initialising LOCATIONS (1900)");

  m_ready = false;
  m_gpslock = false;
  m_gpssq = 0;
  m_gpsgood = false;
  m_latitude = 0;
  m_longitude = 0;
  m_park_latitude = 0;
  m_park_longitude = 0;
  m_park_distance = 0;
  m_park_invalid = true;
  m_last_alarm = 0;
  m_valet_latitude = 0;
  m_valet_longitude = 0;
  m_valet_distance = 0;
  m_valet_invalid = true;
  m_valet_last_alarm = 0;

  // Register our commands
  OvmsCommand* cmd_location = MyCommandApp.RegisterCommand("location","LOCATION framework", location_status, "", 0, 0, false);
  cmd_location->RegisterCommand("list","Show all locations",location_list);
  cmd_location->RegisterCommand("set","Set the position of a location",location_set, "<name> [<latitude> <longitude> [<radius> [<unit>]] ]", 1, 5, true, location_set_validate);
  cmd_location->RegisterCommand("radius","Set the radius of a location (defaults to user 'height' units)",location_radius, "<name> <radius> [<unit>]", 2, 3, true, location_radius_validate);
  cmd_location->RegisterCommand("rm","Remove a defined location",location_rm, "<name>", 1, 1, true, location_validate);
  cmd_location->RegisterCommand("status","Show location status",location_status);
  OvmsCommand* cmd_action = cmd_location->RegisterCommand("action","Set an action for a location");
  OvmsCommand* cmd_enter = cmd_action->RegisterCommand("enter","Set an action upon entering a location", NULL, "<location> $L", 1, 1, true, location_validate);
  OvmsCommand* cmd_leave = cmd_action->RegisterCommand("leave","Set an action upon leaving a location", NULL, "<location> $L", 1, 1, true, location_validate);
  OvmsCommand* cmd_rm_action = cmd_action->RegisterCommand("rm","Remove a location action");
  OvmsCommand* cmd_rm_enter = cmd_rm_action->RegisterCommand("enter","Remove an action from entering a location", location_all, "<location> [$C]", 1, 1, true, location_validate);
  OvmsCommand* cmd_rm_leave = cmd_rm_action->RegisterCommand("leave","Remove an action from leaving a location", location_all, "<location> [$C]", 1, 1, true, location_validate);

  OvmsCommand* enter_homelink = cmd_enter->RegisterCommand("homelink","Transmit Homelink signal",NULL,"$C [<duration=1000ms>]");
  enter_homelink->RegisterCommand("1","Transmit Homelink 1 signal",location_homelink,"[<duration=1000ms>]", 0, 1);
  enter_homelink->RegisterCommand("2","Transmit Homelink 2 signal",location_homelink,"[<duration=1000ms>]", 0, 1);
  enter_homelink->RegisterCommand("3","Transmit Homelink 3 signal",location_homelink,"[<duration=1000ms>]", 0, 1);
  cmd_enter->RegisterCommand("acc","ACC profile",location_acc,"<profile>", 1, 1);
  cmd_enter->RegisterCommand("notify","Text notification",location_notify,"<text>", 1, INT_MAX);
  OvmsCommand* leave_homelink = cmd_leave->RegisterCommand("homelink","Transmit Homelink signal",NULL,"$C [<duration=1000ms>]");
  leave_homelink->RegisterCommand("1","Transmit Homelink 1 signal",location_homelink,"[<duration=1000ms>]", 0, 1);
  leave_homelink->RegisterCommand("2","Transmit Homelink 2 signal",location_homelink,"[<duration=1000ms>]", 0, 1);
  leave_homelink->RegisterCommand("3","Transmit Homelink 3 signal",location_homelink,"[<duration=1000ms>]", 0, 1);
  cmd_leave->RegisterCommand("notify","Text notification",location_notify,"<text>", 1, INT_MAX);

  OvmsCommand* rm_enter_homelink = cmd_rm_enter->RegisterCommand("homelink","Remove Homelink signal",location_homelink_any);
  rm_enter_homelink->RegisterCommand("1","Remove Homelink 1 signal",location_homelink);
  rm_enter_homelink->RegisterCommand("2","Remove Homelink 2 signal",location_homelink);
  rm_enter_homelink->RegisterCommand("3","Remove Homelink 3 signal",location_homelink);
  cmd_rm_enter->RegisterCommand("acc","Remove ACC profile",location_acc,"[<profile>]", 0, 1);
  cmd_rm_enter->RegisterCommand("notify","Remove text notification",location_notify,"[<text>]", 0, INT_MAX);
  OvmsCommand* rm_leave_homelink = cmd_rm_leave->RegisterCommand("homelink","Remove Homelink signal",location_homelink_any);
  rm_leave_homelink->RegisterCommand("1","Remove Homelink 1 signal",location_homelink);
  rm_leave_homelink->RegisterCommand("2","Remove Homelink 2 signal",location_homelink);
  rm_leave_homelink->RegisterCommand("3","Remove Homelink 3 signal",location_homelink);
  cmd_rm_leave->RegisterCommand("notify","Remove text notification",location_notify,"[<text>]", 0, INT_MAX);

  // Register our parameters
  MyConfig.RegisterParam(LOCATIONS_PARAM, "Geo Locations", true, true);

  // Register our callbacks
  using std::placeholders::_1;
  using std::placeholders::_2;
  MyMetrics.RegisterListener(TAG, MS_V_POS_GPSLOCK, std::bind(&OvmsLocations::UpdatedGpsLock, this, _1));
  MyMetrics.RegisterListener(TAG, MS_V_POS_GPSSQ, std::bind(&OvmsLocations::UpdatedGpsSQ, this, _1));
  MyMetrics.RegisterListener(TAG, MS_V_POS_GPSTIME, std::bind(&OvmsLocations::UpdatedPosition, this, _1));
  MyMetrics.RegisterListener(TAG, MS_V_ENV_ON, std::bind(&OvmsLocations::UpdatedVehicleOn, this, _1));
  MyMetrics.RegisterListener(TAG, MS_V_ENV_VALET, std::bind(&OvmsLocations::UpdateValetMode, this, _1));

  MyEvents.RegisterEvent(TAG,"config.mounted", std::bind(&OvmsLocations::UpdatedConfig, this, _1, _2));
  MyEvents.RegisterEvent(TAG,"config.changed", std::bind(&OvmsLocations::UpdatedConfig, this, _1, _2));

#ifdef CONFIG_OVMS_SC_JAVASCRIPT_DUKTAPE
  ESP_LOGI(TAG, "Expanding DUKTAPE javascript engine");
  DuktapeObjectRegistration* dto = new DuktapeObjectRegistration("OvmsLocation");
  dto->RegisterDuktapeFunction(DukOvmsLocationStatus, 1, "Status");
  MyDuktape.RegisterDuktapeObject(dto);
#endif //#ifdef CONFIG_OVMS_SC_JAVASCRIPT_DUKTAPE
  }

OvmsLocations::~OvmsLocations()
  {
  MyMetrics.DeregisterListener(TAG);
  }

void OvmsLocations::UpdatedGpsLock(OvmsMetric* metric)
  {
  OvmsMetricBool* m = (OvmsMetricBool*)metric;
  m_gpslock = m->AsBool();
  if (m_gpslock)
    {
    MyEvents.SignalEvent("gps.lock.acquired", NULL);
    }
  else
    {
    MyEvents.SignalEvent("gps.lock.lost", NULL);
    }
  }

void OvmsLocations::UpdatedGpsSQ(OvmsMetric* metric)
  {
  m_gpssq = StdMetrics.ms_v_pos_gpssq->AsInt();
  int level_good = MyConfig.GetParamValueInt("vehicle", "gps.sq.good", 60);
  int level_bad  = MyConfig.GetParamValueInt("vehicle", "gps.sq.bad",  40);
  if (level_bad >= level_good) level_bad = level_good - 1;

  if (!m_gpsgood && m_gpssq >= level_good)
    {
    m_gpsgood = true;
    if (!m_ready)
      {
      UpdateParkPosition();
      UpdateValetPosition();
      m_ready = true;
      }
    UpdateLocations();
    MyEvents.SignalEvent("gps.sq.good", NULL);
    }
  else if (m_gpsgood && m_gpssq <= level_bad)
    {
    m_gpsgood = false;
    MyEvents.SignalEvent("gps.sq.bad", NULL);
    }
  }

void OvmsLocations::UpdatedPosition(OvmsMetric* metric)
  {
  m_latitude = StdMetrics.ms_v_pos_latitude->AsFloat();
  m_longitude = StdMetrics.ms_v_pos_longitude->AsFloat();
  if (m_gpsgood)
    {
    UpdateLocations();
    CheckTheft();
    CheckValet();
    }
  }

void OvmsLocations::UpdatedVehicleOn(OvmsMetric* metric)
  {
  UpdateParkPosition();
  UpdateValetPosition();
  }

void OvmsLocations::UpdateParkPosition()
  {
  OvmsRecMutexLock lock(&m_park_lock);

  bool caron = StdMetrics.ms_v_env_on->AsBool();
  if (caron)
    {
    m_park_latitude = 0;
    m_park_longitude = 0;
    m_park_distance = 0;
    m_park_invalid = true;
    m_last_alarm = 0;
    ESP_LOGI(TAG, "UpdateParkPosition: vehicle is driving");
    }
  else
    {
    m_park_latitude = m_latitude;
    m_park_longitude = m_longitude;
    m_park_invalid = (!m_gpsgood || StdMetrics.ms_v_pos_latitude->IsStale() || StdMetrics.ms_v_pos_longitude->IsStale());
    m_last_alarm = 0;
    ESP_LOGI(TAG, "UpdateParkPosition: vehicle is parking @%0.6f,%0.6f gpslock=%d satcount=%d hdop=%.1f sq=%d invalid=%d",
      m_park_latitude, m_park_longitude, m_gpslock,
      StdMetrics.ms_v_pos_satcount->AsInt(),
      StdMetrics.ms_v_pos_gpshdop->AsFloat(),
      m_gpssq, m_park_invalid);
    }
  }

void OvmsLocations::ReloadMap()
  {
  OvmsConfigParam* p = MyConfig.CachedParam(LOCATIONS_PARAM);
  if (p == NULL) return;

  // Forward search, updating existing locations
  for (ConfigParamMap::iterator it=p->m_map.begin(); it!=p->m_map.end(); ++it)
    {
    const std::string& name = it->first;
    const std::string& value = it->second;
    OvmsLocation* loc;
    auto k = m_locations.find(name);
    if (k == m_locations.end())
      {
      // Create it
      loc = new OvmsLocation(name);
      m_locations[name] = loc;
      }
    else
      loc = k->second;
    // Parse the parameters
    if (!loc->Parse(value))
      {
      ESP_LOGE(TAG, "Location %s is invalid: %s", name.c_str(), value.c_str());
      delete loc;
      m_locations.erase(name);
      }
    else
      {
      // ESP_LOGI(TAG, "Location %s is at %f,%f (%d)", name.c_str(), loc->m_latitude, loc->m_longitude, loc->m_radius);
      }
    }

  // Reverse search, go through existing locations looking for those to delete
  for (LocationMap::iterator it=m_locations.begin(); it!=m_locations.end();)
    {
    auto k = p->m_map.find(it->first);
    if (k == p->m_map.end())
      {
      // Location no longer exists
      // ESP_LOGI(TAG, "Location %s is removed",it->first.c_str());
      delete it->second;
      it = m_locations.erase(it);
      }
    else
      {
      it++;
      }
    }

  if (m_gpsgood) UpdateLocations();
  }

void OvmsLocations::UpdateLocations()
  {
  if ((m_latitude == 0) && (m_longitude == 0)) return;

  for (LocationMap::iterator it=m_locations.begin(); it!=m_locations.end(); ++it)
    {
    it->second->IsInLocation(m_latitude,m_longitude);
    }
  }

void OvmsLocations::CheckTheft()
  {
  static int last_dist = 0;

  OvmsRecMutexLock lock(&m_park_lock);

  if ((m_park_latitude == 0) && (m_park_longitude == 0)) return;
  if (StandardMetrics.ms_v_env_on->AsBool()) return;

  // Wait for first valid coordinates if we had none when we parked the car:
  if (m_park_invalid)
    {
    UpdateParkPosition();
    return;
    }

  int alarm = MyConfig.GetParamValueInt("vehicle", "flatbed.alarmdistance", 500);
  if (alarm == 0) return;

  double dist = fabs(OvmsLocationDistance((double)m_latitude,(double)m_longitude,(double)m_park_latitude,(double)m_park_longitude));
  // Park distance is the smoothed version
  m_park_distance = (m_park_distance * 4 + dist) / 5;
  if (last_dist != round(dist/10))
    {
    last_dist = round(dist/10);
    ESP_LOGV(TAG, "CheckTheft: vehicle parked @%0.6f,%0.6f now @%0.6f,%0.6f dist=%.0f smoothed=%.0f alarm=%d"
      " gpsmode=%s satcount=%d hdop=%.1f gpsspeed=%.1f",
      m_park_latitude, m_park_longitude, m_latitude, m_longitude, dist, m_park_distance, alarm,
      StdMetrics.ms_v_pos_gpsmode->AsString().c_str(),
      StdMetrics.ms_v_pos_satcount->AsInt(),
      StdMetrics.ms_v_pos_gpshdop->AsFloat(),
      StdMetrics.ms_v_pos_gpsspeed->AsFloat());
    }

  // Suppress false theft alerts due to a suspected SIMCOM GPS bug,
  // the reported location goes from: A,B -> A,B -> 0,B -> 0,A -> A,B -> A,B
  // Also seen: A,B -> A,B -> A,0 -> A,B -> A,B
  int simcombugdist = MyConfig.GetParamValueInt("vehicle", "flatbed.simcombugdistance", 500 * 1000);
  if (simcombugdist > 0 && (m_latitude == 0.0 || m_longitude == 0.0) &&
      m_park_distance > simcombugdist)
    {
    ESP_LOGE(TAG, "CheckTheft: Invalid SIMCOM GPS position @%0.6f,%0.6f dist=%.0f smoothed=%.0f",
      m_latitude, m_longitude, dist, m_park_distance);
    return;
    }

  int alarm_interval = MyConfig.GetParamValueInt("vehicle", "flatbed.alarminterval", 15) * 60;
  if ((m_park_distance > alarm) &&
      (m_last_alarm == 0 || (alarm_interval > 0 && monotonictime > m_last_alarm + alarm_interval)))
    {
    MyNotify.NotifyStringf("alert", "flatbed.moved",
      "Vehicle is being transported while parked - possible theft/flatbed (@%0.6f,%0.6f)",
      m_latitude, m_longitude);
    MyEvents.SignalEvent("location.alert.flatbed.moved", NULL);
    ESP_LOGW(TAG, "CheckTheft: flatbed.moved parked @%0.6f,%0.6f now @%0.6f,%0.6f gpsmode=%s satcount=%d hdop=%.1f gpsspeed=%.1f",
      m_park_latitude, m_park_longitude, m_latitude, m_longitude,
      StdMetrics.ms_v_pos_gpsmode->AsString().c_str(),
      StdMetrics.ms_v_pos_satcount->AsInt(),
      StdMetrics.ms_v_pos_gpshdop->AsFloat(),
      StdMetrics.ms_v_pos_gpsspeed->AsFloat());
    // inhibit further alerts for configured interval:
    m_last_alarm = monotonictime;
    }
  }

void OvmsLocations::UpdateValetPosition()
  {
  OvmsRecMutexLock lock(&m_valet_lock);
  if (!StandardMetrics.ms_v_env_valet->AsBool())
    {
    if (m_valet_enabled)
      ESP_LOGI(TAG, "UpdateValetPosition: Clear Valeting information");
    m_valet_enabled = false;
    m_valet_latitude = 0;
    m_valet_longitude = 0;
    m_valet_distance = 0;
    m_valet_invalid = true;
    m_valet_last_alarm = 0;
    StandardMetrics.ms_v_pos_valet_latitude->Clear();
    StandardMetrics.ms_v_pos_valet_longitude->Clear();
    StandardMetrics.ms_v_pos_valet_distance->Clear();
    }
  else if (!m_valet_enabled || m_valet_invalid)
    {

    m_valet_last_alarm = 0;
    m_valet_enabled = true;
    if (StandardMetrics.ms_v_pos_valet_latitude->IsDefined() && StandardMetrics.ms_v_pos_valet_longitude->IsDefined())
      {
      m_valet_latitude = StandardMetrics.ms_v_pos_valet_latitude->AsFloat();
      m_valet_longitude = StandardMetrics.ms_v_pos_valet_longitude->AsFloat();
      m_valet_distance = StandardMetrics.ms_v_pos_valet_distance->AsFloat();

      m_valet_invalid = false;
      ESP_LOGI(TAG, "UpdateValetPosition: Load from metrics - vehicle is valeting @%0.6f,%0.6f", m_valet_latitude, m_valet_longitude);
      }
    else
      {
      m_valet_latitude = m_latitude;
      m_valet_longitude = m_longitude;
      m_valet_invalid = (!m_gpsgood || StdMetrics.ms_v_pos_latitude->IsStale() || StdMetrics.ms_v_pos_longitude->IsStale());
      if (!m_valet_invalid)
        {
        StandardMetrics.ms_v_pos_valet_latitude->SetValue(m_valet_latitude);
        StandardMetrics.ms_v_pos_valet_longitude->SetValue(m_valet_longitude );
        }

      ESP_LOGI(TAG, "UpdateValetPosition: vehicle is valeting @%0.6f,%0.6f gpslock=%d satcount=%d hdop=%.1f sq=%d invalid=%d",
        m_valet_latitude, m_valet_longitude, m_gpslock,
        StdMetrics.ms_v_pos_satcount->AsInt(),
        StdMetrics.ms_v_pos_gpshdop->AsFloat(),
        m_gpssq, m_valet_invalid);
      }
    }
  }

void OvmsLocations::CheckValet()
  {
  OvmsRecMutexLock lock(&m_valet_lock);
  if (!m_valet_enabled)
    return;
  if (!StandardMetrics.ms_v_env_valet->AsBool())
    {
    UpdateValetPosition();
    return;
    }

  // Wait for first valid coordinates if we had none when we valeted the car:
  if (m_valet_invalid)
    {
    UpdateValetPosition();
    return;
    }

  if ((m_valet_latitude == 0) && (m_valet_longitude == 0))
    return;

  int alarm_dist = MyConfig.GetParamValueInt("vehicle", "valet.alarmdistance", 0);
  if (alarm_dist == 0) return;

  double dist = fabs(OvmsLocationDistance(
    (double)m_latitude,(double)m_longitude,
    (double)m_valet_latitude,(double)m_valet_longitude));
  // Valet distance is the smoothed version
  m_valet_distance = (m_valet_distance * 4 + dist) / 5;
  StandardMetrics.ms_v_pos_valet_distance->SetValue(m_valet_distance);

  int alarm_interval = MyConfig.GetParamValueInt("vehicle", "valet.alarminterval", 15) * 60;
  if ((m_valet_distance > alarm_dist) &&
      (m_valet_last_alarm == 0
        || (alarm_interval > 0 && monotonictime > m_valet_last_alarm + alarm_interval)))
    {
    MyNotify.NotifyStringf("alert", "valet.bounds",
      "Vehicle has moved out of area while being valeted (@%0.6f,%0.6f)",
      m_latitude, m_longitude);
    MyEvents.SignalEvent("location.alert.valet.bounds", NULL);
    ESP_LOGW(TAG, "CheckValet: valet.bounds valeted @%0.6f,%0.6f now @%0.6f,%0.6f gpsmode=%s satcount=%d hdop=%.1f gpsspeed=%.1f",
      m_valet_latitude, m_valet_longitude, m_latitude, m_longitude,
      StdMetrics.ms_v_pos_gpsmode->AsString().c_str(),
      StdMetrics.ms_v_pos_satcount->AsInt(),
      StdMetrics.ms_v_pos_gpshdop->AsFloat(),
      StdMetrics.ms_v_pos_gpsspeed->AsFloat());
    // inhibit further alerts for configured interval:
    m_valet_last_alarm = monotonictime;
    }
  }
void OvmsLocations::UpdateValetMode(OvmsMetric* metric)
  {
  UpdateValetPosition();
  }

void OvmsLocations::UpdatedConfig(std::string event, void* data)
  {
  if (event.compare("config.changed")==0)
    {
    // Only reload if our parameter has changed
    OvmsConfigParam*p = (OvmsConfigParam*)data;
    if (p->GetName().compare(LOCATIONS_PARAM)!=0) return;
    }

  ReloadMap();

  if (event == "config.mounted")
    {
    // Init from persistent position & vehicle state:
    m_latitude = StdMetrics.ms_v_pos_latitude->AsFloat();
    m_longitude = StdMetrics.ms_v_pos_longitude->AsFloat();
    UpdateLocations();
    UpdatedVehicleOn(StdMetrics.ms_v_env_on);
    }
  }
