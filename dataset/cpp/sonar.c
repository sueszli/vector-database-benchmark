/**
 * @file
 * @brief nxt sonar driver
 *
 * @date 17.01.11
 * @author Alexander Batyukov
 */

#include <stdint.h>
#include <drivers/nxt/sonar_sensor.h>
#include <drivers/nxt/sensor.h>

void nxt_sonar_init (nxt_sensor_t *sensor) {
	sensor->def_comm = NXT_SONAR_DISTANCE_COMM;
	nxt_sensor_conf_active(sensor);
}
