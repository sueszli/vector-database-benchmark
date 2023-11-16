"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np
import matplotlib.path as mplPath
from geometry_tools import *

def build(XMLfile, host_model_file, Lon_bg, Lat_bg):
    if False:
        while True:
            i = 10
    Poly = []
    for (x1, y1) in zip(Lon_bg, Lat_bg):
        Poly.append((x1, y1))
    bbPath = mplPath.Path(Poly)
    read_host_file = open(host_model_file, 'r')
    lines_of_the_host_file = read_host_file.readlines()
    lines_of_the_host_file = [x.strip('L\n') for x in lines_of_the_host_file]
    lines_of_the_host_file = [x.strip('\r\n') for x in lines_of_the_host_file]
    lines_of_the_host_file = [x.strip('\n') for x in lines_of_the_host_file]
    line_number = 0
    source_read = False
    simple_fault = False
    complex_fault = False
    area_source = False
    point_source = False
    subduction_source = False
    for line in lines_of_the_host_file:
        if '<simpleFaultSource' in line:
            line_start = line_number
            index_id = line.find('id="') + 4
            source_read = False
            simple_fault = True
            complex_fault = False
            area_source = False
            point_source = False
            subduction_source = False
            if 'Subduction' in line or 'subduction' in line:
                subduction_source = True
        if '<areaSource' in line:
            Xing_bg = False
            type_increment = False
            type_mfd = False
            zone_defined = False
            line_start = line_number
            index_id = line.find('id="') + 4
            source_read = False
            simple_fault = False
            complex_fault = False
            area_source = True
            point_source = False
            subduction_source = False
            if 'Subduction' in line or 'subduction' in line:
                subduction_source = True
        if '<complexFaultSource' in line:
            line_start = line_number
            index_id = line.find('id="') + 4
            source_read = False
            simple_fault = False
            complex_fault = True
            area_source = False
            point_source = False
            subduction_source = False
            if 'Subduction' in line or 'subduction' in line:
                subduction_source = True
        if '<pointSource' in line:
            line_start = line_number
            index_id = line.find('id="') + 4
            source_read = False
            simple_fault = False
            complex_fault = False
            area_source = False
            point_source = True
            subduction_source = False
            if 'Subduction' in line or 'subduction' in line:
                subduction_source = True
        if '</sourceModel' in line:
            source_read = False
            simple_fault = False
            complex_fault = False
            area_source = False
            point_source = False
            subduction_source = False
        if simple_fault == True or complex_fault == True:
            print_source = True
            if '<gml:posList>' in line:
                line_start_lonlat = line_number
            if '</gml:posList>' in line:
                line_stop_lonlat = line_number
                lon_lat = ''
                for line_lon_lat in lines_of_the_host_file[line_start_lonlat:line_stop_lonlat + 1]:
                    line_lon_lat = line_lon_lat.replace('<gml:posList>', '')
                    line_lon_lat = line_lon_lat.replace('</gml:posList>', '')
                    lon_lat += ' ' + line_lon_lat
                lon_lat = lon_lat.replace('  ', ' ')
                lon_lat = lon_lat.replace('  ', ' ')
                lon_lat = lon_lat.split(' ')
                points = []
                for i in range(len(lon_lat) - 1):
                    if lon_lat[i] != '':
                        if len(points) != 0:
                            if float(lon_lat[i]) != points[-1][1]:
                                points.append([float(lon_lat[i]), float(lon_lat[i + 1])])
                        else:
                            points.append([float(lon_lat[i]), float(lon_lat[i + 1])])
                for point in points:
                    if bbPath.contains_point((point[0], point[1])) == True:
                        print_source = False
            if '<\\simpleFaultSource' in line or '</complexFaultSource' in line:
                line_end = line_number
                source_read = True
            if print_source == True or subduction_source == True:
                if source_read == True:
                    line_to_print = lines_of_the_host_file[line_start][:index_id] + '1111' + lines_of_the_host_file[line_start][index_id:]
                    XMLfile.write(line_to_print + '\n')
                    for line_to_print in lines_of_the_host_file[line_start + 1:line_end + 1]:
                        XMLfile.write(line_to_print + '\n')
        if point_source == True:
            print_source = True
            if '<gml:posList>' in line:
                line_start_lonlat = line_number
            if '</gml:posList>' in line:
                line_stop_lonlat = line_number
                lon_lat = ''
                for line_lon_lat in lines_of_the_host_file[line_start_lonlat:line_stop_lonlat + 1]:
                    line_lon_lat = line_lon_lat.replace('<gml:posList>', '')
                    line_lon_lat = line_lon_lat.replace('</gml:posList>', '')
                    lon_lat += ' ' + line_lon_lat
                lon_lat = lon_lat.replace('  ', ' ')
                lon_lat = lon_lat.replace('  ', ' ')
                lon_lat = lon_lat.split(' ')
                points = []
                for i in range(len(lon_lat) - 1):
                    if lon_lat[i] != '':
                        if len(points) != 0:
                            if float(lon_lat[i]) != points[-1][1]:
                                points.append([float(lon_lat[i]), float(lon_lat[i + 1])])
                        else:
                            points.append([float(lon_lat[i]), float(lon_lat[i + 1])])
                for point in points:
                    if bbPath.contains_point((point[0], point[1])) == True:
                        print_source = False
            if '<\\pointSource' in line:
                line_end = line_number
                source_read = True
            if print_source == True or subduction_source == True:
                if source_read == True:
                    line_to_print = lines_of_the_host_file[line_start][:index_id] + '1111' + lines_of_the_host_file[line_start][index_id:]
                    XMLfile.write(line_to_print + '\n')
                    for line_to_print in lines_of_the_host_file[line_start + 1:line_end + 1]:
                        XMLfile.write(line_to_print + '\n')
        if area_source == True:
            if '<gml:posList>' in line:
                line_start_lonlat = line_number
            if '</gml:posList>' in line:
                line_stop_lonlat = line_number
                lon_lat = ''
                for line_lon_lat in lines_of_the_host_file[line_start_lonlat:line_stop_lonlat + 1]:
                    line_lon_lat = line_lon_lat.replace('<gml:posList>', '')
                    line_lon_lat = line_lon_lat.replace('</gml:posList>', '')
                    lon_lat += ' ' + line_lon_lat
                lon_lat = lon_lat.replace('  ', ' ')
                lon_lat = lon_lat.replace('  ', ' ')
                lon_lat = lon_lat.split(' ')
                points_zone = []
                for i in range(len(lon_lat) - 1):
                    if lon_lat[i] != '':
                        if len(points_zone) != 0:
                            if float(lon_lat[i]) != points_zone[-1][1]:
                                points_zone.append([float(lon_lat[i]), float(lon_lat[i + 1])])
                        else:
                            points_zone.append([float(lon_lat[i]), float(lon_lat[i + 1])])
                ColX = []
                ColY = []
                for point in points_zone:
                    ColX.append(point[0])
                    ColY.append(point[1])
                    if bbPath.contains_point((point[0], point[1])) == True:
                        Xing_bg = True
                Poly = []
                for (x1, y1) in zip(ColX, ColY):
                    Poly.append((x1, y1))
                bbPath_zone = mplPath.Path(Poly)
                for (lon, lat) in zip(Lon_bg, Lat_bg):
                    if bbPath_zone.contains_point((lon, lat)) == True:
                        Xing_bg = True
            if '</areaSource>' in line:
                line_end = line_number
                source_read = True
            if Xing_bg == False or subduction_source == True:
                if source_read == True:
                    line_to_print = lines_of_the_host_file[line_start][:index_id] + '1111' + lines_of_the_host_file[line_start][index_id:]
                    XMLfile.write(line_to_print + '\n')
                    for line_to_print in lines_of_the_host_file[line_start + 1:line_end + 1]:
                        XMLfile.write(line_to_print + '\n')
            elif Xing_bg == True and subduction_source == False:
                if zone_defined == False:
                    zone_defined = True
                    listpoint_in_bg = []
                    ColX = []
                    ColY = []
                    for point in points_zone:
                        ColX.append(point[0])
                        ColY.append(point[1])
                        if bbPath.contains_point((point[0], point[1])) == True:
                            listpoint_in_bg.append(1)
                        else:
                            listpoint_in_bg.append(0)
                    bg_point_inzone = []
                    Poly = []
                    for (x1, y1) in zip(ColX, ColY):
                        Poly.append((x1, y1))
                    bbPath_zone = mplPath.Path(Poly)
                    for (lon, lat) in zip(Lon_bg, Lat_bg):
                        if bbPath_zone.contains_point((lon, lat)) == True:
                            bg_point_inzone.append(1)
                        else:
                            bg_point_inzone.append(0)
                    nb_Xing_zone = 0
                    for index_pt_z in range(len(listpoint_in_bg)):
                        if index_pt_z != 0:
                            if listpoint_in_bg[index_pt_z] != listpoint_in_bg[index_pt_z - 1]:
                                nb_Xing_zone += 1
                        elif listpoint_in_bg[-1] != listpoint_in_bg[index_pt_z]:
                            nb_Xing_zone += 1
                    nb_Xing_bg = 0
                    for index_pt_bg in range(len(bg_point_inzone)):
                        if index_pt_bg != 0:
                            if bg_point_inzone[index_pt_bg] != bg_point_inzone[index_pt_bg - 1]:
                                nb_Xing_bg += 1
                        elif bg_point_inzone[-1] != bg_point_inzone[index_pt_bg]:
                            nb_Xing_bg += 1
                    number_of_Xing = max([nb_Xing_zone, nb_Xing_bg])
                    if sum(bg_point_inzone) == len(bg_point_inzone) and sum(listpoint_in_bg) == 0:
                        lon_zone_modif = []
                        lat_zone_modif = []
                        for point in points_zone:
                            lon_zone = point[0]
                            lat_zone = point[1]
                            lon_zone_modif.append(lon_zone)
                            lat_zone_modif.append(lat_zone)
                        distances = []
                        for (lon_bg, lat_bg) in zip(Lon_bg, Lat_bg):
                            distances.append(distance(lon_bg, lat_bg, points_zone[-1][0], points_zone[-1][1]))
                        index_dist_min = np.argmin(distances)
                        lon_bg_modif = Lon_bg[index_dist_min:] + Lon_bg[:index_dist_min]
                        lat_bg_modif = Lat_bg[index_dist_min:] + Lat_bg[:index_dist_min]
                        if distance(lon_bg_modif[-1], lat_bg_modif[-1], points_zone[0][0], points_zone[0][1]) > distance(lon_bg_modif[0], lat_bg_modif[0], points_zone[0][0], points_zone[0][1]):
                            lon_bg_modif = list(reversed(lon_bg_modif))
                            lat_bg_modif = list(reversed(lat_bg_modif))
                        for (lon_bg, lat_bg) in zip(lon_bg_modif, lat_bg_modif):
                            lon_zone_modif.append(lon_bg)
                            lat_zone_modif.append(lat_bg)
                        line1 = [[points_zone[-1][0], points_zone[-1][1]], [lon_bg_modif[0], lat_bg_modif[0]]]
                        line2 = [[np.mean([lon_bg_modif[0], points_zone[-1][0]]) + 0.0001, np.mean([lat_bg_modif[0], points_zone[-1][1]])], [np.mean([lon_bg_modif[0], points_zone[-1][0]]), np.mean([lat_bg_modif[-1], points_zone[-1][1]])]]
                        (x, y) = line_intersection(line1, line2)
                        if x != 'no_intesection':
                            if points_aligned([np.mean([lon_bg_modif[0], points_zone[-1][0]]) + 0.0001, np.mean([lat_bg_modif[0], points_zone[-1][1]])], [np.mean([lon_bg_modif[0], points_zone[-1][0]]), np.mean([lat_bg_modif[-1], points_zone[-1][1]])], [x, y]) == False:
                                lon_zone_modif.append(lon_bg_modif[0] + 0.0001)
                                lat_zone_modif.append(lat_bg_modif[0])
                                lon_zone_modif.append(points_zone[-1][0] + 0.0001)
                                lat_zone_modif.append(points_zone[-1][1])
                            else:
                                lon_zone_modif.append(lon_bg_modif[0] - 0.0001)
                                lat_zone_modif.append(lat_bg_modif[0])
                                lon_zone_modif.append(points_zone[-1][0] - 0.0001)
                                lat_zone_modif.append(points_zone[-1][1])
                    else:
                        lon_zone_modif = []
                        lat_zone_modif = []
                        index_point_z = 0
                        for point in points_zone:
                            lon_zone = point[0]
                            lat_zone = point[1]
                            if listpoint_in_bg[index_point_z] == 0:
                                lon_zone_modif.append(lon_zone)
                                lat_zone_modif.append(lat_zone)
                            index_bg_intercept = None
                            if index_point_z != len(points_zone) - 1:
                                if listpoint_in_bg[index_point_z] == 0 and listpoint_in_bg[index_point_z + 1] == 1 or (listpoint_in_bg[index_point_z] == 1 and listpoint_in_bg[index_point_z + 1] == 0):
                                    index_point_bg = 0
                                    for (lon_bg, lat_bg) in zip(Lon_bg, Lat_bg):
                                        if index_point_bg != len(bg_point_inzone) - 1:
                                            line1 = [[lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]]]
                                            line2 = [[lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]]]
                                            (x, y) = line_intersection(line1, line2)
                                            if x != 'no_intesection':
                                                if points_aligned([lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]], [x, y]) == True:
                                                    lon_zone_modif.append(float(x))
                                                    lat_zone_modif.append(float(y))
                                                    if bg_point_inzone[index_point_bg] == 1:
                                                        index_bg_intercept = index_point_bg
                                                    else:
                                                        index_bg_intercept = index_point_bg + 1
                                        else:
                                            line1 = [[lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]]]
                                            line2 = [[lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]]]
                                            (x, y) = line_intersection(line1, line2)
                                            if x != 'no_intesection':
                                                if points_aligned([lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]], [x, y]) == True:
                                                    lon_zone_modif.append(float(x))
                                                    lat_zone_modif.append(float(y))
                                                    if bg_point_inzone[-1] == 1:
                                                        index_bg_intercept = -1
                                                    else:
                                                        index_bg_intercept = 0
                                        index_point_bg += 1
                                else:
                                    index_point_bg = 0
                                    for (lon_bg, lat_bg) in zip(Lon_bg, Lat_bg):
                                        if index_point_bg != len(bg_point_inzone) - 1:
                                            line1 = [[lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]]]
                                            line2 = [[lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]]]
                                            (x, y) = line_intersection(line1, line2)
                                            if x != 'no_intesection':
                                                if points_aligned([lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]], [x, y]) == True:
                                                    lon_zone_modif.append(float(x))
                                                    lat_zone_modif.append(float(y))
                                                    if bg_point_inzone[index_point_bg] == 1:
                                                        index_bg_intercept = index_point_bg
                                                    else:
                                                        index_bg_intercept = index_point_bg + 1
                                        else:
                                            line1 = [[lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]]]
                                            line2 = [[lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]]]
                                            (x, y) = line_intersection(line1, line2)
                                            if x != 'no_intesection':
                                                if points_aligned([lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[index_point_z + 1][0], points_zone[index_point_z + 1][1]], [x, y]) == True:
                                                    lon_zone_modif.append(float(x))
                                                    lat_zone_modif.append(float(y))
                                                    if bg_point_inzone[-1] == 1:
                                                        index_bg_intercept = -1
                                                    else:
                                                        index_bg_intercept = 0
                                        index_point_bg += 1
                            elif listpoint_in_bg[index_point_z] == 0 and listpoint_in_bg[0] == 1 or (listpoint_in_bg[index_point_z] == 1 and listpoint_in_bg[0] == 0):
                                index_point_bg = 0
                                for (lon_bg, lat_bg) in zip(Lon_bg, Lat_bg):
                                    if index_point_bg != len(bg_point_inzone) - 1:
                                        line1 = [[lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]]]
                                        line2 = [[lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]]]
                                        (x, y) = line_intersection(line1, line2)
                                        if x != 'no_intesection':
                                            if points_aligned([lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]], [x, y]) == True:
                                                lon_zone_modif.append(float(x))
                                                lat_zone_modif.append(float(y))
                                                if bg_point_inzone[index_point_bg] == 1:
                                                    index_bg_intercept = index_point_bg
                                                else:
                                                    index_bg_intercept = index_point_bg + 1
                                    else:
                                        line1 = [[lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]]]
                                        line2 = [[lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]]]
                                        (x, y) = line_intersection(line1, line2)
                                        if x != 'no_intesection':
                                            if points_aligned([lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]], [x, y]) == True:
                                                lon_zone_modif.append(float(x))
                                                lat_zone_modif.append(float(y))
                                                if bg_point_inzone[-1] == 1:
                                                    index_bg_intercept = -1
                                                else:
                                                    index_bg_intercept = 0
                                    index_point_bg += 1
                            else:
                                index_point_bg = 0
                                for (lon_bg, lat_bg) in zip(Lon_bg, Lat_bg):
                                    if index_point_bg != len(bg_point_inzone) - 1:
                                        line1 = [[lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]]]
                                        line2 = [[lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]]]
                                        (x, y) = line_intersection(line1, line2)
                                        if x != 'no_intesection':
                                            if points_aligned([lon_bg, lat_bg], [Lon_bg[index_point_bg + 1], Lat_bg[index_point_bg + 1]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]], [x, y]) == True:
                                                lon_zone_modif.append(float(x))
                                                lat_zone_modif.append(float(y))
                                                if bg_point_inzone[index_point_bg] == 1:
                                                    index_bg_intercept = index_point_bg
                                                else:
                                                    index_bg_intercept = index_point_bg + 1
                                    else:
                                        line1 = [[lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]]]
                                        line2 = [[lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]]]
                                        (x, y) = line_intersection(line1, line2)
                                        if x != 'no_intesection':
                                            if points_aligned([lon_bg, lat_bg], [Lon_bg[0], Lat_bg[0]], [x, y]) == True and points_aligned([lon_zone, lat_zone], [points_zone[0][0], points_zone[0][1]], [x, y]) == True:
                                                lon_zone_modif.append(float(x))
                                                lat_zone_modif.append(float(y))
                                                if bg_point_inzone[-1] == 1:
                                                    index_bg_intercept = -1
                                                else:
                                                    index_bg_intercept = 0
                                    index_point_bg += 1
                            if listpoint_in_bg[index_point_z] == 0 and index_bg_intercept != None:
                                lon_bg_modif = Lon_bg[index_bg_intercept:] + Lon_bg[:index_bg_intercept]
                                lat_bg_modif = Lat_bg[index_bg_intercept:] + Lat_bg[:index_bg_intercept]
                                bg_point_inzone_modif = bg_point_inzone[index_bg_intercept:] + bg_point_inzone[:index_bg_intercept]
                                if index_bg_intercept != 0 and bg_point_inzone_modif[-1] == 1:
                                    lon_bg_modif = list(reversed(lon_bg_modif))
                                    lat_bg_modif = list(reversed(lat_bg_modif))
                                    bg_point_inzone_modif = list(reversed(bg_point_inzone_modif))
                                i = 0
                                while bg_point_inzone_modif[i] == 1:
                                    lon_zone_modif.append(lon_bg_modif[i])
                                    lat_zone_modif.append(lat_bg_modif[i])
                                    i += 1
                            index_point_z += 1
                    (x, y) = reproject(ColY, ColX)
                    area_of_the_zone = area_of_polygon(x, y)
                    (x, y) = reproject(lat_zone_modif, lon_zone_modif)
                    area_of_the_zone_modified = area_of_polygon(x, y)
                    ratio_areas = area_of_the_zone_modified / area_of_the_zone
                    if number_of_Xing >= 3:
                        from itertools import groupby
                        indexes_for_test_init = [list(j) for (i, j) in groupby(listpoint_in_bg)]
                        indexes_for_test = []
                        index = 0
                        for i in indexes_for_test_init:
                            indexes_for_test_i = []
                            for ii in i:
                                if ii == 1:
                                    indexes_for_test_i.append(index)
                                index += 1
                            if ii == 1:
                                indexes_for_test.append(indexes_for_test_i)
                        indexes_for_test_modif = []
                        for indexes in list(indexes_for_test):
                            indexes_modif = []
                            if len(indexes) > 1:
                                for index in indexes:
                                    i = 0
                                    for (lon, lat) in zip(lon_zone_modif, lat_zone_modif):
                                        if lon == ColX[index] and lat == ColY[index]:
                                            indexes_modif.append(i)
                                        i += 1
                            else:
                                i = 0
                                for (lon, lat) in zip(lon_zone_modif, lat_zone_modif):
                                    if lon == ColX[indexes[0]] and lat == ColY[indexes[0]]:
                                        indexes_modif.append(i)
                                    i += 1
                            indexes_for_test_modif.append(indexes_modif)
                        for indexes in list(indexes_for_test_modif):
                            if len(indexes) > 1:
                                indexes = sorted(indexes)
                                lon_to_test = lon_zone_modif
                                lat_to_test = lat_zone_modif
                                i = 0
                                for index in indexes:
                                    del lon_to_test[index - i]
                                    del lat_to_test[index - i]
                                    i += 1
                                (x, y) = reproject(lat_to_test, lon_to_test)
                                area_of_the_zone_to_test = area_of_polygon(x, y)
                                ratio_areas_test = area_of_the_zone_to_test / area_of_the_zone
                            else:
                                lon_to_test = lon_zone_modif
                                lat_to_test = lat_zone_modif
                                if len(indexes) != 0:
                                    del lon_to_test[indexes[0]]
                                    del lat_to_test[indexes[0]]
                                (x, y) = reproject(lat_to_test, lon_to_test)
                                area_of_the_zone_to_test = area_of_polygon(x, y)
                                ratio_areas_test = area_of_the_zone_to_test / area_of_the_zone
                            if ratio_areas_test > 1.0:
                                print('included')
                            else:
                                print('excluded')
                if sum(bg_point_inzone) == len(bg_point_inzone) and sum(listpoint_in_bg) == 0:
                    (x, y) = reproject(lat_bg_modif, lon_bg_modif)
                    area_of_the_bg = area_of_polygon(x, y)
                    ratio_areas = (area_of_the_zone - area_of_the_bg) / area_of_the_zone
                if 'hterMFD' in line:
                    line_mfd_start_number = line_number
                    type_mfd = True
                if type_mfd == True and source_read == True:
                    index_a = line.find('aValue="') + 8
                    if line.find('aValue="') == -1:
                        index_a = line.find('aValue= "') + 9
                    if line.find('aValue="') == -1:
                        print('PROBLEM with reading the host file to find the a value')
                    a_str = line[index_a:]
                    i = a_str.find('"')
                    a = float(a_str[:i])
                    a_modif = a * ratio_areas
                    line_mfd_modif = line[:index_a] + str(a_modif) + line[line.find('"', index_a + 1):]
                if '<occurRates' in line:
                    line_mfd_start_number = line_number
                    type_increment = True
                if '/occurRates' in line:
                    line_mfd_stop_number = line_number
                if type_increment == True and source_read == True:
                    mfd_str = ''
                    for line_mfd_str in lines_of_the_host_file[line_mfd_start_number:line_mfd_stop_number + 1]:
                        mfd_str += ' ' + line_mfd_str
                    mfd_str = mfd_str.replace('<occurRates>', '')
                    mfd_str = mfd_str.replace('</occurRates>', '')
                    mfd_str = mfd_str.split(' ')
                    mfd_modif = []
                    for value in mfd_str:
                        if value != '':
                            mfd_modif.append(float(value) * ratio_areas)
                    line_mfd_modif = '<occurRates>'
                    for value in mfd_modif:
                        line_mfd_modif += str(value) + ' '
                    line_mfd_modif += '</occurRates>'
                if source_read == True:
                    line_to_print = lines_of_the_host_file[line_start][:index_id] + '1111' + lines_of_the_host_file[line_start][index_id:]
                    XMLfile.write(line_to_print + '\n')
                    for line_to_print in lines_of_the_host_file[line_start + 1:line_start_lonlat]:
                        XMLfile.write(line_to_print + '\n')
                    line_geom = '<gml:posList> '
                    for (lon, lat) in zip(lon_zone_modif, lat_zone_modif):
                        line_geom += str(lon) + ' ' + str(lat) + ' '
                    line_geom += '</gml:posList> '
                    XMLfile.write(line_geom + '\n')
                    if number_of_Xing >= 3:
                        print('POSSIBLE ERROR : please check if the host model is incorporate correctly, problems might have occured!!!')
                        print(lines_of_the_host_file[line_start][-9:-2], 'number_of_Xing=', number_of_Xing)
                        print('ratio_areas', ratio_areas)
                        import matplotlib.pyplot as plt
                        plt.scatter(ColX, ColY, c='b', alpha=0.2)
                        plt.scatter(Lon_bg, Lat_bg, c='r', alpha=0.2)
                        plt.scatter(lon_zone_modif, lat_zone_modif, c='k', alpha=0.2, marker='s')
                        plt.plot(lon_zone_modif, lat_zone_modif, ':k')
                        plt.xlim(min(Lon_bg) - 0.5, max(Lon_bg) + 0.5)
                        plt.ylim(min(Lat_bg) - 0.5, max(Lat_bg) + 0.5)
                        plt.show()
                    for line_to_print in lines_of_the_host_file[line_stop_lonlat + 1:line_mfd_start_number]:
                        XMLfile.write(line_to_print + '\n')
                    XMLfile.write(line_mfd_modif + '\n')
                    for line_to_print in lines_of_the_host_file[line_mfd_stop_number + 1:line_end + 1]:
                        XMLfile.write(line_to_print + '\n')
        line_number += 1