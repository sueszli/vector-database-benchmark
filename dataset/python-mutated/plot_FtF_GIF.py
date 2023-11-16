"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.0 

@author: thomas
"""
import os, sys, inspect, copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import xml.etree.ElementTree as ET
import numpy as np
import math
import pylab as pl
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import Read_file as Read_file
import utils.Geometry_scenario as Geometry_scenario
import maps.maps_utils as maps_utils
import maps.geom as geom
from sources.background import bg
import geojson
from geojson import LineString, Feature, FeatureCollection, dump
import utils.read_input as read_input

def map_faults(Run_name, Model_list, scenarios_names_list, ScL_complet_list, BG_hyp_list, sample_list, b_value_list, MFD_type_list, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, File_bg, FileName_Prop, File_geom, plot_sr_use, visual_FtF, sub_area_file):
    if False:
        i = 10
        return i + 15
    nb_on_maps = False
    use_basemap = False
    available_sets = read_input.extract_sc_input('input/' + Run_name + '/ruptures.txt')
    for Model in Model_list:
        if not '.geojson' in FileName_Prop:
            Prop = np.genfromtxt(FileName_Prop, dtype=['U100', 'U100', 'f8', 'U100', 'U100', 'f8', 'f8', 'f8', 'f8', 'f8', 'U100', 'f8'], skip_header=1)
            Column_model_name = list(map(lambda i: Prop[i][0], range(len(Prop))))
            Column_fault_name = list(map(lambda i: Prop[i][1], range(len(Prop))))
            index_model = np.where(np.array(Column_model_name) == Model)[0]
            Prop = np.take(Prop, index_model)
            faults_names = np.array(Column_fault_name[index_model[0]:index_model[-1] + 1])
            faults_names = list(faults_names)
        else:
            with open(FileName_Prop) as f:
                gj = geojson.load(f)
            faults = gj['features']
            faults_names = []
            for fi in range(len(faults)):
                if faults[fi]['properties']['model'] == Model:
                    faults_names.append(str(faults[fi]['properties']['si']))
        geom_scenar = Geometry_scenario.Geom_scenar(faults_names, File_geom, Model)
        faults_lon = geom_scenar.faults_lon
        faults_lat = geom_scenar.faults_lat
        (Column_Fault_name, Depths) = geom.FaultGeometry(Model, File_geom)
        (Lon_bg, Lat_bg) = bg.geom(Model, File_bg)
        fault_geom = {}
        index_fault = 0
        for Fault_name in faults_names:
            lons = faults_lon[index_fault]
            lats = faults_lat[index_fault]
            i_d = np.where(np.array(Column_Fault_name) == Fault_name)
            depth = list(map(lambda i: Depths[i], i_d[0]))
            (dip, oriented, upper_sismo_depth, lower_sismo_depth) = geom.FaultProperties(FileName_Prop, Fault_name, Model)
            if depth[0] == 'sf':
                (trace_lon, trace_lat, plot_trace, polygon) = geom.get_sf_polygon(lons, lats, upper_sismo_depth, lower_sismo_depth, dip, oriented)
            else:
                (trace_lon, trace_lat) = ([], [])
                plot_trace = False
                (lon_top, lon_bottom, lat_top, lat_bottom) = ([], [], [], [])
                for (lon_i, lat_i, d_i) in zip(lons, lats, depth):
                    if d_i == min(depth):
                        lon_top.append(lon_i)
                        lat_top.append(lat_i)
                    if d_i == max(depth):
                        lon_bottom.append(lon_i)
                        lat_bottom.append(lat_i)
                poly_lons = np.concatenate([lon_top, np.array(list(reversed(lon_bottom)))])
                poly_lats = np.concatenate([lat_top, np.array(list(reversed(lat_bottom)))])
                polygon = [poly_lons, poly_lats]
            fault_geom.update({index_fault: {'name': Fault_name, 'trace_lon': trace_lon, 'trace_lat': trace_lat, 'plot_trace': plot_trace, 'polygon': polygon}})
            index_fault += 1
        '########################\n        Print the faults activated for each scenario\n        #########################'
        if visual_FtF == True and use_basemap == True:
            for scenario_set in scenarios_names_list:
                file_names = []
                if not os.path.exists(str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + scenario_set):
                    os.makedirs(str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + scenario_set)
                log_Mmax_file = str(Run_name) + '/' + str(Model) + '/Log/Mmax_sample_' + str(ScL_complet_list[0]) + '_' + scenario_set + '_1.txt'
                (sources_names, sources_Mmax, sources_Lengths, sources_Areas) = Read_file.read_file_Mmax_log(log_Mmax_file)
                rupture_set = available_sets['sc_' + scenario_set]
                m = Basemap(projection='mill', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='i')
                index_rupt = 0
                for rupt in rupture_set:
                    if len(Lon_bg) != 0:
                        maps_utils.draw_screen_poly(Lon_bg, Lat_bg, m, 'g', 0.2, 0.5, 'k')
                    m1 = copy.copy(m)
                    fault_colors = []
                    for fault in faults_names:
                        if fault in rupt:
                            fault_colors.append('r')
                        else:
                            fault_colors.append('k')
                    title = str()
                    i = 0
                    for fault in rupt:
                        title += ' ' + str(fault)
                        i += 1
                        if i == 6:
                            title += '\n'
                            i = 0
                    Mmax = sources_Mmax[len(faults_names) + index_rupt]
                    figpath = str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + scenario_set + '/' + str(index_rupt) + '.png'
                    maps_utils.make_fault_map(m1, fault_geom, fault_colors, figpath, title, Mmax=Mmax)
                    index_rupt += 1
        '########################\n        Print the map of the model\n        #########################'
        if use_basemap == True:
            m = Basemap(projection='mill', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='h')
            if len(Lon_bg) != 0:
                maps_utils.draw_screen_poly(Lon_bg, Lat_bg, m, 'g', 0.2, 0.5, 'k')
            fault_colors = ['k' for fault in faults_names]
            title = 'Map of the model : ' + Model
            figpath = str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + 'map.png'
            m.drawcoastlines(linewidth=0.1)
            if os.path.exists(sub_area_file):
                read_sub_area_file = open(sub_area_file, 'rU')
                lines_sub_area = read_sub_area_file.readlines()
                sub_area_names = []
                sub_area_coord = []
                for line in lines_sub_area:
                    model_sub_area = line.split('\t')[0]
                    if model_sub_area in Model_list:
                        sub_area_names.append(line.split('\t')[1])
                        sub_area_coord.append(line.split('\t')[2:])
                        sub_area_lon_i = []
                        sub_area_lat_i = []
                        for sub_area_coord_i in line.split('\t')[2:]:
                            if not '\n' in sub_area_coord_i.split(','):
                                if not '' in sub_area_coord_i.split(','):
                                    sub_area_lon_i.append(float(sub_area_coord_i.split(',')[1]))
                                    sub_area_lat_i.append(float(sub_area_coord_i.split(',')[0]))
                        maps_utils.draw_screen_poly(sub_area_lon_i, sub_area_lat_i, m, 'k', 0.01, 0.1, 'k')
                        (x, y) = m(sub_area_lon_i, sub_area_lat_i)
                        m.plot(x, y, linewidth=0.2, color='k', linestyle='dotted')
            maps_utils.make_fault_map(m, fault_geom, fault_colors, figpath, title, dpi=400, use_arcgis=True)
        '########################\n        Print the sliprate map\n        #########################'
        mean_param = np.genfromtxt(Run_name + '/analysis/txt_files/mean_parameters_faults.txt', dtype=['U100', 'U100', 'U1000', 'f8', 'f8'], delimiter='\t')
        model_mean_param = list(map(lambda i: mean_param[i][0], range(len(mean_param))))
        set_mean_param = list(map(lambda i: mean_param[i][1], range(len(mean_param))))
        fault_name_mean_param = list(map(lambda i: mean_param[i][2], range(len(mean_param))))
        sr_mean = list(map(lambda i: mean_param[i][3], range(len(mean_param))))
        Mmax_mean = list(map(lambda i: mean_param[i][4], range(len(mean_param))))
        index_model = np.where(np.array(model_mean_param) == Model)[0]
        set_mean_param_model = np.take(set_mean_param, index_model)
        fault_name_mean_param_model = np.take(fault_name_mean_param, index_model)
        sr_mean_model = np.take(sr_mean, index_model)
        Mmax_mean_model = np.take(Mmax_mean, index_model)
        if use_basemap == True:
            m = Basemap(projection='mill', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='i')
            title = 'Slip rate : ' + Model
            figpath = str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + 'map_sliprate.png'
        fault_colors = []
        for fault in faults_names:
            index_fault = np.where(np.array(fault_name_mean_param_model) == fault)[0][0]
            sr = sr_mean_model[index_fault]
            cmap = matplotlib.cm.get_cmap('rainbow')
            rgba = cmap(float(sr) / max(sr_mean))
            fault_colors.append(rgba)
        if use_basemap == True:
            maps_utils.make_fault_map(m, fault_geom, fault_colors, figpath, title, dpi=180, use_arcgis=False)
        '########################\n        Print the  Mmax map\n        #########################'
        if use_basemap == True:
            m = Basemap(projection='mill', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='i')
        for scenario_set in scenarios_names_list:
            if use_basemap == True:
                m1 = copy.copy(m)
                title = 'Slip rate : ' + Model + ' ' + scenario_set
                figpath = str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + 'map_Mmax_' + scenario_set + '.png'
            index_set = np.where(np.array(set_mean_param_model) == scenario_set)[0]
            fault_name_mean_param_set = np.take(fault_name_mean_param_model, index_set)
            sr_mean_set = np.take(sr_mean_model, index_set)
            Mmax_mean_set = np.take(Mmax_mean_model, index_set)
            fault_colors = []
            for fault in faults_names:
                index_fault = np.where(np.array(fault_name_mean_param_set) == fault)[0][0]
                Mmax = Mmax_mean_set[index_fault]
                cmap = matplotlib.cm.get_cmap('rainbow')
                rgba = cmap((float(Mmax) - min(Mmax_mean)) / (max(Mmax_mean) - min(Mmax_mean)))
                fault_colors.append(rgba)
            if use_basemap == True:
                maps_utils.make_fault_map(m1, fault_geom, fault_colors, figpath, title, dpi=180, use_arcgis=False)
        '########################\n        Print the NMS\n        #########################'
        if use_basemap == True:
            m = Basemap(projection='mill', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, resolution='i')
        for scenario_set in scenarios_names_list:
            index_set = np.where(np.array(set_mean_param_model) == scenario_set)[0]
            fault_name_mean_param_set = np.take(fault_name_mean_param_model, index_set)
            Mmax_mean_set = np.take(Mmax_mean_model, index_set)
            for MFD_type in MFD_type_list:
                if use_basemap == True:
                    title = 'NMS : ' + Model + ' ' + MFD_type + ' ' + scenario_set
                    figpath = str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + 'map_NMS_' + MFD_type + '_' + str(scenario_set) + '.png'
                slip_rep_data = np.genfromtxt(Run_name + '/analysis/txt_files/slip_rep_on_faults_mean_' + str(Model) + '_' + MFD_type + '_' + str(scenario_set) + '.txt', dtype=['U100', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'], delimiter='\t')
                fault_name_rep = list(map(lambda i: slip_rep_data[i][0], range(len(slip_rep_data))))
                p_NMS = list(map(lambda i: slip_rep_data[i][14], range(len(slip_rep_data))))
                if use_basemap == True:
                    m1 = copy.copy(m)
                if use_basemap == True:
                    fault_colors = []
                    for fault in faults_names:
                        index_fault = np.where(np.array(fault_name_rep) == fault)[0][0]
                        NMS_i = p_NMS[index_fault]
                        cmap = matplotlib.cm.get_cmap('rainbow')
                        if NMS_i >= 50.0:
                            NMS_i = 50.0
                        rgba = cmap(float(NMS_i * 2.0) / 100.0)
                        fault_colors.append(rgba)
                        maps_utils.make_fault_map(m, fault_geom, fault_colors, figpath, title, dpi=180, use_arcgis=False)
                '########################\n                Print the seismic sliprate\n                #########################'
                if use_basemap == True:
                    m2 = copy.copy(m)
                    title = 'Seismic slip rate : ' + Model + ' ' + MFD_type + ' ' + scenario_set
                    figpath = str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + 'map_seismicsliprate_' + MFD_type + '_' + str(scenario_set) + '.png'
                    fault_colors = []
                    for fault in faults_names:
                        index_fault = np.where(np.array(fault_name_rep) == fault)[0][0]
                        NMS_i = p_NMS[index_fault]
                        index_fault = np.where(np.array(fault_name_mean_param) == fault)[0][0]
                        sr = sr_mean[index_fault]
                        sr_seismic_i = sr * (1.0 - float(NMS_i) / 100.0)
                        cmap = matplotlib.cm.get_cmap('rainbow')
                        rgba = cmap(sr_seismic_i / max(sr_mean))
                        fault_colors.append(rgba)
                        maps_utils.make_fault_map(m2, fault_geom, fault_colors, figpath, title, dpi=180, use_arcgis=False)
                '########################\n                Build the geojson file\n                #########################'
                features = []
                id = 0
                for fault in faults_names:
                    index_fault = np.where(np.array(fault_name_rep) == fault)[0][0]
                    NMS_i = p_NMS[index_fault]
                    index_fault = np.where(np.array(fault_name_mean_param) == fault)[0][0]
                    sr = sr_mean[index_fault]
                    sr_seismic_i = sr * (1.0 - float(NMS_i) / 100.0)
                    index_fault = np.where(np.array(fault_name_mean_param_set) == fault)[0][0]
                    Mmax = Mmax_mean_set[index_fault]
                    trace = []
                    for (lon_i, lat_i) in zip(fault_geom[id]['trace_lon'], fault_geom[id]['trace_lat']):
                        trace.append([lon_i, lat_i])
                    trace = LineString(trace)
                    features.append(Feature(geometry=trace, properties={'id': id, 'name': fault, 'Mmax': Mmax, 'sliprate': sr, 'NMS': NMS_i, 'sr_seismic': sr_seismic_i}))
                    id += 1
                feature_collection = FeatureCollection(features)
                if not os.path.exists(str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + scenario_set):
                    os.makedirs(str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + scenario_set)
                with open(str(Run_name) + '/analysis/figures/FtF/' + Model + '/' + MFD_type + '_' + str(scenario_set) + '.geojson', 'w') as f:
                    dump(feature_collection, f)