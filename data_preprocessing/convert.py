from netCDF4 import Dataset
from osgeo import gdal
import os
from osgeo import osr
import pandas as pd
import numpy as np
import datetime
import csv
import glob
import sys
import os
import xarray as xr
import pygrib
import cfgrib

# grib_data = cfgrib.open_datasets(
#     '/Users/dongochuyen/Downloads/data_processing/102020.grib')
# data = grib_data[0]
# print(data)

year = '2019'
list_mon = ['01']
for mon in list_mon:

    grib_data = cfgrib.open_datasets(
        "/Users/dongochuyen/Downloads/ERA5/" + mon + year + ".grib")
    data = grib_data[0].isor
    # print(len(data.values))

    arq = gdal.Open(
        "/Users/dongochuyen/Downloads/ERA5/" + mon + year + ".grib")
    GT_geo = arq.GetGeoTransform()
    # print(GT_geo)
    gribfile = pygrib.open(
        "/Users/dongochuyen/Downloads/ERA5/" + mon + year + ".grib")
    grbs = gribfile

    # total cloud cover
    for i in range(len(data.values)):
        # print(i)
        selected_grb = grbs.select(name='Total cloud cover')[i]

        date = selected_grb.dataDate
        time = selected_grb.dataTime

        print("Date: ", date)

        if time == 0:
            time = str(date)+'_'+str(time)+'000'
        elif time > 0 and time < 1000:
            time = str(date)+'_'+'0'+str(time)
        else:
            time = str(date)+'_'+str(time)
        print(time)

        tif_out = '/Users/dongochuyen/Downloads/ERA5_test/TCC/' + \
            year+'/'+mon+'/'+'TCC_' + time+'.tif'
        data_actual, lats, lons = selected_grb.data()

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            tif_out, data_actual.shape[1], data_actual.shape[0], 1, gdal.GDT_Float32)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform(GT_geo)
        # dst_ds.SetGeoTransform((originX, 0.05, 0, originY, 0, -0.05))
        dst_ds. GetRasterBand(1).WriteArray(data_actual)
        dst_ds. GetRasterBand(1).SetNoDataValue(-99999)
        dst_ds.FlushCache()
        dst_ds = None

    # total column water
    for i in range(len(data.values)):
        # print(i)
        selected_grb = grbs.select(name='Total column water')[i]

        date = selected_grb.dataDate
        time = selected_grb.dataTime

        print("Date: ", date)

        if time == 0:
            time = str(date)+'_'+str(time)+'000'
        elif time > 0 and time < 1000:
            time = str(date)+'_'+'0'+str(time)
        else:
            time = str(date)+'_'+str(time)
        print(time)

        tif_out = '/Users/dongochuyen/Downloads/data_processing/ERA5_output/TCW/' + \
            year+'/'+mon+'/'+'TCW_' + time+'.tif'
        data_actual, lats, lons = selected_grb.data()

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            tif_out, data_actual.shape[1], data_actual.shape[0], 1, gdal.GDT_Float32)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform(GT_geo)
        # dst_ds.SetGeoTransform((originX, 0.05, 0, originY, 0, -0.05))
        dst_ds. GetRasterBand(1).WriteArray(data_actual)
        dst_ds. GetRasterBand(1).SetNoDataValue(-99999)
        dst_ds.FlushCache()
        dst_ds = None

    # Total column vertically-integrated water vapour (TCWV)

    for i in range(len(data.values)):
        # print(i)
        selected_grb = grbs.select(name='Total column water vapour')[i]

        date = selected_grb.dataDate
        time = selected_grb.dataTime

        print("Date: ", date)

        if time == 0:
            time = str(date)+'_'+str(time)+'000'
        elif time > 0 and time < 1000:
            time = str(date)+'_'+'0'+str(time)
        else:
            time = str(date)+'_'+str(time)
        print(time)

        tif_out = '/Users/dongochuyen/Downloads/data_processing/ERA5_output/TCWV/' + \
            year+'/'+mon+'/'+'TCWV_' + time+'.tif'
        data_actual, lats, lons = selected_grb.data()

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            tif_out, data_actual.shape[1], data_actual.shape[0], 1, gdal.GDT_Float32)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform(GT_geo)
        # dst_ds.SetGeoTransform((originX, 0.05, 0, originY, 0, -0.05))
        dst_ds. GetRasterBand(1).WriteArray(data_actual)
        dst_ds. GetRasterBand(1).SetNoDataValue(-99999)
        dst_ds.FlushCache()
        dst_ds = None

    # convective available potential energy (CAPE)

    for i in range(len(data.values)):
        # print(i)
        selected_grb = grbs.select(
            name='Convective available potential energy')[i]

        date = selected_grb.dataDate
        time = selected_grb.dataTime

        print("Date: ", date)

        if time == 0:
            time = str(date)+'_'+str(time)+'000'
        elif time > 0 and time < 1000:
            time = str(date)+'_'+'0'+str(time)
        else:
            time = str(date)+'_'+str(time)
        print(time)

        tif_out = '/Users/dongochuyen/Downloads/data_processing/ERA5_output/CAPE/' + \
            year+'/'+mon+'/'+'CAPE_' + time+'.tif'
        data_actual, lats, lons = selected_grb.data()

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            tif_out, data_actual.shape[1], data_actual.shape[0], 1, gdal.GDT_Float32)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform(GT_geo)
        # dst_ds.SetGeoTransform((originX, 0.05, 0, originY, 0, -0.05))
        dst_ds. GetRasterBand(1).WriteArray(data_actual)
        dst_ds. GetRasterBand(1).SetNoDataValue(-99999)
        dst_ds.FlushCache()
        dst_ds = None

    # anisotropy of sub-gridscale orography (ISOR)
    for i in range(len(data.values)):

        selected_grb = grbs.select(
            name='Anisotropy of sub-gridscale orography')[i]

        date = selected_grb.dataDate
        time = selected_grb.dataTime

        print("Date: ", date)

        if time == 0:
            time = str(date)+'_'+str(time)+'000'
        elif time > 0 and time < 1000:
            time = str(date)+'_'+'0'+str(time)
        else:
            time = str(date)+'_'+str(time)
        print(time)

        tif_out = '/Users/dongochuyen/Downloads/data_processing/ERA5_output/ISOR/' + \
            year+'/'+mon+'/'+'ISOR_' + time+'.tif'
        data_actual, lats, lons = selected_grb.data()

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            tif_out, data_actual.shape[1], data_actual.shape[0], 1, gdal.GDT_Float32)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        dst_ds.SetProjection(outRasterSRS.ExportToWkt())
        dst_ds.SetGeoTransform(GT_geo)
        # dst_ds.SetGeoTransform((originX, 0.05, 0, originY, 0, -0.05))
        dst_ds. GetRasterBand(1).WriteArray(data_actual)
        dst_ds. GetRasterBand(1).SetNoDataValue(-99999)
        dst_ds.FlushCache()
        dst_ds = None
