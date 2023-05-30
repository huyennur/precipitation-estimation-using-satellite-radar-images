import os
from osgeo import gdal
import glob


def isInputHimaOrRadar(inn, year, month, day, hour):
    input_file = str(inn)
    output_file = ''

    if("B09B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_B09B_' + str(hour)
    elif("B10B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_B10B_' + str(hour)
    elif("B11B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_B11B_' + str(hour)
    elif("B12B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_B12B_' + str(hour)
    elif("B14B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_B14B_' + str(hour)
    elif("B16B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_B16B_' + str(hour)
    elif("I2B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_I2B_' + str(hour)
    elif("I4B" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_I4B_' + str(hour)
    elif("IRB" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_IRB_' + str(hour)
    elif("WVB" in input_file):
        output_file = 'hima_' + \
            str(year) + str(month) + str(day) + '_WVB_' + str(hour)

    return output_file


def cropping_image1(inn, input, out, output):

    in_path = str(inn)
    input_filename = str(input)
    out_path = str(out)
    output_filename = str(output)

    tile_size_x = 64
    tile_size_y = 64

    ds = gdal.Open(in_path + input_filename)
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    # print(xsize)
    # print(ysize)
    # print('\n')

    for i in range(0, xsize, tile_size_x):
        if (i == 896):
            print("i = 896")
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(888) + " " + str(j) + " " + str(tile_size_x) + " " + str(
                    tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + f'{888:04d}' + "_" + f'{j:04d}' + ".tif"
                if (j == 1728):
                    print('j = 1728')
                else:
                    os.system(com_string)
                    print(com_string)
        else:
            for j in range(0, ysize, tile_size_y):
                com_string = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(i) + " " + str(j) + " " + str(tile_size_x) + " " + str(
                    tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + f'{i:04d}' + "_" + f'{j:04d}' + ".tif"
                if (j == 1728):
                    com_string = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(i) + " " + str(1688) + " " + str(tile_size_x) + " " + str(
                        tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + f'{i:04d}' + "_" + f'{1688:04d}' + ".tif"
                    if(i == 896):
                        print("i = 896")
                    else:
                        os.system(com_string)
                        print(com_string)
                else:
                    os.system(com_string)
                    print(com_string)

    com_string_last = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(888) + " " + str(1688) + " " + str(tile_size_x) + " " + str(
        tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + f'{888:04d}' + "_" + f'{1688:04d}' + ".tif"
    os.system(com_string_last)
    print(com_string_last)

# cắt ảnh theo tham số truyền vào là ratio


def cropping_image(inn, input, out, output, ratioo):

    in_path = str(inn)
    input_filename = str(input)
    out_path = str(out)
    output_filename = str(output)
    ratio = str(ratioo)
    first, second = ratio.split('_')
    print(first, second)

    tile_size_x = 512
    tile_size_y = 512

    # ds = gdal.Open(in_path + "/" + input_filename)
    # band = ds.GetRasterBand(1)
    # xsize = band.XSize
    # ysize = band.YSize

    # com_string_last = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(first) + " " + str(second) + " " + str(tile_size_x) + " " + str(
    #     tile_size_y) + " " + str(in_path) + "/" + str(input_filename) + " " + str(out_path) + "/" + str(output_filename) + str(first) + "_" + str(second) + ".tif"
    com_string_last = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(first) + " " + str(second) + " " + str(tile_size_x) + " " + str(
        tile_size_y) + " " + str(in_path) + "/" + str(input_filename) + " " + str(out_path) + "/" + str(output_filename) + ".tif"

    os.system(com_string_last)
    print(com_string_last)


def createHourFolder(year, month, day, hour):
    os.mkdir("/Users/dongochuyen/Desktop/precipitation/dataset2/himawari/" + str(year) +
             "/" + str(month) + "/" + str(day) + "/" + str(hour))
    hour_ratio = "/Users/dongochuyen/Desktop/precipitation/dataset2/himawari/" + \
        str(year) + "/" + str(month) + "/" + str(day) + "/" + str(hour)
    print("Create folder: ", hour_ratio)


# def createRatioFolder(year, month, day, hour, ratio):
#     # os.mkdir("/Users/dongochuyen/Desktop/data_processing/data_hima/crop/" + str(year) +
#     #          "/" + str(month) + "/" + str(day) + "/" + str(hour) + "/" + str(ratio))
#     path_ratio = "/Users/dongochuyen/Desktop/data_processing/data_hima/crop/" + \
#         str(year) + "/" + str(month) + "/" + \
#         str(day) + "/" + str(hour) + "/" + str(ratio)
#     return path_ratio


# in_path = '/Users/dongochuyen/Downloads/'
# input_filename = 'output_radar_20200802_00.tif'

# out_path = '/Users/dongochuyen/Downloads/'
# output_filename = 'radar_20200802_00_'

# cropping_image1(in_path, input_filename, out_path, output_filename)

path_folder = "/Users/dongochuyen/Downloads/IRB"
path_folder_out = "/Users/dongochuyen/Desktop/precipitation/dataset2/himawari"


def croppingHima():

    for year in sorted(os.listdir(path_folder)):
        year_path = os.path.join(path_folder, year)
        path_year_out = os.path.join(path_folder_out, year)
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            month_path_out = os.path.join(path_year_out, month)
            for day in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day)
                day_path_out = os.path.join(month_path_out, day)
                for filename in sorted(os.listdir(day_path)):
                    filename_path = os.path.join(day_path, filename)
                    hour = filename[14:16]

                    hour_path_out = os.path.join(day_path_out, hour)
                    if(os.path.isdir(hour_path_out) == False):
                        createHourFolder(year, month, day, hour)
                    output_filename = isInputHimaOrRadar(
                        filename_path, year, month, day, hour)

                    print("In path: ", day_path)
                    print("Input filename: ", filename)  # input_filename
                    print("Out path: ", hour_path_out)
                    print("Output filename: ", output_filename)
                    cropping_image(day_path, filename,
                                   hour_path_out, output_filename, "0256_0320")
                    print("\n")


croppingHima()


# -----------------------
#  cut new ratio image

# def isInputHimaOrRadar(inn, year, month, day, hour):
#     input_file = str(inn)
#     output_file = ''

#     if("B09B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_B09B_' + str(hour) + '_'
#     elif("B10B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_B10B_' + str(hour) + '_'
#     elif("B11B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_B11B_' + str(hour) + '_'
#     elif("B12B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_B12B_' + str(hour) + '_'
#     elif("B14B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_B14B_' + str(hour) + '_'
#     elif("B16B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_B16B_' + str(hour) + '_'
#     elif("I2B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_I2B_' + str(hour) + '_'
#     elif("I4B" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_I4B_' + str(hour) + '_'
#     elif("IRB" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_IRB_' + str(hour) + '_'
#     elif("WVB" in input_file):
#         output_file = 'hima_' + \
#             str(year) + str(month) + str(day) + '_WVB_' + str(hour) + '_'
#     elif("radar" in input_file):
#         output_file = 'radar_' + \
#             str(year) + str(month) + str(day) + '_' + str(hour)

#     return output_file


# def cropping_image(inn, input, out, output, ratioo):

#     in_path = str(inn)
#     input_filename = str(input)
#     out_path = str(out)
#     output_filename = str(output)
#     ratio = str(ratioo)
#     first, second = ratio.split('_')
#     print(first, second)

#     tile_size_x = 512
#     tile_size_y = 512

#     # ds = gdal.Open(in_path + "/" + input_filename)
#     # band = ds.GetRasterBand(1)
#     # xsize = band.XSize
#     # ysize = band.YSize

#     com_string_last = "gdal_translate -of GTIFF -co 'COMPRESS=LZW' -srcwin " + str(first) + " " + str(second) + " " + str(tile_size_x) + " " + str(
#         tile_size_y) + " " + str(in_path) + "/" + str(input_filename) + " " + str(out_path) + "/" + str(output_filename) + ".tif"
#     os.system(com_string_last)
#     print(com_string_last)


# def createHourFolder(year, month, day, hour):
#     os.mkdir("/Users/dongochuyen/Desktop/precipitation/dataset2/radar/" +
#              str(year) + "/" + str(month) + "/" + str(day) + "/" + str(hour))
#     hour_ratio = "/Users/dongochuyen/Desktop/precipitation/dataset2/radar/" + \
#         str(year) + "/" + str(month) + "/" + str(day) + "/" + str(hour)
#     print("Create folder: ", hour_ratio)


# path_folder_standardised_data = "/Users/dongochuyen/Downloads/standardised_data/radar"
# path_folder_out = "/Users/dongochuyen/Desktop/precipitation/dataset2/radar"


# def croppingradar():
#     for year in sorted(os.listdir(path_folder_standardised_data)):
#         year_path = os.path.join(path_folder_standardised_data, year)
#         year_path_out = os.path.join(path_folder_out, year)
#         for month in sorted(os.listdir(year_path)):
#             month_path = os.path.join(year_path, month)
#             month_path_out = os.path.join(year_path_out, month)
#             for day in sorted(os.listdir(month_path)):
#                 day_path = os.path.join(month_path, day)
#                 day_path_out = os.path.join(month_path_out, day)
#                 for hour in sorted(os.listdir(day_path)):
#                     hour_path = os.path.join(day_path, hour)
#                     hour_path_out = os.path.join(day_path_out, hour)
#                     if(os.path.isdir(hour_path_out) == False):
#                         createHourFolder(year, month, day, hour)
#                     for filename in sorted(os.listdir(hour_path)):
#                         filename_path = os.path.join(hour_path, filename)
#                         # print(filename)
#                         output_filename = isInputHimaOrRadar(
#                             filename_path, year, month, day, hour)
#                         print("In path: ", hour_path)
#                         print("Input filename: ", filename)  # input_filename
#                         print("Out path: ", hour_path_out)
#                         print("Output filename: ", output_filename)
#                         cropping_image(hour_path, filename,
#                                        hour_path_out, output_filename, "0256_0320")

#                         print('\n')


# croppingradar()
