#!/user/bin/env python
# -*- coding:utf-8 -*-

"""
DESC：
（1）将atom相关数据的无效值设置为 np.nan
（2）通过atom相关数据计算GDD、EDD、PRE
"""

from osgeo import gdal, osr
import os
import numpy as np
import datetime
import calendar
import time

gdal.UseExceptions()

def getRootPath():
    """
    获取当前根目录
    :return:
    """
    root_path = "/Volumes/MP/GSL/gsl"
    print(u"root_path is: {0}".format(root_path))
    return root_path

def get_raster_band_info(rasterFileName, bandNumber=1):
    """
    获得影像的信息
    :param rasterFileName:
    :param bandNumber:
    :return:
    """
    try:
        raster = gdal.Open(rasterFileName)
        band = raster.GetRasterBand(bandNumber)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        rtnX = geotransform[2]
        rtnY = geotransform[4]
        cols = raster.RasterXSize
        rows = raster.RasterYSize
        print(u"=============rasterFileName = {0} ================".format(rasterFileName))
        print(u"rtnX = {0}, rtnY = {1}".format(rtnX, rtnY))
        print(u"originX = {0}, originY = {1}".format(originX, originY))
        print(u"pixelWidth = {0}, pixelHeight = {1}".format(pixelWidth, pixelHeight))
        print(u"cols = {0}, rows = {1}".format(cols, rows))
        bandArray = band.ReadAsArray()
        print(u"=======================end==========================")
        return {"array": bandArray, "geotransform": geotransform, "cols": cols, "rows": rows}
    except RuntimeError as e:
        print(u"can not open rasterFile : {0}, error is {1}".format(rasterFileName, e))
        return None

def generate_single_tif_by_data_array(in_tif, in_array, out_tif, invalid_value=None, dtype=gdal.GDT_Byte):
    """
    通过数组生成tif
    :param in_tif:
    :param in_array:
    :param out_tif:
    :param invalid_value:
    :param dtype:
    :return:
    """
    if os.path.exists(out_tif):
        os.remove(out_tif)

    out_tif_path = os.path.dirname(out_tif)
    if not os.path.exists(out_tif_path):
        os.makedirs(out_tif_path)
    try:
        raster = gdal.Open(in_tif)
    except RuntimeError as e:
        print("can not open rasterFile : {0}, error is {1}".format(in_tif, e))
        return False
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(out_tif, cols, rows, 1, dtype, options=["COMPRESS=LZW"])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    if invalid_value is not None:
        outband.SetNoDataValue(invalid_value)

    band = raster.GetRasterBand(1)
    blockSize = band.GetBlockSize()
    xBlockSize = blockSize[0]
    yBlockSize = blockSize[1]
    for i in range(0, rows, yBlockSize):
        if i + yBlockSize < rows:
            numRows = yBlockSize
        else:
            numRows = rows - i
        for j in range(0, cols, xBlockSize):
            if j + xBlockSize < cols:
                numCols = xBlockSize
            else:
                numCols = cols - j
            srcRasterArray = band.ReadAsArray(j, i, numCols, numRows)
            resultArray = in_array[i:i + numRows, j:j + numCols]
            if resultArray is None:
                resultArray = np.ones(srcRasterArray.shape) * invalid_value
            outband.WriteArray(resultArray, j, i)
    outband.FlushCache()

def walkDirFile(srcPath, ext=".tif"):
    """
    遍历文件夹
    :param srcPath:
    :param ext:
    :return:
    """
    if not os.path.exists(srcPath):
        print("not find path:{0}".format(srcPath))
        return None
    if os.path.isfile(srcPath):
        return None

    if os.path.isdir(srcPath):
        fileList = []
        for root, dirs, files in os.walk(srcPath):
            for name in files:
                filePath = os.path.join(root, name)
                if ext:
                    if ext == os.path.splitext(name)[1]:
                        # print("find file: {0}".format(filePath))
                        fileList.append(filePath)
                else:
                    fileList.append(filePath)
        fileList.sort()
        return fileList
    else:
        return None

def getDateInfo(year):
    """
    计算日期相关内容
    :param year:
    :return:
    """
    year = int(year)
    dt = datetime.datetime(year=year, month=12, day=31)
    day_num = int(dt.strftime("%j"))
    date_list = []
    for month in range(1, 13):
        month_day = calendar.monthrange(year, month)[1]
        for day in range(1, month_day + 1):
            date_list.append({
                "year": year,
                "month": month,
                "day": day
            })
    return {
        "day_num": day_num,
        "date_list": date_list
    }

def clip_tif_batch(src_path, dest_path, shp, resample_path):
    """
    批量裁剪
    :param src_path:
    :param dest_path:
    :param shp:
    :param resample_path:
    :return:
    """
    file_list = walkDirFile(src_path, ext=".tif")
    if not file_list:
        print("path:{0} has no files".format(src_path))
        return
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for _file in file_list:
        out_tif = os.path.join(dest_path, os.path.basename(_file))
        out_tif_list = os.path.basename(_file).split("_")
        resample_tif = os.path.join(resample_path, "maize_{0}.tif".format(out_tif_list[1]))
        clip_tif_single(src_file=_file,
                        dest_file=out_tif,
                        shp=shp,
                        resample_tif=resample_tif)

def clip_tif_single(src_file, dest_file, shp, resample_tif):
    """
    裁剪单个文件
    :param src_file:
    :param dest_file:
    :param shp:
    :param resample_tif:
    :return:
    """
    if not os.path.exists(src_file):
        return
    if os.path.exists(dest_file):
        os.remove(dest_file)
    dest_path = os.path.dirname(dest_file)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if resample_tif:
        resample_info = get_raster_band_info(resample_tif)
    else:
        resample_info = get_raster_band_info(src_file)
    if not resample_info:
        return
    xres = resample_info["geotransform"][1]
    yres = resample_info["geotransform"][5]

    cmd = "gdalwarp -co COMPRESS=LZW -tr {xres} {yres} -r average '{inputTif}' '{outputTif}' -cutline '{shapeFile}' -crop_to_cutline".format(
        xres=xres,
        yres=yres,
        inputTif=src_file,
        outputTif=dest_file,
        shapeFile=shp
    )
    print("clip raster tif command is : {0}".format(cmd))
    os.system(cmd)

def calculatePRE(pre_path, season_path, result_path, sday_band=1, eday_band=2):
    """
    计算年降水（PRE）的和
    :param pre_path:
    :param season_path:
    :param result_path:
    :param sday_band: 选择开始日期的波段
    :param eday_band: 选择结束日期的波段
    :return:
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    season_file_list = walkDirFile(season_path)
    for sfile in season_file_list:
        sfile_name = os.path.basename(sfile)
        sfile_name = os.path.splitext(sfile_name)[0]
        year = int(sfile_name.split("_")[-1])
        out_tif = os.path.join(result_path, "pre_{0}.tif".format(year))
        sdayData = get_raster_band_info(sfile, sday_band).get("array")
        edayData = get_raster_band_info(sfile, eday_band).get("array")
        mask1 = sdayData == -2
        _tempArray = np.zeros(sdayData.shape)
        date_info = getDateInfo(year)
        day_num = date_info.get("day_num")
        date_list = date_info.get("date_list")
        flag = False
        for _day in range(1, day_num + 1):
            _day_info = date_list[_day - 1]
            _m = _day_info.get("month")
            _d = _day_info.get("day")
            _pre_file = os.path.join(pre_path, "%04d_%02d_%02d.tif" % (year, _m, _d))
            if not os.path.exists(_pre_file):
                continue
            mask2 = np.logical_or(sdayData > _day, edayData < _day)
            mask3 = np.logical_or(mask1, mask2)
            _preObj = get_raster_band_info(_pre_file)
            _preData = _preObj.get("array")
            mask4 = np.logical_or(mask3, np.isnan(_preData))
            _preData[mask4] = 0
            _tempArray += _preData
            flag = True
        if not flag:
            continue
        _tempArray[mask1] = np.nan

        generate_single_tif_by_data_array(
            in_tif=sfile,
            in_array=_tempArray,
            out_tif=out_tif,
            invalid_value=np.nan,
            dtype=gdal.GDT_Float32
        )

def calculateDayGDD(year, tem_path, result_path):
    """
    计算每日GDD的和
    https://www.docin.com/p-1517945680.html
    :param year:
    :param tem_path:
    :param result_path:
    :return:
    """
    t_base = 0
    t_opt = 30
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    date_info = getDateInfo(year)
    day_num = date_info.get("day_num")
    date_list = date_info.get("date_list")
    for _day in range(1, day_num+1):
        flag = False
        sfile = None
        _dayList = []
        _tem_path = os.path.join(tem_path, str(year))
        _day_info = date_list[_day - 1]
        _m = _day_info.get("month")
        _d = _day_info.get("day")
        out_tif = os.path.join(result_path, "gdd_{0}_{1}_{2}.tif".format(year, _m, _d))
        for _hour in range(24):
            _tem_file = os.path.join(_tem_path, "{0}_{1}_{2}_{3}.tif".format(year, _m, _d, _hour))
            if not os.path.exists(_tem_file):
                continue
            if not sfile:
                sfile = _tem_file
            _temObj = get_raster_band_info(_tem_file)
            _temData = _temObj.get("array")
            _temData[np.isnan(_temData)] = 0
            _dayList.append(_temData)
            flag = True

        if flag:
            _tempArray = np.dstack(_dayList)
            _dayAvgArr = (np.nanmax(_tempArray, axis=2)+np.nanmin(_tempArray, axis=2))/2.0
            _dayAvgArr[_dayAvgArr < t_base] = 0
            _dayAvgArr[_dayAvgArr > t_opt] = t_opt - t_base
            generate_single_tif_by_data_array(
                in_tif=sfile,
                in_array=_dayAvgArr,
                out_tif=out_tif,
                invalid_value=np.nan,
                dtype=gdal.GDT_Float32
            )

def calculateYearGDD(tem_path, season_path, result_path, sday_band=1, eday_band=2):
    """
    计算年GDD的和
    :param tem_path:
    :param season_path:
    :param result_path:
    :param sday_band: 选择开始日期的波段
    :param eday_band: 选择结束日期的波段
    :return:
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    season_file_list = walkDirFile(season_path)
    for sfile in season_file_list:
        sfile_name = os.path.basename(sfile)
        sfile_name = os.path.splitext(sfile_name)[0]
        year = int(sfile_name.split("_")[-1])
        out_tif = os.path.join(result_path, "gdd_{0}.tif".format(year))
        sdayData = get_raster_band_info(sfile, sday_band).get("array")
        edayData = get_raster_band_info(sfile, eday_band).get("array")
        mask1 = sdayData == -2
        _tempArray = np.zeros(sdayData.shape)
        date_info = getDateInfo(year)
        day_num = date_info.get("day_num")
        date_list = date_info.get("date_list")
        flag = False
        for _day in range(1, day_num+1):
            _day_info = date_list[_day - 1]
            _m = _day_info.get("month")
            _d = _day_info.get("day")
            _tem_file = os.path.join(tem_path, "gdd_{0}_{1}_{2}.tif".format(year, _m, _d))
            if not os.path.exists(_tem_file):
                continue
            mask2 = np.logical_or(sdayData > _day, edayData < _day)
            mask3 = np.logical_or(mask1, mask2)
            _temObj = get_raster_band_info(_tem_file)
            _temData = _temObj.get("array")
            mask4 = np.logical_or(mask3, np.isnan(_temData))
            _temData[mask4] = 0
            _tempArray += _temData
            flag = True
        if not flag:
            continue
        _tempArray[mask1] = np.nan

        generate_single_tif_by_data_array(
            in_tif=sfile,
            in_array=_tempArray,
            out_tif=out_tif,
            invalid_value=np.nan,
            dtype=gdal.GDT_Float32
        )

def calculateDayEDD(year, tem_path, result_path):
    """
    计算年EDD的每日和
    :param tem_path:
    :param year:
    :param result_path:
    :return:
    """
    t_opt = 34
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    date_info = getDateInfo(year)
    day_num = date_info.get("day_num")
    date_list = date_info.get("date_list")
    for _day in range(1, day_num+1):
        flag = False
        sfile = None
        _dayList = []
        _tem_path = os.path.join(tem_path, str(year))
        _day_info = date_list[_day - 1]
        _m = _day_info.get("month")
        _d = _day_info.get("day")
        out_tif = os.path.join(result_path, "edd_{0}_{1}_{2}.tif".format(year, _m, _d))
        for _hour in range(24):
            _tem_file = os.path.join(_tem_path, "{0}_{1}_{2}_{3}.tif".format(year, _m, _d, _hour))
            if not os.path.exists(_tem_file):
                continue
            if sfile is None:
                sfile = _tem_file
            _temObj = get_raster_band_info(_tem_file)
            _temData = _temObj.get("array")
            _temData[np.isnan(_temData)] = 0
            _dayList.append(_temData)
            flag = True
        if flag:
            _tempArray = np.dstack(_dayList)
            _dayMaxArr = np.nanmax(_tempArray, axis=2)
            _dayMaxArr[_dayMaxArr < t_opt] = 0
            generate_single_tif_by_data_array(
                in_tif=sfile,
                in_array=_dayMaxArr,
                out_tif=out_tif,
                invalid_value=np.nan,
                dtype=gdal.GDT_Float32
            )

def calculateYearEDD(tem_path, season_path, result_path, sday_band=1, eday_band=2):
    """
    计算年EDD的和
    :param tem_path:
    :param season_path:
    :param result_path:
    :param sday_band: 选择开始日期的波段
    :param eday_band: 选择结束日期的波段
    :return:
    """
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    season_file_list = walkDirFile(season_path)
    for sfile in season_file_list:
        sfile_name = os.path.basename(sfile)
        sfile_name = os.path.splitext(sfile_name)[0]
        year = int(sfile_name.split("_")[-1])
        out_tif = os.path.join(result_path, "edd_{0}.tif".format(year))
        sdayData = get_raster_band_info(sfile, sday_band).get("array")
        edayData = get_raster_band_info(sfile, eday_band).get("array")
        mask1 = sdayData == -2
        _tempArray = np.zeros(sdayData.shape)
        date_info = getDateInfo(year)
        day_num = date_info.get("day_num")
        date_list = date_info.get("date_list")
        flag = False
        for _day in range(1, day_num+1):
            _day_info = date_list[_day - 1]
            _m = _day_info.get("month")
            _d = _day_info.get("day")
            _tem_file = os.path.join(tem_path, "edd_{0}_{1}_{2}.tif".format(year, _m, _d))
            if not os.path.exists(_tem_file):
                continue
            mask2 = np.logical_or(sdayData > _day, edayData < _day)
            mask3 = np.logical_or(mask1, mask2)
            _temObj = get_raster_band_info(_tem_file)
            _temData = _temObj.get("array")
            mask4 = np.logical_or(mask3, np.isnan(_temData))
            _temData[mask4] = 0
            _tempArray += _temData
            flag = True
        if not flag:
            continue
        _tempArray[mask1] = np.nan

        generate_single_tif_by_data_array(
            in_tif=sfile,
            in_array=_tempArray,
            out_tif=out_tif,
            invalid_value=np.nan,
            dtype=gdal.GDT_Float32
        )

######################################################################
def step1(tag):
    """
    计算整个生长季

    计算PRE
    :param tag
    :return:
    """
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        season_path = os.path.join(root_path, "{0}/season/04clip_tif/{1}".format(tag, key))
        pre_path = os.path.join(root_path, "{0}/atom/pre/03clip_tif/{1}".format(tag, key))
        result_path = os.path.join(root_path, "{0}/process/pre/{1}".format(tag, key))
        calculatePRE(pre_path=pre_path,
                     season_path=season_path,
                     result_path=result_path,
                     sday_band=1,
                     eday_band=2)

def step2_1(tag, start_year, end_year):
    """
    由于这个温度数据比较特殊，所以这里计算方式需要做一下改变。
    先根据GDD公式计算每日的GDD合成数据
    :param tag
    :param start_year
    :param end_year
    :return:
    """
    root_path = getRootPath()
    tem_path = os.path.join(root_path, "public/atom/tem/01transform_tif")
    result_path = os.path.join(root_path, "{0}/process/gdd_day".format(tag))
    for year in range(start_year, end_year+1):
        calculateDayGDD(year=year,
                        tem_path=tem_path,
                        result_path=result_path)

def step2_2(tag):
    """
    裁剪计算每日的GDD合成数据的结果
    :param tag
    :return:
    """
    print("clip tif by shape")
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        clip_tif_batch(
            src_path=os.path.join(root_path, "{0}/process/gdd_day".format(tag)),
            dest_path=os.path.join(root_path, "{0}/process/clip_gdd_day/{1}".format(tag, key)),
            shp=os.path.join(root_path, "public/basic/{0}.shp".format(key)),
            resample_path=os.path.join(root_path, "{0}/season/04clip_tif/{1}".format(tag, key))
        )

def step2_3(tag):
    """
    计算整个生长季的GDD
    :param tag
    :return:
    """
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        season_path = os.path.join(root_path, "{0}/season/04clip_tif/{1}".format(tag, key))
        tem_path = os.path.join(root_path, "{0}/process/clip_gdd_day/{1}".format(tag, key))
        result_path = os.path.join(root_path, "{0}/process/gdd_year/{1}".format(tag, key))
        calculateYearGDD(tem_path=tem_path,
                         season_path=season_path,
                         result_path=result_path,
                         sday_band=1,
                         eday_band=2)


def step3_1(tag, start_year, end_year):
    """
    由于这个温度数据比较特殊，所以这里计算方式需要做一下改变。
    先根据EDD公式计算每日的EDD合成数据
    :param tag
    :param start_year
    :param end_year
    :return:
    """
    root_path = getRootPath()
    tem_path = os.path.join(root_path, "public/atom/tem/01transform_tif")
    result_path = os.path.join(root_path, "{0}/process/edd_day".format(tag))
    for year in range(start_year, end_year+1):
        calculateDayEDD(year=year,
                        tem_path=tem_path,
                        result_path=result_path)

def step3_2(tag):
    """
    裁剪计算每日的EDD合成数据的结果
    :param tag
    :return:
    """
    print("clip tif by shape")
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        clip_tif_batch(
            src_path=os.path.join(root_path, "{0}/process/edd_day".format(tag)),
            dest_path=os.path.join(root_path, "{0}/process/clip_edd_day/{1}".format(tag, key)),
            shp=os.path.join(root_path, "public/basic/{0}.shp".format(key)),
            resample_path=os.path.join(root_path, "{0}/season/04clip_tif/{1}".format(tag, key))
        )

def step3_3(tag):
    """
    计算整个生长季的EDD
    :param tag
    :return:
    """
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        season_path = os.path.join(root_path, "{0}/season/04clip_tif/{1}".format(tag, key))
        tem_path = os.path.join(root_path, "{0}/process/clip_edd_day/{1}".format(tag, key))
        result_path = os.path.join(root_path, "{0}/process/edd_year/{1}".format(tag, key))
        calculateYearEDD(tem_path=tem_path,
                         season_path=season_path,
                         result_path=result_path,
                         sday_band=1,
                         eday_band=2)


if __name__ == "__main__":
    t1 = time.time()
    print("running start ...")
    tag_list = {
        "CGMS-WOFOST.Maize": ["firr"],
        # "CGMS-WOFOST.Maize": ["noirr"],
        # "CLM-Crop.Maize": ["firr", "noirr"],
        # "EPIC-IIASA.Maize": ["firr", "noirr"],
        # "GEPIC.Maize": ["firr", "noirr"],
        # "LPJ-GUESS.Maize": ["firr", "noirr"],
        # "LPJmL.Maize": ["firr", "noirr"],
        # "ORCHIDEE-crop.Maize": ["firr", "noirr"],
        # "pAPSIM.Maize": ["firr", "noirr"],
        # "pDSSAT.Maize": ["firr", "noirr"],
        # "PEGASUS.Maize": ["firr"]
    }
    for key, value in tag_list.items():
        for v in value:
            tag = "{0}.{1}".format(key, v)
            print("--------------{0}---------------".format(tag))
            # 计算整个生长季
            print("calculate pre")
            step1(tag)
            print("calculate gdd")
            step2_1(tag, 1980, 2010)
            step2_2(tag)
            step2_3(tag)
            print("calculate edd")
            step3_1(tag, 1980, 2010)
            step3_2(tag)
            step3_3(tag)
    print("running end, use time: {0} ...".format((time.time() - t1)/3600.0))
