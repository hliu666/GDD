#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
DESC：利用fix effected model做线性拟合
（1）读取所有年的数据合成后栅格数据
（2）做线性拟合
GSL = a0 + a1 * GDD + a2 * EDD + a3 * PRE
（3）输出相关统计系数，比如R方、平均绝对误差（mean_absolute_error MAE）、
均方误差（mean_squared_error MSE）等
"""
from osgeo import gdal
import os
import numpy as np
import pandas as pd
import datetime
from linearmodels.panel import PanelOLS

gdal.UseExceptions()

def getRootPath():
    """
    获取当前根目录
    :return:
    """
    root_path = "/Volumes/MP/GSL/gsl"
    # print(u"root_path is: {0}".format(root_path))
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

def get_raster_band_array(src_tif_file):
    """
    获取影像的数据数组
    :param src_tif_file:
    :return:
    """
    bandArray = []
    try:
        raster = gdal.Open(src_tif_file)
        mask = None
        band_list = []
        band_num = raster.RasterCount
        for i in range(1, band_num + 1):
            band = raster.GetRasterBand(i)
            arr = band.ReadAsArray()
            _temp_mask = np.logical_and(np.logical_not(np.isnan(arr)), np.logical_not(np.isinf(arr)))
            if mask is None:
                mask = _temp_mask
            else:
                mask = np.logical_and(mask, _temp_mask)
            band_list.append(arr)
        for i in range(band_num):
            _band_arr = band_list[i][mask]
            bandArray.append(_band_arr.reshape(-1, 1))
    except RuntimeError as e:
        print("can not open rasterFile : {0}, error is {1}".format(src_tif_file, e))
    return np.hstack(bandArray)

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
                    if ext == os.path.splitext(name)[1] \
                        and len(name.split(".")) == 2:
                        fileList.append(filePath)
                else:
                    fileList.append(filePath)
        fileList.sort()
        return fileList
    else:
        return None

def getMonth(year, doy):
    """
    获取月
    :param year:
    :param doy:
    :return:
    """
    sdate = datetime.datetime(int(year), month=1, day=1)
    edate = sdate + datetime.timedelta(days=int(doy) - 1)
    return edate.month

def process_data(tag, area_tag):
    """
    处理预测数据：
    :param area_tag:
    :return:
    """
    print("process data area_tag: {}".format(area_tag))
    root_path = getRootPath()
    src_path = os.path.join(root_path, "{0}/process/merge/{1}".format(tag, area_tag))
    tif_files = walkDirFile(src_path, ext=".tif")
    bandArray = None
    flag = False
    for tif_file in tif_files:
        tempArr = get_raster_band_array(tif_file)
        if not flag:
            bandArray = tempArr
            flag = True
        else:
            bandArray = np.vstack((bandArray, tempArr))

    if not flag:
        return
    df = pd.DataFrame(bandArray, columns=["sday", "eday", "gsl", "year", "gdd", "edd", "pre"])
    df.sday = df.sday.astype(np.int64)
    df.eday = df.eday.astype(np.int64)
    df.year = df.year.astype(np.int64)
    df = df.set_index(["year","eday"])
    df.dropna()
    print("-------- use EntityEffects ---------")
    mod = PanelOLS.from_formula('gsl ~ 1 + gdd + edd + pre + EntityEffects', df)
    res = mod.fit(cov_type='unadjusted')
    print(res)

    # print("-------- use TimeEffects ---------")
    # mod = PanelOLS.from_formula('gsl ~ 1 + gdd + edd + pre + TimeEffects', df)
    # res = mod.fit(cov_type='unadjusted')
    # print(res)


if __name__ == "__main__":
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
            process_data(tag, "china_hb")
            process_data(tag, "china_db")
    print("running end ...")


