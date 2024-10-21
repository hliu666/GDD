#!/user/bin/env python
# -*- coding:utf-8 -*-

"""
利用fix effected model做线性拟合
（1）读取所有年的数据合成后栅格数据
（2）做线性拟合
（3）输出相关统计系数，比如R方、平均绝对误差（mean_absolute_error MAE）、
均方误差（mean_squared_error MSE）等

"""

from osgeo import gdal, osr
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

def generate_multi_tif_by_data_array(in_tif, in_array, out_tif, band_nums=1, invalid_value=None, dtype=gdal.GDT_Byte):
    """
    通过数组生成tif
    :param in_tif:
    :param in_array:
    :param out_tif:
    :param band_nums:
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
    outRaster = driver.Create(out_tif, cols, rows, band_nums, dtype, options=["COMPRESS=LZW"])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())

    if band_nums == 1:
        all_array = [in_array]
    else:
        all_array = in_array

    band = raster.GetRasterBand(1)
    blockSize = band.GetBlockSize()
    xBlockSize = blockSize[0]
    yBlockSize = blockSize[1]

    for b_num in range(band_nums):
        outband = outRaster.GetRasterBand(b_num+1)
        if invalid_value is not None:
            outband.SetNoDataValue(invalid_value)
        temp_array = all_array[b_num]
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
                resultArray = temp_array[i:i + numRows, j:j + numCols]
                if resultArray is None:
                    resultArray = np.ones(srcRasterArray.shape) * invalid_value
                outband.WriteArray(resultArray, j, i)
        outband.FlushCache()

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
    处理数据
    :param area_tag
    :return:
    """
    root_path = getRootPath()
    tif_file = os.path.join(root_path, "{0}/result/avg_data/avg_{1}.tif".format(tag, area_tag))
    bandArray = get_raster_band_array(tif_file)
    df = pd.DataFrame(bandArray, columns=["sday", "eday", "gsl", "gdd", "edd", "pre"])
    df.sday = df.sday.astype(np.int64)
    df.eday = df.eday.astype(np.int64)
    df = df.set_index(["eday", "sday"])
    df.dropna()
    print("-------- use EntityEffects ---------")
    mod = PanelOLS.from_formula('gsl ~ 1 + gdd + edd + pre + EntityEffects', df)
    res = mod.fit(cov_type='unadjusted')
    print(res)

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




