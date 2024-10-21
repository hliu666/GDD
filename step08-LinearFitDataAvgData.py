#!/user/bin/env python
# -*- coding:utf-8 -*-

"""
相关数据利用使用sklearn中的linear_model做线性回归
    GSL = a0 + a1 * GDD + a2 * EDD + a3 * PRE
（1）读取所有年的数据合成后栅格数据
（2）做线性拟合
（3）输出相关统计系数，比如R方、平均绝对误差（mean_absolute_error MAE）、
均方误差（mean_squared_error MSE）等

"""

from osgeo import gdal, osr
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

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

def process_data(tag, area_tag):
    """
    处理数据
    :param area_tag:
    :return:
    """
    root_path = getRootPath()
    tif_file = os.path.join(root_path, "{0}/result/avg_data/avg_{1}.tif".format(tag, area_tag))
    print("================process data area_tag: {0}================".format(area_tag))
    try:
        raster = gdal.Open(tif_file)
    except Exception as e:
        print(u"error is: {0}".format(e))
        return

    gsl_band = raster.GetRasterBand(3)
    gsl_data = gsl_band.ReadAsArray()
    _gsl_mask = np.logical_and(
        np.logical_not(np.isnan(gsl_data)),
        np.logical_not(np.isinf(gsl_data))
    )

    gdd_band = raster.GetRasterBand(4)
    gdd_data = gdd_band.ReadAsArray()
    _gdd_mask = np.logical_and(
        np.logical_not(np.isnan(gdd_data)),
        np.logical_not(np.isinf(gdd_data))
    )

    edd_band = raster.GetRasterBand(5)
    edd_data = edd_band.ReadAsArray()
    _edd_mask = np.logical_and(
        np.logical_not(np.isnan(edd_data)),
        np.logical_not(np.isinf(edd_data))
    )

    pre_band = raster.GetRasterBand(6)
    pre_data = pre_band.ReadAsArray()
    _pre_mask = np.logical_and(
        np.logical_not(np.isnan(pre_data)),
        np.logical_not(np.isinf(pre_data))
    )

    _mask = np.logical_and(
        np.logical_and(_gsl_mask, _gdd_mask),
        np.logical_and(_edd_mask, _pre_mask)
    )

    gsl_data = gsl_data[_mask]
    gsl_data = gsl_data.reshape(gsl_data.shape[0], 1)

    gdd_data = gdd_data[_mask]
    gdd_data = gdd_data.reshape(gdd_data.shape[0], 1)

    edd_data = edd_data[_mask]
    edd_data = edd_data.reshape(edd_data.shape[0], 1)

    pre_data = pre_data[_mask]
    pre_data = pre_data.reshape(pre_data.shape[0], 1)

    X = np.hstack((gdd_data, edd_data, pre_data))
    y = np.array(gsl_data)

    model = LinearRegression()
    model.fit(X, y)
    coef = model.coef_
    coef = coef[0]
    intercept = model.intercept_
    intercept = intercept[0]
    print("gdd: {gdd}, edd: {edd}, pre: {pre}, constant: {intercept}".format(
        gdd=coef[0], edd=coef[1], pre=coef[2], intercept=intercept
    ))
    print("score: {0}".format(model.score(X, y)))
    y_predict = model.predict(np.array(X))
    print("mean_absolute_error: {0}".format(mean_absolute_error(y, y_predict)))
    print("mean_squared_error : {0}".format(mean_squared_error(y, y_predict)))
    print("r2_score : {0}".format(r2_score(y, y_predict)))

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




