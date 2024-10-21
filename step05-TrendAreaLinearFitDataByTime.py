#!/user/bin/env python
# -*- coding:utf-8 -*-

"""
遥感不去趋势
    GSL = a0 + a1 * GDD + a2 * EDD + a3 * PRE
"""

from osgeo import gdal, osr
import os
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

gdal.UseExceptions()

def getRootPath():
    """
    获取当前根目录
    :return:
    """
    root_path = "/Volumes/MP/GSL/gsl"
    # print(u"root_path is: {0}".format(root_path))
    return root_path

def linearModel(x, y):
    """
    用statsmodels实现线性拟合
    :param x:
    :param y:
    :return:
    """
    X = sm.add_constant(x)   #下面三行代码实现了线性拟合
    mod = sm.OLS(y, X)
    res = mod.fit()          #res是存放拟合结果的对象
    # print(res.summary())   #用res的summary()函数可以给出拟合结果的描述
    return res

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

def generate_multi_tif_by_data_array(in_tif, in_array, band_names, out_tif, band_nums=1, invalid_value=None, dtype=gdal.GDT_Byte):
    """
    通过数组生成tif
    :param in_tif:
    :param in_array:
    :param band_names:
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
        outband.SetCategoryNames([band_names[b_num]])
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

def get_all_tif_array(tif_files):
    """
    获取指定波段的信息
    :param tif_files:
    :return:
    """
    # sday_list = []
    # eday_list = []
    gsl_list = []
    # year_list = []
    gdd_list = []
    edd_list = []
    pre_list = []
    for tif_file in tif_files:
        # print(u"process file: {0}".format(tif_file))
        try:
            raster = gdal.Open(tif_file)
        except Exception as e:
            print(u"error is: {0}".format(e))
            continue

        # sday_band = raster.GetRasterBand(1)
        # _sday_data = sday_band.ReadAsArray()
        # sday_list.append(_sday_data)
        #
        # eday_band = raster.GetRasterBand(2)
        # _eday_data = eday_band.ReadAsArray()
        # eday_list.append(_eday_data)

        gsl_band = raster.GetRasterBand(3)
        _gsl_data = gsl_band.ReadAsArray()
        mask = _gsl_data == 0
        _gsl_data[mask] = np.nan
        gsl_list.append(_gsl_data)

        # year_band = raster.GetRasterBand(4)
        # _year_data = year_band.ReadAsArray()
        # year_list.append(_year_data)

        gdd_band = raster.GetRasterBand(5)
        _gdd_data = gdd_band.ReadAsArray()
        _gdd_data[mask] = np.nan
        gdd_list.append(_gdd_data)

        edd_band = raster.GetRasterBand(6)
        _edd_data = edd_band.ReadAsArray()
        _edd_data[mask] = np.nan
        edd_list.append(_edd_data)

        pre_band = raster.GetRasterBand(7)
        _pre_data = pre_band.ReadAsArray()
        _pre_data[mask] = np.nan
        pre_list.append(_pre_data)

    # sday_data = np.array(sday_list)
    # eday_data = np.array(eday_list)
    gsl_data = np.array(gsl_list)
    # year_data = np.array(year_list)
    gdd_data = np.array(gdd_list)
    edd_data = np.array(edd_list)
    pre_data = np.array(pre_list)
    return {
        "gsl": gsl_data,
        "gdd": gdd_data,
        "edd": edd_data,
        "pre": pre_data
    }

def process_datas(tag, area_tag):
    """
    华北玉米结果：
    :param area_tag:
    :return:
    """
    print("process_datas area_tag: {0}".format(area_tag))
    root_path = getRootPath()
    tif_path = os.path.join(root_path, "{0}/process/merge/{1}".format(tag, area_tag))
    dest_file = os.path.join(root_path, "{0}/result/area_trend/result/linearFit_trend_cof_{1}.tif".format(tag, area_tag))
    tif_files = walkDirFile(tif_path)
    obj = get_all_tif_array(tif_files)
    gsl_arr = obj.get("gsl")
    gdd_arr = obj.get("gdd")
    edd_arr = obj.get("edd")
    pre_arr = obj.get("pre")

    shape = gsl_arr.shape
    rows = shape[1]
    cols = shape[2]

    param_gdd = np.full((rows, cols), np.nan)
    param_edd = np.full((rows, cols), np.nan)
    param_pre = np.full((rows, cols), np.nan)
    param_intercept = np.full((rows, cols), np.nan)
    param_rsquared = np.full((rows, cols), np.nan)
    for row in range(rows):
        for col in range(cols):
            temp_gsl = gsl_arr[:, row, col]
            temp_gdd = gdd_arr[:, row, col]
            temp_edd = edd_arr[:, row, col]
            temp_pre = pre_arr[:, row, col]
            if np.isnan(temp_gsl).any() \
                or np.isnan(temp_gdd).any() \
                or np.isnan(temp_edd).any() \
                or np.isnan(temp_pre).any() \
                or np.isinf(temp_gsl).any() \
                or np.isinf(temp_gdd).any() \
                or np.isinf(temp_edd).any() \
                or np.isinf(temp_pre).any():
                continue

            temp_gsl = temp_gsl[np.logical_not(np.isnan(temp_gsl))]
            temp_gdd = temp_gdd[np.logical_not(np.isnan(temp_gdd))]
            temp_edd = temp_edd[np.logical_not(np.isnan(temp_edd))]
            temp_pre = temp_pre[np.logical_not(np.isnan(temp_pre))]
            X = np.hstack((temp_gdd.reshape(-1, 1), temp_edd.reshape(-1, 1), temp_pre.reshape(-1, 1)))
            y = np.array(temp_gsl.reshape(-1, 1))
            try:
                res = linearModel(X, y)
            except Exception as e:
                print("error is: {}".format(e))
                continue
            if len(res.params) <= 3:
                continue
            _intercept = res.params[0]
            _gdd = res.params[1]
            _edd = res.params[2]
            _pre = res.params[3]
            _rsquared = res.rsquared
            param_gdd[row, col] = _gdd
            param_edd[row, col] = _edd
            param_pre[row, col] = _pre
            param_intercept[row, col] = _intercept
            param_rsquared[row, col] = _rsquared

    generate_multi_tif_by_data_array(
        in_tif=tif_files[0],
        in_array=[param_intercept, param_gdd, param_edd, param_pre, param_rsquared],
        band_names=["intercept", "gdd", "edd", "pre", "rsquared"],
        out_tif=dest_file,
        band_nums=5,
        invalid_value=np.nan,
        dtype=gdal.GDT_Float32
    )

def step1(tag):
    process_datas(tag, "china_hb")
    process_datas(tag, "china_db")

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
            step1(tag)
    print("running end ...")




