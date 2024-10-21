#!/user/bin/env python
# -*- coding:utf-8 -*-

"""
DESC：
（1）处理合并SDAY、EDAY、GSL、YEAR、GDD、EDD、PRE波段到一个文件中
（2）波段的顺序是 SDAY、EDAY、GSL、YEAR、GDD、EDD、PRE
（3）波段的无效值是 np.nan
"""

from osgeo import gdal, osr
import os
import numpy as np

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
    band_list = []
    try:
        raster = gdal.Open(src_tif_file)
        mask = None
        band_num = raster.RasterCount
        for i in range(1, band_num + 1):
            band = raster.GetRasterBand(i)
            arr = band.ReadAsArray()
            _temp_mask = np.logical_or(np.isnan(arr), np.isinf(arr))
            if mask is None:
                mask = _temp_mask
            else:
                mask = np.logical_and(mask, _temp_mask)
            band_list.append(arr)
        for i in range(band_num):
            band_list[i][mask] = np.nan
    except RuntimeError as e:
        print("can not open rasterFile : {0}, error is {1}".format(src_tif_file, e))
    return band_list

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


def mergeAllBands(tag):
    """
    合并所有的波段
    sday,eday,gsl,year,gdd,edd,pre
    :return:
    """
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        season_path = os.path.join(root_path, "{0}/season/04clip_tif/{1}".format(tag, key))
        pre_path = os.path.join(root_path, "{0}/process/pre/{1}".format(tag, key))
        gdd_path = os.path.join(root_path, "{0}/process/gdd_year/{1}".format(tag, key))
        edd_path = os.path.join(root_path, "{0}/process/edd_year/{1}".format(tag, key))
        crop_mask_file = os.path.join(root_path, "{0}/basic/maize_{1}.tif".format(tag, key))
        dest_path = os.path.join(root_path, "{0}/process/merge/{1}".format(tag, key))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        crop_data = get_raster_band_info(crop_mask_file).get("array")
        crop_mask = np.logical_or(crop_data <= 0.1, np.isnan(crop_data))
        season_file_list = walkDirFile(season_path)
        for sfile in season_file_list:
            sfile_name = os.path.basename(sfile)
            sfile_name = os.path.splitext(sfile_name)[0]
            year = int(sfile_name.split("_")[-1])
            pre_file = os.path.join(pre_path, "pre_{0}.tif".format(year))
            gdd_file = os.path.join(gdd_path, "gdd_{0}.tif".format(year))
            edd_file = os.path.join(edd_path, "edd_{0}.tif".format(year))
            if not os.path.exists(pre_file) \
                    or not os.path.exists(gdd_file) \
                    or not os.path.exists(edd_file):
                continue
            out_file = os.path.join(dest_path, "merge_{0}.tif".format(year))
            pre_data = get_raster_band_info(pre_file).get("array")
            pre_data[crop_mask] = np.nan
            gdd_data = get_raster_band_info(gdd_file).get("array")
            gdd_data[crop_mask] = np.nan
            edd_data = get_raster_band_info(edd_file).get("array")
            edd_data[crop_mask] = np.nan
            sday_data = get_raster_band_info(sfile, 1).get("array")
            mask = np.logical_or(sday_data == -2, crop_mask)
            sday_data = sday_data.astype(np.float32)
            sday_data[mask] = np.nan
            eday_data = get_raster_band_info(sfile, 2).get("array")
            eday_data = eday_data.astype(np.float32)
            eday_data[mask] = np.nan
            gsl_data = get_raster_band_info(sfile, 3).get("array")
            gsl_data = gsl_data.astype(np.float32)
            gsl_data[mask] = np.nan
            year_data = np.ones(pre_data.shape, np.float32) * year
            year_data[mask] = np.nan

            generate_multi_tif_by_data_array(
                in_tif=sfile,
                in_array=[sday_data,eday_data,gsl_data,year_data,gdd_data,edd_data,pre_data],
                band_names=["sday", "eday", "gsl", "year", "gdd", "edd", "pre"],
                out_tif=out_file,
                band_nums=7,
                invalid_value=np.nan,
                dtype=gdal.GDT_Float32
            )

def calculateAvgData(tag):
    """
    计算均值
    :return:
    """
    root_path = getRootPath()
    key_list = ["china_db", "china_hb"]
    for key in key_list:
        src_path = os.path.join(root_path, "{0}/process/merge/{1}".format(tag, key))
        out_file = os.path.join(root_path, "{0}/result/avg_data/avg_{1}.tif".format(tag, key))
        src_files = walkDirFile(src_path, ext=".tif")
        if len(src_files) == 0:
            continue
        sday_datas = []
        eday_datas = []
        gsl_datas = []
        gdd_datas = []
        edd_datas = []
        pre_datas = []
        for src_file in src_files:
            band_datas = get_raster_band_array(src_file)
            sday_datas.append(band_datas[0])
            eday_datas.append(band_datas[1])
            gsl_datas.append(band_datas[2])
            gdd_datas.append(band_datas[4])
            edd_datas.append(band_datas[5])
            pre_datas.append(band_datas[6])
        sday_data = np.nanmean(np.dstack(sday_datas), axis=2)
        eday_data = np.nanmean(np.dstack(eday_datas), axis=2)
        gsl_data = np.nanmean(np.dstack(gsl_datas), axis=2)
        gdd_data = np.nanmean(np.dstack(gdd_datas), axis=2)
        edd_data = np.nanmean(np.dstack(edd_datas), axis=2)
        pre_data = np.nanmean(np.dstack(pre_datas), axis=2)

        generate_multi_tif_by_data_array(
            in_tif=src_files[0],
            in_array=[sday_data, eday_data, gsl_data, gdd_data, edd_data, pre_data],
            band_names=["sday", "eday", "gsl", "gdd", "edd", "pre"],
            out_tif=out_file,
            band_nums=6,
            invalid_value=np.nan,
            dtype=gdal.GDT_Float32
        )

######################################################################

def step1(tag):
    # 波段合并到一个文件中
    mergeAllBands(tag)

def step2(tag):
    # 计算均值
    calculateAvgData(tag)

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
            print("merge bands")
            step1(tag)
            print("calculate avg data")
            step2(tag)
    print("running end ...")
