#!/user/bin/env python
# -*- coding:utf-8 -*-

"""
DESC: 处理img4数据 处理气象数据 降雨数据
（1）读取数据
（2）转换数据格式
（3）转换数据的投影
（4）裁剪数据
（5）生成数据
"""

from osgeo import gdal
import os
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

def walkDirFile(srcPath, ext=".img"):
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
        resample_tif = os.path.join(resample_path, "maize_{0}.tif".format(out_tif_list[0]))
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


def step3(tag):
    """
    裁剪计算的结果
    :param tag
    :return:
    """
    print("clip tif by shape")
    root_path = getRootPath()
    clip_tif_batch(
        src_path=os.path.join(root_path, "public/atom/pre/02transform_tif"),
        dest_path=os.path.join(root_path, "{0}/atom/pre/03clip_tif/china_hb".format(tag)),
        shp=os.path.join(root_path, "public/basic/china_hb.shp"),
        resample_path=os.path.join(root_path, "{0}/season/04clip_tif/china_hb".format(tag))
    )

    clip_tif_batch(
        src_path=os.path.join(root_path, "public/atom/pre/02transform_tif"),
        dest_path=os.path.join(root_path, "{0}/atom/pre/03clip_tif/china_db".format(tag)),
        shp=os.path.join(root_path, "public/basic/china_db.shp"),
        resample_path=os.path.join(root_path, "{0}/season/04clip_tif/china_db".format(tag))
    )

def step0(tag):
    """
    生成玉米掩膜文件
    :param tag
    :return:
    """
    root_path = getRootPath()
    resample_tif_path = os.path.join(root_path, "{0}/season/04clip_tif/china_hb".format(tag))
    resample_tifs = walkDirFile(resample_tif_path, ext=".tif")
    resample_tif_hb = resample_tifs[0]
    resample_tif_path = os.path.join(root_path, "{0}/season/04clip_tif/china_db".format(tag))
    resample_tifs = walkDirFile(resample_tif_path, ext=".tif")
    resample_tif_db = resample_tifs[0]
    clip_tif_single(
        src_file=os.path.join(root_path, "public/basic/maize_HarvestedAreaFraction.tif"),
        dest_file=os.path.join(root_path, "{0}/basic/maize_china_hb.tif".format(tag)),
        shp=os.path.join(root_path, "public/basic/china_hb.shp"),
        resample_tif=os.path.join(root_path, resample_tif_hb)
    )

    clip_tif_single(
        src_file=os.path.join(root_path, "public/basic/maize_HarvestedAreaFraction.tif"),
        dest_file=os.path.join(root_path, "{0}/basic/maize_china_db.tif".format(tag)),
        shp=os.path.join(root_path, "public/basic/china_db.shp"),
        resample_tif=os.path.join(root_path, resample_tif_db)
    )

if __name__ == "__main__":
    t1 = time.time()
    print("running start ...")

    tag_list = {
        "CGMS-WOFOST.Maize": ["firr", "noirr"],
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
            print("clip maize mask")
            step0(tag)
            print("clip tif")
            step3(tag)
    print("running end, use time: {0} ...".format((time.time() - t1)/3600.0))


