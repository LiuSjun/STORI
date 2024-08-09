import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import pearsonr

def import_data(path):
    raster_dataset = gdal.Open(path, gdal.GA_ReadOnly)
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data)
    return bands_data
def write_geotiff(fname, data, geo_transform, projection):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('ENVI')
    rows, cols, nbands = data.shape[:3]
    dataset = driver.Create(fname, cols, rows, nbands, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    if nbands == 1:
        dataset.GetRasterBand(1).WriteArray(data)  # 写入数组数据
    else:
        for i in range(nbands):
            dataset.GetRasterBand(i + 1).WriteArray(data[:, :, i])
    del dataset

# dataroot_predict = r"D:\studydata\OSC\data\CDL_sub\simu_gap\result\BRITS\brits_CDL_simugap_predicteddata.npy"
# dataroot_true = r"D:\studydata\OSC\data\CDL_sub\simu_gap\result\BRITS\brits_CDL_simugap_truedata.npy"
# ndvi_data_path = r"D:\studydata\OSC\data\CDL_sub\sentinel2_SR_NDVI\CDL_S2_NDVI_2019_2020_8d"
# testmask_path = r"D:\studydata\OSC\data\CDL_sub\testmask"
#
# ndvi_data = import_data(ndvi_data_path)
# testmask = import_data(testmask_path)
#
# value_true = np.load(dataroot_true)
# value_pre = np.load(dataroot_predict)
#
#
# rows, cols, n_bands = ndvi_data.shape
# n_samples = rows * cols
# testmask0 = testmask.reshape((n_samples, 1))
# ndvi_data0 = ndvi_data.reshape((n_samples, n_bands))
# test_index0 = np.where(testmask0 == 1)
# test_index0 = test_index0[0]
#
# # 测试区域分块
# patch_num = 17
# test_patch = np.array_split(test_index0, patch_num)
# patch_idx = 0
# test_index = test_patch[patch_idx]
#
# ndvi_data1 = ndvi_data0[test_index, :]
#
# raster_dataset = gdal.Open(ndvi_data_path, gdal.GA_ReadOnly)
# geo_transform = raster_dataset.GetGeoTransform()
# proj = raster_dataset.GetProjectionRef()
#
# # 第一段
# ndvi_dataP = value_pre[:,:,0]
#
#
# #保存图像
# predict_NDVI_series = np.zeros((rows, cols, n_bands))
# for i in range(n_bands):
#     oneresult = ndvi_dataP[:, i]
#     one_img = np.full((rows * cols), np.nan)
#     one_img[test_index] = oneresult
#     onepredict = one_img.reshape((rows, cols))
#     predict_NDVI_series[:, :, i] = onepredict
#
# output_fname = 'D:/studydata/OSC/data/CDL_sub/simu_gap/result/BRITS/CDL_BRITS_NDVI_2019_2020_8d_0.tif'
# write_geotiff(output_fname, predict_NDVI_series, geo_transform, proj)

# ndvi_data_path = r"D:\studydata\OSC\data\CDL_sub\sentinel2_SR_NDVI\CDL_S2_NDVI_2019_2020_8d"
# testmask_path = r"D:\studydata\OSC\data\CDL_sub\testmask"
# trainmask_path = r"D:\studydata\OSC\data\CDL_sub\trainmask"

# ndvi_data_path = r"D:\studydata\OSC\data\CIA_sub\sentinel2_SR_NDVI\CIA_S2_NDVI_2020_2021_8d"
# testmask_path = r"D:\studydata\OSC\data\CIA_sub\testmask"
# trainmask_path = r"D:\studydata\OSC\data\CIA_sub\trainmask"
ndvi_data_path = r"D:\studydata\OSC\data\CIA_sub\sentinel2_SR_NDVI\CIA_S2_NDVI_2020_2021_8d"
testmask_path = r"D:\studydata\OSC\data\CIA_sub\testmask"
trainmask_path = r"D:\studydata\OSC\data\CIA_sub\trainmask"

raster_dataset = gdal.Open(ndvi_data_path, gdal.GA_ReadOnly)
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()

ndvi_data = import_data(ndvi_data_path)
testmask = import_data(testmask_path)
trainmask = import_data(trainmask_path)
rows, cols, n_bands = ndvi_data.shape
n_samples = rows * cols
testmask0 = testmask.reshape((n_samples, 1))
trainmask0 = trainmask.reshape((n_samples, 1))
ndvi_data0 = ndvi_data.reshape((n_samples, n_bands))
test_index0 = np.where(testmask0 == 1)
test_index0 = test_index0[0]
train_index0 = np.where(trainmask0 == 1)
train_index0 = train_index0[0]
testloc0 = np.arange(test_index0.shape[0])

patch_num = 17
test_patch = np.array_split(test_index0, patch_num)
loc_patch = np.array_split(testloc0, patch_num)

fullpre = np.zeros([test_index0.shape[0],n_bands])
for ip in range(patch_num):
    predictpath = 'D:/studydata/OSC/data/CIA_sub/simu_ratio/10/result/BRITS/brits_CIA_prediction_' + str(ip) + '.npy'
    value_pre = np.load(predictpath)
    testloc = loc_patch[ip]
    ndvi_dataP = value_pre[:, :, 0]
    fullpre[testloc,:] = ndvi_dataP

# trainloc0 = np.arange(train_index0.shape[0])
# patch_num = 5
# train_patch = np.array_split(train_index0, patch_num)
# loc_patch = np.array_split(trainloc0, patch_num)
# fulltrain = np.zeros([train_index0.shape[0],n_bands])
# for ip in range(5):
#     predictpath = 'D:/studydata/OSC/data/CDL_sub/result/BRITS/brits_CDL_trainarea_predicteddata' + str(ip) + '.npy'
#     value_pre = np.load(predictpath)
#     trainloc = loc_patch[ip]
#     ndvi_dataP = value_pre[:, :, 0]
#     fulltrain[trainloc,:] = ndvi_dataP

predict_NDVI_series = np.zeros((rows, cols, n_bands))
for i in range(n_bands):
    oneresult = fullpre[:, i]
    # oneresult1 = fulltrain[:, i]
    one_img = np.full((rows * cols), np.nan)
    one_img[test_index0] = oneresult
    # one_img[train_index0] = oneresult1
    onepredict = one_img.reshape((rows, cols))
    predict_NDVI_series[:, :, i] = onepredict


output_fname = 'D:/studydata/OSC/data/CIA_sub/simu_ratio/10/result/CIA_GFSG_NDVI_2020_2021_8d'
write_geotiff(output_fname, predict_NDVI_series, geo_transform, proj)