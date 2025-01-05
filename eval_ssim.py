import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(imageA, imageB):
    # 将图片转换为灰度图，因为SSIM主要在灰度图上进行计算
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 计算两个灰度图像的SSIM
    # 返回值包括SSIM指数和满分图像
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score
    # print("SSIM: {}".format(score))

#
# 读取两幅图像
# imageA = cv2.imread(
#     '/home/wzj/PycharmProjects/sphere_resconstruct/render_gt_black/398259/398259_image_270.0/normal/009000.png')
# # imageA = cv2.imread('/home/wzj/PycharmProjects/sphere_resconstruct/render_gt_images/441708/441708_image_270.0/normal/000000.png')
#
# imageB = cv2.imread('/home/wzj/PycharmProjects/sphere_resconstruct/aaa_new_pe_results/39_32_1128/398259/398259_image_270.0/normal/009000.png')
# # imageB = cv2.imread('/home/wzj/PycharmProjects/sphere_resconstruct/aaa_thingi32_new_results/512_64_36/1_128/398259/398259_image_270.0/normal/009000.png')
# # imageB = cv2.imread("/home/wzj/project/nglod_main_code/result_wzj/lod3/441708/normal/000000.png")
#
# # 计算SSIM
# print(calculate_ssim(imageA, imageB))
# a = 0

if __name__ == "__main__":
    gt_image_dir = "/home/wzj/PycharmProjects/sphere_resconstruct/render_gt_black"
    my_result_dir = "/home/wzj/PycharmProjects/sphere_resconstruct/aaa_thingi32_new_results/512_64_36/4_128"
    # my_result_dir = '/home/wzj/project/nglod_main_code/result_wzj/lod3'
    avgavg_ssim = 0
    i = 0
    for img_dir in sorted(os.listdir(my_result_dir)):
        i+=1
        mname = os.path.splitext(img_dir)[0]
        print("mname = {}".format(mname))
        model_file1 = os.path.join(my_result_dir,img_dir)
        model_file2 = os.path.join(model_file1,mname+"_image_270.0/normal")#我的图片文件夹
        # model_file2 = os.path.join(model_file1, "normal")
        model_gt_1 = os.path.join(gt_image_dir,mname)
        model_gt_2 = os.path.join(model_gt_1,mname+"_image_270.0/normal")#gt图片文件夹
        sum_ssim = 0
        for img in os.listdir(model_file2):

            curr_img_my = os.path.join(model_file2,img)
            curr_img_gt = os.path.join(model_gt_2,img)
            img_my = cv2.imread(curr_img_my)
            img_gt = cv2.imread(curr_img_gt)
            sum_ssim = sum_ssim + calculate_ssim(img_my,img_gt)
        avg_ssim = sum_ssim/30.0
        print(avg_ssim)
        avgavg_ssim = avgavg_ssim + avg_ssim
    print("平均ssim：",avgavg_ssim/i)
