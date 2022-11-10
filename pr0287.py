# ################################################################################
# Library
# ################################################################################
import numpy as np
import torch
from torchmetrics import JaccardIndex
from os.path import basename

# ################################################################################
# 2차원 배열에 원하는 값이 몇개 있는지 확인
# ################################################################################
def count_val_in_2darray(arr, ele):
    M = arr.reshape(1, -1).tolist()[0]
    return M.count(ele)

# ################################################################################
# tensor에서 특정 값의 개수를 세는 방법
# ################################################################################
def count_val_in_tensor(t1, val):
    ele  = torch.sum(t1 ==val).item()
    return ele

# ################################################################################
# Seg acc 계산
# ################################################################################
def seg_acc(segs, anns, batch_size):
    # segs와 anns는 둘다 tensor이다.

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    for i in range(batch_size):

        seg = segs[i]
        ann = anns[i]

        TP_px = torch.sum((seg == ann)*(ann == 1)).item()
        TN_px = torch.sum((seg == ann)*(ann == 0)).item()
        FP_px = torch.sum((seg != ann)*(ann == 0)).item()
        FN_px = torch.sum((seg != ann)*(ann == 1)).item()

        TP_all += TP_px
        TN_all += TN_px
        FP_all += FP_px
        FN_all += FN_px

    # 전체 픽셀에서 맞춘 픽셀 수
    acc = round((TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all),4)

    # 검출된 것의 정확도
    precision = round(TP_all / (TP_all + FP_all + 1),5)

    # 전체 크랙에서 몇퍼센트를 맞추었는지
    recall = round(TP_all / (TP_all + FN_all + 1),5)

    TPr = round(TP_all / (TP_all + TN_all + FP_all + FN_all), 4)
    TNr = round(TN_all / (TP_all + TN_all + FP_all + FN_all), 4)
    FPr = round(FP_all / (TP_all + TN_all + FP_all + FN_all), 4)
    FNr = round(FN_all / (TP_all + TN_all + FP_all + FN_all), 4)


    return acc, precision, recall, TPr, TNr, FPr, FNr

# ################################################################################
# Seg mIou 계산
# ################################################################################
def seg_miou(batch_size, anns, segs, class_cnt):

    iou_list = []

    for i in range(batch_size):
        seg = segs[i]
        ann = anns[i]
        jaccard = JaccardIndex(num_classes=class_cnt)
        rlt = jaccard(ann, seg)
        iou_list.append(rlt.item())

    miou = round(np.mean(iou_list), 4)
    return miou

# ################################################################################
# maskfile list 불러오기
# ################################################################################

def mask_list_load(img_list, mask_dir):
    mask_list = []

    for img_nm in img_list:
        img_filename = basename(img_nm)
        mask_filepath = mask_dir + img_filename
        mask_list.append(mask_filepath)

    return mask_list


# ################################################################################
if __name__ == '__main__':
    arr = np.array([[1,1,2],[1,3,3]])
    val = 3
    print(count_val_in_2darray(arr, val))
