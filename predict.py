import os
import numpy as np
import cv2
import SimpleITK as sitk
import pickle
from keras.models import load_model

def normalize_resize(arr, is_mask=False):
    """
    arr:(n, h, w)
    """
    if not is_mask:
        try:
            arr = (arr - np.mean(arr)) / np.std(arr)
        except:
            arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)
        arr = np.clip(arr, -5, 5)
        try:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        except:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)
        arr1 = []
        for i in range(arr.shape[0]):
            arr_mat = arr[i, :, :]
            arr_mat = cv2.resize(arr_mat, (inputshape[1], inputshape[0]))
            arr1.append(arr_mat)
        arr1 = np.array(arr1)
        return arr1
    else:
        arr[arr != 0] = 1
        arr1 = []
        for i in range(arr.shape[0]):
            arr_mat = arr[i, :, :]
            arr_mat = cv2.resize(arr_mat, (inputshape[1], inputshape[0]), interpolation=cv2.INTER_NEAREST)
            arr1.append(arr_mat)
        arr1 = np.array(arr1)
        arr1 = arr1.astype(np.int)
        return arr1

def get_one_data(filedir_path):
    t2_nii_path = os.path.join(filedir_path, "t2.nii.gz")
    t1c_nii_path = os.path.join(filedir_path, "t1c_regist.nii.gz")
    mask_nii_path = os.path.join(filedir_path, "mask.nii.gz")
    t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_nii_path))
    t1c_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1c_nii_path))
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_nii_path))

    t2_arr, t1c_arr, mask_arr = normalize_resize(t2_arr), normalize_resize(t1c_arr), normalize_resize(mask_arr, is_mask=True)
    one_data = []
    for i in range(mask_arr.shape[0]):
        mask_mat = mask_arr[i, :, :]
        t1c_mat = t1c_arr[i, :, :]
        t2_mat = t2_arr[i, :, :]

        t2_mat = cv2.bitwise_and(t2_mat, mask_mat)
        t1c_mat = cv2.bitwise_and(t1c_mat, mask_mat)

        image_mask = np.concatenate([t1c_mat[:, :, np.newaxis], t2_mat[:, :, np.newaxis], mask_mat[:, :, np.newaxis]], axis=-1)  # (256, 256, 3) 0-1
        for index in delete_indexs:
            image_mask[:, :, index] = 0
        if mask_mat.any():
            one_data.append(image_mask)
    one_data = np.array(one_data)
    return one_data

def predict_one_prob(model, filedir_path):
    one_data = get_one_data(filedir_path)
    res = model.predict(one_data) # (n, 2)
    prob = np.mean(res[:, 1])
    return prob

if __name__ == "__main__":
    inputshape = (256, 256, 3)
    data_root = "/hd/Lizh/Lizh/spinal_classification_t2_t1c/nii"

    split_path = "../split_MS_NMO.pkl"
    lesions = ["astrocytoma", "ependymoma"]
    model_path = "../models/MS_NMO_t2+t1c_zscore_clip.h5"

    #model_path = "../models/astr_epen_t2_zscore_clip.h5"
    #model_path = "../models/astr_epen_t1c_zscore_clip.h5"
    delete_indexs = []
    model = load_model(model_path, compile=False)

    labels_pres_save_path = "../results"
    labels_pres_name = os.path.split(model_path)[-1][:-3] + "_labels_pres.pkl"
    if not os.path.exists(labels_pres_save_path):
        os.makedirs(labels_pres_save_path)

    with open(split_path, "rb") as f:
        split = pickle.load(f)
    test0_filedirs = split["test0"]
    test1_filedirs = split["test1"]
    print(len(test0_filedirs), len(test1_filedirs))

    labels, pres = [], []
    for lesion in lesions:
        lesion_path = os.path.join(data_root, lesion)
        for filedir in os.listdir(lesion_path):
            if filedir in test0_filedirs:
                filedir_path = os.path.join(lesion_path, filedir)
                labels.append(0)
                prob = predict_one_prob(model, filedir_path)
                pres.append(prob)
                print(filedir, 0, prob)
            elif filedir in test1_filedirs:
                filedir_path = os.path.join(lesion_path, filedir)
                labels.append(1)
                prob = predict_one_prob(model, filedir_path)
                pres.append(prob)
                print(filedir, 1, prob)
    with open(os.path.join(labels_pres_save_path, labels_pres_name), "wb") as f:
        pickle.dump((labels, pres), f)
