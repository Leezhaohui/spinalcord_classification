import os
import shutil
import pandas as pd
import pydicom


# path1 = r"D:\work\spinal_classification\data\multi_xulie\dcm\before_19_t1c_dcm\ependymoma"
# path2 = r"D:\work\spinal_classification\data\multi_xulie\dcm\before_19_t2_t1c_dcm\ependymoma"
# path3 = r"D:\work\spinal_classification\data\multi_xulie\dcm\before_19_t2_t1c\ependymoma"

# for filedir in os.listdir(path1):
#     des_filedir_path = os.path.join(path3, filedir)
#     if not os.path.exists(des_filedir_path):
#         os.makedirs(des_filedir_path)
#     shutil.copytree(os.path.join(path1, filedir), os.path.join(des_filedir_path, "t1c"))

# for filedir in os.listdir(path2):
#     if filedir in os.listdir(path3):
#         shutil.copytree(os.path.join(path2, filedir, "image"), os.path.join(path3, filedir, "t2"))

# path4 = r"D:\work\spinal_classification\data\multi_xulie\dcm\after_19_t1c_t2_dcm\NMO"
# for filedir in os.listdir(path4):
#     print(os.listdir(os.path.join(path4, filedir)), filedir)

# path1 = r"D:\work\spinal_classification\data\multi_xulie\dcm\after_19_t1c_t2_dcm"
# path2 = r"D:\work\spinal_classification\data\multi_xulie\dcm\before_19_t2_t1c"
# path3 = r"D:\work\spinal_classification\data\multi_xulie\dcm\19_dcm"
#
# for lesion in os.listdir(path1):
#     print(lesion)
#     lesion_path1 = os.path.join(path1, lesion)
#     lesion_path2 = os.path.join(path2, lesion)
#     lesion_Path3 = os.path.join(path3, lesion)
#     if not os.path.exists(lesion_Path3):
#         os.makedirs(lesion_Path3)
#     for filedir in os.listdir(lesion_path1):
#         filedir_path = os.path.join(lesion_path1, filedir)
#         shutil.copytree(filedir_path, os.path.join(lesion_Path3, filedir))
#     for filedir in os.listdir(lesion_path2):
#         filedir_path = os.path.join(lesion_path2, filedir)
#         shutil.copytree(filedir_path, os.path.join(lesion_Path3, filedir))

dic = {}
split_path = r"D:\work\spinal_classification\data\multi_xulie\nii_split"
for lesion in os.listdir(split_path):
    lesion_path = os.path.join(split_path, lesion)
    for split in os.listdir(lesion_path):
        split_dir = os.path.join(lesion_path, split)
        for filedir in os.listdir(split_dir):
            dic[filedir] = split

def get_dcm_info(dcm_path, attrs):
    info = []
    dcmpath = os.path.join(dcm_path, os.listdir(dcm_path)[0])
    dcm = pydicom.read_file(dcmpath, force=True)
    for attr in attrs:
        if attr == "slice_num":
            info.append(len(os.listdir(dcm_path)))
        else:
            if hasattr(dcm, attr):
                info.append(getattr(dcm, attr))
            else:
                info.append("")
    return info

t1c_dcm_attrs = ["Manufacturer", "MagneticFieldStrength", "FlipAngle", "RepetitionTime", "EchoTime", "Rows", "Columns", "PixelSpacing", "SliceThickness", "slice_num"]
t2_dcm_attrs = ["Manufacturer", "MagneticFieldStrength", "FlipAngle", "RepetitionTime", "EchoTime", "Rows", "Columns", "PixelSpacing", "SliceThickness", "slice_num"]
column = ["lesion", "filedir", "split"] + t1c_dcm_attrs + t2_dcm_attrs
info_path = "../results/dcm_info.xlsx"

dcm_info = []
dcm_path = r"D:\work\spinal_classification\data\multi_xulie\dcm\19_dcm"
for lesion in os.listdir(dcm_path):
    print(lesion)
    lesion_path = os.path.join(dcm_path, lesion)
    for filedir in os.listdir(lesion_path):
        t1c_dcm_path = os.path.join(lesion_path, filedir, "t1c")
        if "image" in os.listdir(t1c_dcm_path):
            t1c_dcm_path = os.path.join(t1c_dcm_path, "image")
        t2_dcm_path = os.path.join(lesion_path, filedir, "t2")
        if "image" in os.listdir(t2_dcm_path):
            t2_dcm_path = os.path.join(t2_dcm_path, "image")

        one_data = []
        one_data.extend([lesion, filedir, dic[filedir]])
        one_data.extend(get_dcm_info(t1c_dcm_path, t1c_dcm_attrs))
        one_data.extend(get_dcm_info(t2_dcm_path, t2_dcm_attrs))

        dcm_info.append(one_data)

dcm_info = pd.DataFrame(dcm_info, columns=column)
dcm_info.to_excel(info_path, index=False)
