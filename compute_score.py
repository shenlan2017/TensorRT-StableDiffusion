import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from canny2image_TRT import hackathon

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx]).to("cuda")

def PD(base_img, new_img):
    inception_feature_ref, _ = fid_score.calculate_activation_statistics([base_img], model, batch_size = 1, device="cuda")
    inception_feature, _ = fid_score.calculate_activation_statistics([new_img], model, batch_size = 1, device="cuda")
    pd_value = np.linalg.norm(inception_feature - inception_feature_ref)
    pd_string = F"Perceptual distance to: {pd_value:.2f}"
    print(pd_string)
    return pd_value

def get_score(t, p):
    tFactor = 7000 / t
    # if t < 520 and t > 0:
        # tFactor = 20000 - 28.8462 * t
    # elif t >=520 and t < 2600:
        # tFactor = 5000 - 2.40385 * (t - 520)
    # else:
        # tFactor = 0

    pdFactor = 0
    if p < 4 and p >= 0:
        pdFactor = 1.0 - 0.1*p
    elif p >=4 and p < 8:
        pdFactor = 0.6 - 0.0125 *( p -4)
    elif p >= 8 and p <= 12:
        pdFactor = 0.55 - 0.1375 *(p - 8)
    else:
        pdFactor = 0

    return pdFactor * tFactor

scores = []
latencys = []
hk = hackathon()
hk.initialize()
for i in range(10):
    path = "./pictures_croped/bird_"+ str(i) + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = hk.process(img,
            "a bird",
            "best quality, extremely detailed",
            "longbody, lowres, bad anatomy, bad hands, missing fingers",
            1,
            256,
            20,
            False,
            1,
            9,
            2946901,
            0.0,
            100,
            200)
    end = datetime.datetime.now().timestamp()
    t = (end-start)*1000
    print("time cost is: ", t)
    new_path = "./bird_"+ str(i) + ".jpg"
    cv2.imwrite(new_path, new_img[0])

    # generate the base_img by running the pytorch fp32 pipeline (origin code in canny2image_TRT.py)
    base_path = "base_imgs/" + new_path
    pd_score = PD(base_path, new_path)
    score = get_score(t, pd_score)
    print("score is: ", score)

