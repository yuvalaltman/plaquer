# ==============================================================================
#                                 IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import os
import re
import zipfile
import shutil
import locale
from natsort import natsorted
import warnings
import nd2
import cv2
from skimage import exposure
from skimage.util import img_as_float, img_as_ubyte
from shapely import geometry
from tensorflow.python.client import device_lib
import tensorflow as tf
from ultralytics import YOLO
import ultralytics
import torch
import torchvision
from ensemble_boxes import weighted_boxes_fusion
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# ==============================================================================
#                                 CODE VERSION
# ==============================================================================
plaquer_version = "v1.1.0"

# ==============================================================================
#                            PARSING INPUT ARGUMENTS
# ==============================================================================

parser = ArgumentParser(description="Counting plaques from ND2 files of fluorescent microscopy images",
			formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--TTA", default=1.0, type=float, help="Amount of Test Time Augmentations (0-1)")
parser.add_argument("-l", "--low-conf", action="store_false", help="Exporting plots of low confidence detections")
parser.add_argument("data", help="Path of the folder containing ND2 images to count")
args = vars(parser.parse_args())

# ==============================================================================
#                                 GPU HANDLING
# ==============================================================================

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
# ==============================================================================
#                             UTILITY FUNCTIONS
# ==============================================================================
  
def validate_folder(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)

# ------------------------------------------------------------------------------

def validate_folders(folders):
  for f in folders:
    if isinstance(f, str):
      validate_folder(f)
    elif isinstance(f, dict):
      for fi in f.values():
        validate_folder(fi)
    elif isinstance(f, list):
      for fi in f:
        validate_folder(fi)

# ------------------------------------------------------------------------------

def count_files(dir, ext=None):
  if ext is not None:
    return len([1 for x in list(os.scandir(dir)) if x.is_file() and x.name.endswith(ext)])
  return len([1 for x in list(os.scandir(dir)) if x.is_file()])

# ------------------------------------------------------------------------------

def read_img(filename, color_fluo={}):
  channels = ["R", "G", "B"]

  img = nd2.imread(filename)
  f = nd2.ND2File(filename)

  metadata = f.metadata
  channelCount = metadata.contents.channelCount

  if np.isin(channels, list(color_fluo.values())).all():
    rgb_order = [
        color_fluo[metadata.channels[i].channel.name]
        for i in range(channelCount)
        if metadata.channels[i].channel.name in color_fluo.keys()
        ]
    ind_rgb = [rgb_order.index(c) for c in channels]
  else:
    ind_rgb = [1, 0, 2]

  f.close()

  return img, ind_rgb

# ------------------------------------------------------------------------------

def preprocess_img(img, ind_rgb=[1,0,2], resize=True, img_size=384, p=(2, 98), rescale=True, ubyte=False):
  ind_channel = np.array(img.shape).argmin()
  if img.shape[ind_channel]>3:
    img = img[:3]
  if ind_channel==0:
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC

  if resize:
    img = cv2.resize(img, (img_size, img_size))
  img = img_as_float(img)
  img = img[:, :, ind_rgb] # GRB -> RGB
  if rescale:
    pvals = np.percentile(img, p)
    img = exposure.rescale_intensity(img, in_range=tuple(pvals))
  if ubyte:
    img = img_as_ubyte(img)
  return img

# ------------------------------------------------------------------------------

def preprocess_sub_img(sub_img):
  sub_img = exposure.rescale_intensity(sub_img)
  sub_img = img_as_ubyte(sub_img)
  return sub_img

# ------------------------------------------------------------------------------

def get_large_img_fname(n):
  return "IMG"+str(n) if n>9 else "IMG0"+str(n)

# ------------------------------------------------------------------------------

def convert_series_to_col_df(s):
  return pd.DataFrame(s.tolist(), index=s.index, columns=[s.name])

# ------------------------------------------------------------------------------

def process_classes_model(classes_model, model_list):
  if classes_model is None:
    classes_model = {mdl: list(classes_id.values()) for mdl in model_list}
  else:
    for exp in classes_model.keys():
      if len(classes_model[exp])>0:
        if isinstance(classes_model[exp][0], str):
          classes_model[exp] = [classes_id[cls] for cls in classes_model[exp]]

  assert list(classes_model.keys()) == model_list
  return classes_model

# ------------------------------------------------------------------------------

def get_rect_corners_from_box(x, y, w, h):
  """
  given a rectangle's center (x, y), its width (w) and height (h), this
  function returns its corner  points coordinates:
  ([x_bottom, x_top], [y_bottom, y_top])
  """
  return ([x-w/2, x+w/2], [y-h/2, y+h/2])

# ------------------------------------------------------------------------------

def get_box_coords(x, y, w, h):
  """
  given a bounding box with center (x,y), width (w) and height (h), this function
  returns the (x,y) coordinates of the bounding box corners as a 2d list
  """
  pts_x, pts_y = get_rect_corners_from_box(x, y, w, h)
  return [[px, py] for px in pts_x for py in pts_y]

# ------------------------------------------------------------------------------

def clip_box_coords(pts, w, h):
  return [[int(np.amin([np.amax([np.round(px), 0]), w])),
           int(np.amin([np.amax([np.round(py), 0]), h]))]
          for px, py in pts]

# ------------------------------------------------------------------------------

def get_cnt_bounds(cnt):
  min_x = np.amin([p[0] for p in cnt])
  max_x = np.amax([p[0] for p in cnt])
  min_y = np.amin([p[1] for p in cnt])
  max_y = np.amax([p[1] for p in cnt])
  return min_x, max_x, min_y, max_y

# ------------------------------------------------------------------------------

def get_box_tl(x, y, w, h, n=1):
  return n*np.array([x-w/2, y-h/2])

# ------------------------------------------------------------------------------

def add_ext(fname, ext=".png"):
  return fname + ext

# ------------------------------------------------------------------------------

# ==============================================================================
#                                 CONSTANTS
# ==============================================================================
 
MODELS = ["synth_12g", "synth_13a"]

n_models = len(MODELS)
weights = [2, 1]

DATA_PATH_SAVE = "inference"

DATA_PATH_LOCAL = {mdl: os.path.join(DATA_PATH_SAVE, mdl) for mdl in MODELS}
DATA_PATH_IMGS = {mdl: os.path.join(DATA_PATH_LOCAL[mdl], "images") for mdl in MODELS}

MAIN_PATH = r"/content/drive/MyDrive/plaquer/"

TRAIN_FOLDER = "train"
PRED_FOLDER = "predict"
WEIGHTS_FOLDER = "weights"

WEIGHTS_PATH = {mdl: os.path.join(MAIN_PATH, WEIGHTS_FOLDER, f"{mdl}_weights.pt") for mdl in MODELS}
PREDICTIONS_PATH_LOCAL = {mdl: os.path.join(DATA_PATH_LOCAL[mdl], "labels") for mdl in MODELS}

validate_folders([DATA_PATH_SAVE, DATA_PATH_LOCAL, DATA_PATH_IMGS, PREDICTIONS_PATH_LOCAL])

IOU = 0.5
AGNOSTIC_NMS = True

color_fluo = {"RFP": "R", "YFP": "G", "CFP": "B"}
classes = ["R", "C+Y", "R+Y", "C", "C+R", "Y", "ALL 3"]
n_classes = len(classes)
classes_id = {k: v for k, v in zip(classes, range(n_classes))}
class_from_id = {v: k for k, v in classes_id.items()}

majority_classes = ["R", "C+Y"]
mid_classes = ["R+Y", "C"]
minority_classes = ["C+R", "Y", "ALL 3"]

classes_model = {"synth_12g": classes, "synth_13a": majority_classes}
classes_model = process_classes_model(classes_model, MODELS)

yolo_stride = 32
stride_factor = 12
IMG_SIZE = int(yolo_stride*stride_factor)
RATIO_PATCHES_L = 0.125
OVERLAP = 0.5
FREQ_MAJ_CLASS = 0.9
MIN_AREA = 16**2 # in pixels², in normalized units relative to IMG_SIZE=384 equals to MIN_AREA=(1/24)²=0.00174
MIN_SIDE = 32 # in pixels, in normalized units relative to IMG_SIZE=384 equals to MIN_SIDE=(1/12)=0.083
SMALL_OBJ_AREA = 32**2 # in pixels², in normalized units relative to IMG_SIZE=384 equals to SMALL_OBJ_AREA=(1/12)²=0.00694

RESIZE = False
RESCALE = True
UBYTE = True
PREPROCESS_SUB_IMG = False if (RESCALE and UBYTE) else True

WEIGHTED_IMG = {"synth_12g": False, "synth_13a": True}

augs = ["hflip", "vflip", "rot90", "gamma"]
TTA = args["TTA"] # test time augmentations, float in [0-1], for the probability of performing augmentations

MIN_AREA_REMOVE = 24**2 # predicted objects with an area smaller than this will be removed (given in pixels²)
MIN_SIDE_REMOVE = 24 # predicted objects with a side shorter than this will be removed (given in pixels) -> NOTE: this is smaller than MIN_SIDE in the patching, i.e., dataset creation
SMALL_OBJ_AREA = 32**2 # in pixels² (for analyzing results, not used in processing)
LARGE_OBJ_AREA = 260**2 # in pixels² (for analyzing results, not used in processing)
BOX_COLUMNS = ["x", "y", "w", "h"]

IOU_NMS = 0.25
IOU_WBF = 0.3
IOA = 0.5
SKIP_BOX_THRESHOLD = 0.001

FIGSIZE = (18, 18)
DPI = 140

ALPHA_LOW = 0.2
ALPHA_HIGH =  0.9

SAVE_PRED_FIGS = True
PLOT_PRED_ID = False
PLOT_CONF = True
CLOSE_FIG = True

LOW_CONF_THRESHOLD = 0.1
LOW_CONF_PERCENTILE = 0.05
EXPORT_LOW_CONF = args["low_conf"]

COUNTING_FNAME = "Counting Result.xlsx"
FILENAMES_DICT_FNAME = "IMGXX to Filename.xlsx"

COLORS = {
    "fig_bg_dark": "#37474f",

    "class": {
        "R": "#ff006e",
        "C": "#3a86ff",
        "Y": "#06d6a0",
        "C+R": "#8338ec",
        "R+Y": "#ffbe0b",
        "C+Y": "#00afb9",
        "ALL 3": "#f1faee"
    }
}

FONT_KW = {
    "plot_text_small" : {
        "fontname": "serif",
        "weight": "normal",
        "size": "10",
        "style": "normal"
    },
}

SEED = 42
tf.random.set_seed(SEED)

# ==============================================================================
#                                 LOADING DATA
# ==============================================================================

INPUT_DATA_PATH = args["data"]

def scan_inputs(input_path):
  input_data_fnames = []
  for entry in os.scandir(input_path):
    if entry.name.endswith(".nd2"):
      input_data_fnames.append(entry.path)
  return input_data_fnames
  
# ==============================================================================
#                                 PATCHING
# ==============================================================================

def calc_dim_pad(d, s):
    last_patch_d = int(np.round((d/s - np.floor(d/s)) * s))
    if last_patch_d>0:
      if last_patch_d>0.5*s:
        return s-last_patch_d
      else:
        return last_patch_d
    return 0

# ------------------------------------------------------------------------------

def get_patches_top_left_coords(W, H, s=410, overlap=0.5):
  # calculate the number of pixels required to be added to the last patch
  # for it to be PATCH_SIZE. note that this will be the amount of additional
  # overlap with the one before last patch in this dimension.
  last_patch_w_pad = calc_dim_pad(W, s)
  last_patch_h_pad = calc_dim_pad(H, s)

  tl_xs = np.arange(0, W-s+last_patch_w_pad+1, overlap*s, dtype=int)
  tl_xs[-1] = W-s

  tl_ys = np.arange(0, H-s+last_patch_h_pad+1, overlap*s, dtype=int)
  tl_ys[-1] = H-s

  return tl_xs, tl_ys

# ------------------------------------------------------------------------------

def add_sample_class_freq_to_dict(class_freq, d, method="append"):
  if method.lower().startswith("a"):
    for cls in d.keys():
      if cls in class_freq.index:
        d[cls].append(class_freq.loc[cls])
      else:
        d[cls].append(0)
  else:
    for cls in class_freq.keys():
      d[cls].extend(class_freq[cls])
  return d

# ------------------------------------------------------------------------------

def get_gaussian_filter_shape(IMG_SIZE):
    return IMG_SIZE//4 - 1

# ------------------------------------------------------------------------------

def blur_image(image, sigma=10):
    filter_shape=get_gaussian_filter_shape(IMG_SIZE)
    return cv2.GaussianBlur(image, (filter_shape, filter_shape), sigma)

# ------------------------------------------------------------------------------

def weighted_image(image, alpha=4, beta=4, gamma=128):
    return image*alpha - blur_image(image)*beta + gamma

# ------------------------------------------------------------------------------

def save_img(sub_img, d, path, large_img_fname, tl, aug=""):
  img_fname = get_patch_fname(path, large_img_fname, tl, d, aug=aug)
  cv2.imwrite(img_fname, cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR))

# ------------------------------------------------------------------------------

def get_patch_fname(path, large_img_fname, tl, d, aug=""):
  ext = ".png"

  if len(aug)>0:
    aug = "_" + aug

  fname_short = large_img_fname +\
  "_" + str(tl).replace(", ", "_").replace("(", "").replace(")", "") +\
  "_" + str(d) + aug + ext

  return os.path.join(path, fname_short)

# ------------------------------------------------------------------------------

def augment(img, rng, s, aug="hflip"):
  img_copy = img.copy()
  if aug=="hflip":
    img_aug = tf.image.flip_left_right(img_copy).numpy()
  elif aug=="vflip":
    img_aug = tf.image.flip_up_down(img_copy).numpy()
  elif aug=="rot90":
    img_aug = tf.image.rot90(img_copy).numpy()
  elif aug=="gamma":
    img_aug = tf.image.adjust_gamma(img_copy, gamma=rng.uniform(low=0.5, high=2, size=[1,])).numpy()

  return img_aug

# ------------------------------------------------------------------------------

def augment_and_save(img, d, rng, t, path, large_img_fname, tl, aug):
  """
  img - original image to augment (3D numpy array)
  d - dimension of the patch size covered in the original large image, before resizing
  rng - random number generator
  t - probability threshold (0-1) for determining whether to augment
  path - absolute path in which augmented image and label will be saved
  large_img_fname - original name of the large image of which img is a sub-image
  tl - coordinates of the top left corner of the sub-image img relative to the large original image
  aug - name of the augmentation to perform
  """
  p = rng.random()
  if p>t:
    img_aug = augment(img, rng, d, aug)
    save_img(img_aug, d, path, large_img_fname, tl, aug)

# ------------------------------------------------------------------------------

def generate_samples_from_patch_inference(sub_img,
                                          d,
                                          tl,
                                          path,
                                          large_img_fname,
                                          rng):
  """
  INPUTS
  sub_img: 3D numpy array containing a patch, or a sub-image, from an originally large image
  d: dimension of the patch size covered in the original large image, before resizing
  tl: tuple (x,y) for the coordinate of the top-left corner of sub_img in the coordinate system relative to the original large image
  path: the absolute path in which samples generated from sub_img should be saved
  large_img_fname: the original name of the large image, of which sub_img is a patch of
  rng: random number generator for reproducability
  """

  # additional preprocessing for sub_img
  if PREPROCESS_SUB_IMG:
    sub_img = preprocess_sub_img(sub_img)

  # save sub-image
  save_img(sub_img, d, path, large_img_fname, tl, aug="")

  # AUGMENTATIONS
  # define a probability threshold. for each possible augmentation, a random
  # probability is drawn - should that probability be higher than the set
  # threshold, the augmentation is done and saved. set a low threshold for
  # sub-images with at least one minority class and a high threshold otherwise
  # iterate over the possible augmentations and for each determine if to
  # augment and if so, save the new image and respective label

  t_aug = 1-TTA
  for aug in augs:
    augment_and_save(sub_img, d, rng, t_aug, path, large_img_fname, tl, aug)

# ------------------------------------------------------------------------------

def generate_samples_from_img_inference(img,
                                        d,
                                        s,
                                        overlap,
                                        weighted,
                                        min_area,
                                        min_side,
                                        path,
                                        large_img_fname):
  """
  INPUTS
  img: 3d numpy array of an original large image
  d: integer of the desired pathces dimension to be taken from img
  s: integer of the desired dimension for each patch. can be the same as d or different if requires resizing
  overlap: a float in the range (0-1) for the amount of overlap between patches
  weighted: a boolean indicating whether the sub_img should be weighted or not
  min_area: a float for the threshold of minimal area of an annotated box to be considered
  min_side: a float for the threshold of minimal side of an annotated box to be considered
  path: the absolute path in which samples generated from sub_img should be saved
  large_img_fname: the original name of the large image, of which sub_img is a patch of
  """
  rng = np.random.default_rng(seed=SEED)

  IMG_W, IMG_H, _ = img.shape
  xv, yv = get_patches_top_left_coords(IMG_W, IMG_H, s=d, overlap=overlap)

  for tlx in xv:
    for tly in yv:
      tl = (tlx, tly)
      sub_img = img[tly:tly+d, tlx:tlx+d, :]
      if s!=d:
        sub_img = cv2.resize(sub_img, (s, s))

      if weighted:
        sub_img = weighted_image(sub_img)

      generate_samples_from_patch_inference(sub_img,
                                            d,
                                            tl,
                                            path,
                                            large_img_fname,
                                            rng)

# ------------------------------------------------------------------------------

def patching_inner(img,
                   ind_rgb,
                   large_img_fname,
                   img_size=IMG_SIZE,
                   resize=RESIZE,
                   rescale=RESCALE,
                   ubyte=UBYTE):

  img_processed = preprocess_img(img,
                                 ind_rgb,
                                 img_size=img_size,
                                 resize=resize,
                                 rescale=rescale,
                                 ubyte=ubyte)
  if SAVE_PRED_FIGS:
    figname_save = os.path.join(DATA_PATH_SAVE, add_ext(large_img_fname))
    cv2.imwrite(figname_save, cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))

  # calculate the inputs for the patching process
  IMG_W, IMG_H, _ = img_processed.shape
  PATCH_SIZE_L = int(np.round(RATIO_PATCHES_L*IMG_W))
  for mdl in MODELS:
    generate_samples_from_img_inference(img_processed,
                                        PATCH_SIZE_L,
                                        img_size,
                                        OVERLAP,
                                        WEIGHTED_IMG[mdl],
                                        MIN_AREA,
                                        MIN_SIDE,
                                        DATA_PATH_IMGS[mdl],
                                        large_img_fname)

# ------------------------------------------------------------------------------

def update_fname_dict(fname_dict, img_name, original_file, channel):
  fname_dict["img_name"].append(img_name)
  fname_dict["original_file"].append(os.path.basename(original_file))
  fname_dict["channel"].append(channel)
  return fname_dict

# ------------------------------------------------------------------------------

def patching(input_data_fnames, img_size=IMG_SIZE, resize=RESIZE, rescale=RESCALE, ubyte=UBYTE):
  idx = 0
  fname_dict = {"img_name": [], "original_file": [], "channel": []}
  for fname in input_data_fnames:
    img_nd2, ind_rgb = read_img(fname, color_fluo)
    if len(img_nd2.shape)>3:
      for channel, im in enumerate(img_nd2):
        large_img_fname = get_large_img_fname(idx+1)
        patching_inner(im, ind_rgb, large_img_fname, img_size, resize, rescale, ubyte)
        fname_dict = update_fname_dict(fname_dict, large_img_fname, fname, channel)
        idx+=1
    else:
      channel = 0
      large_img_fname = get_large_img_fname(idx+1)
      patching_inner(img_nd2, ind_rgb, large_img_fname, img_size, resize, rescale, ubyte)
      fname_dict = update_fname_dict(fname_dict, large_img_fname, fname, channel)
      idx+=1
  return pd.DataFrame.from_dict(fname_dict)

# ==============================================================================
#                                 PREDICTION
# ==============================================================================

def predict_patches(mdl):
  model = YOLO(WEIGHTS_PATH[mdl])
  results = model(DATA_PATH_IMGS[mdl],
                  imgsz=IMG_SIZE,
                  iou=IOU,
                  agnostic_nms=AGNOSTIC_NMS,
                  stream=True,
                  verbose=False,
                  device=DEVICE,
                  project=DATA_PATH_LOCAL[mdl],
                  name="labels",
                  save=False,
                  save_txt=True,
                  save_conf=True,
                  exist_ok=True)

  n_samples = count_files(DATA_PATH_IMGS[mdl], ext=".png")
  for r in tqdm(results, total=n_samples):
    if r.boxes.cls.numel() > 0:
      df = pd.DataFrame(
          np.concatenate([
              r.boxes.cls.cpu().numpy()[..., None],
              r.boxes.xywhn.cpu().numpy(),
              r.boxes.conf.cpu().numpy()[..., None]
              ], axis=1)).astype({0: int})
      fname = os.path.join(PREDICTIONS_PATH_LOCAL[mdl],
                           os.path.basename(r.path).replace(".png", ".txt"))
      df.to_csv(fname, sep=" ", header=False, index=False)
            
# ==============================================================================
#                                 STITCHING
# ==============================================================================

def extract_sample_fname(fname, sep="_"):
  elements = fname.split(sep)
  n_elements = len(elements)
  if n_elements<4:
    raise ValueError("Sample name contains too few elements")

  data = {
      "img_large" : elements[0],
      "tl" : (int(elements[1]), int(elements[2])),
      "d" : int(elements[3]),
      "aug": ""
      }

  if n_elements>4:
    if elements[4] in augs:
      data["aug"] = elements[4]

  return pd.Series(data)

# ------------------------------------------------------------------------------

def add_df_id(df, pref="pred"):
  df[f"{pref}_id"] = df.index.astype(int)
  return df

# ------------------------------------------------------------------------------

def get_annotation_df(path, is_conf=True, ext=".txt"):
  label_cols = ["class_id", "x", "y", "w", "h"]
  if is_conf:
    label_cols.append("conf")

  d = {
      "sample": []
  }

  for col in label_cols:
    d[col] = []

  n_samples = count_files(path, ext)

  for entry in tqdm(os.scandir(path), total=n_samples):
    if entry.name.endswith(ext):
      sample_path = os.path.join(path, entry.name)
      target = pd.read_csv(sample_path, header=None, sep=" ")
      target.columns = label_cols
      sample_basename = os.path.splitext(entry.name)[0]
      n_objs = len(target)
      d["sample"].extend(n_objs*[sample_basename])
      for col in label_cols:
        d[col].extend(target[col].values)

  df = pd.DataFrame.from_dict(d)
  df_additional_data = df["sample"].apply(extract_sample_fname)

  df = pd.concat([df, df_additional_data], axis=1)

  pref = "pred" if is_conf else "target"
  df = add_df_id(df, pref=pref)

  return df

# ------------------------------------------------------------------------------

def inv_augment_labels(x, y, w, h, s, aug="hflip"):
  """
  https://en.wikipedia.org/wiki/Rotation_matrix

  rotating point (x,y) by theta degrees to get (x', y'):
  x' = x*cos(theta) - y*sin(theta)
  y' = x*sin(theta) + y*cos(theta)

  given cos(90)=0, sin(90)=1, cos(270)=0, sin(270)=-1:
  - rotating (x,y) by 90 degrees -> (-y,x)
  - rotating (x,y) by 270 degrees -> (y,-x)

  """
  if aug=="hflip":
    x_new = s-x
    y_new = y
    w_new = w
    h_new = h
  elif aug=="vflip":
    x_new = x
    y_new = s-y
    w_new = w
    h_new = h
  elif aug=="rot90":
    x_new = s-y
    y_new = x
    w_new = h
    h_new = w
  else:
    x_new = x
    y_new = y
    w_new = w
    h_new = h

  return x_new, y_new, w_new, h_new

# ------------------------------------------------------------------------------

def df_inv_augment_labels(df_boxes_patch, s=1):
  df_boxes_patch_aug = df_boxes_patch.copy()
  if isinstance(df_boxes_patch, pd.DataFrame):
    for box_id, box_row in df_boxes_patch.iterrows():
      if box_row["aug"]:
        x_aug, y_aug, w_aug, h_aug = inv_augment_labels(
          box_row["x"],
          box_row["y"],
          box_row["w"],
          box_row["h"],
          s,
          aug=box_row["aug"]
          )
        df_boxes_patch_aug.at[box_id, "x"] = x_aug
        df_boxes_patch_aug.at[box_id, "y"] = y_aug
        df_boxes_patch_aug.at[box_id, "w"] = w_aug
        df_boxes_patch_aug.at[box_id, "h"] = h_aug
  elif isinstance(df_boxes_patch, pd.Series):
    if df_boxes_patch["aug"]:
      x_aug, y_aug, w_aug, h_aug = inv_augment_labels(
          df_boxes_patch["x"],
          df_boxes_patch["y"],
          df_boxes_patch["w"],
          df_boxes_patch["h"],
          s,
          aug=df_boxes_patch["aug"]
          )
      df_boxes_patch_aug["x"] = x_aug
      df_boxes_patch_aug["y"] = y_aug
      df_boxes_patch_aug["w"] = w_aug
      df_boxes_patch_aug["h"] = h_aug

  return df_boxes_patch_aug

# ------------------------------------------------------------------------------

def convert_bbox_to_img_coords(box):
  img_d = box["d"]/RATIO_PATCHES_L
  x_new = (box["x"]*box["d"] + box["tl"][0]) / img_d
  y_new = (box["y"]*box["d"] + box["tl"][1]) / img_d
  w_new = box["w"]*box["d"]/img_d
  h_new = box["h"]*box["d"]/img_d
  return x_new, y_new, w_new, h_new

# ------------------------------------------------------------------------------

def convert_preds_to_img_coords(df):
  bboxes = df.apply(convert_bbox_to_img_coords, axis=1, result_type="expand")
  bboxes.columns = BOX_COLUMNS
  df[BOX_COLUMNS] = bboxes
  return df

# ------------------------------------------------------------------------------

def is_small_box(box):
  img_d = box["d"]/RATIO_PATCHES_L
  w = box["w"]*img_d
  h = box["h"]*img_d
  area = w*h
  return (w<MIN_SIDE_REMOVE) or (h<MIN_SIDE_REMOVE) or (area<MIN_AREA_REMOVE)

# ------------------------------------------------------------------------------

def process_annotation_df(df, remove_small=False):
  # inverse augmentations for bbox dimensions
  df = df_inv_augment_labels(df)
  # convert bbox dimensions relative to image dimensions
  df = convert_preds_to_img_coords(df)
  if remove_small:
    df = df[~df.apply(is_small_box, axis=1)]
  return df

# ------------------------------------------------------------------------------

def get_model_preds(mdl):
  pred_path = PREDICTIONS_PATH_LOCAL[mdl]
  preds_df = get_annotation_df(pred_path, is_conf=True)
  preds_df_processed = process_annotation_df(preds_df, remove_small=True)
  return preds_df_processed

# ------------------------------------------------------------------------------

def _boxes_convert(df, in_fmt="cxcywh", out_fmt="xyxy"):
  """"
  convert boxes from YOLO format to COCO format, i.e.:
  cx, cy, w, h -> x1, y1, x2, y2
  where cx, cy are the box center and x1, y1 and x2, y2 are corner points
  """
  return torchvision.ops.box_convert(
      torch.from_numpy(df[BOX_COLUMNS].values),
      in_fmt=in_fmt,
      out_fmt=out_fmt
      )

# ------------------------------------------------------------------------------

def img_nms(df_preds, iou_threshold=0.5):
  nms_inds = torchvision.ops.nms(
      _boxes_convert(df_preds),
      torch.tensor(df_preds["conf"].values),
      iou_threshold
      )
  return df_preds.iloc[nms_inds]

# ------------------------------------------------------------------------------

def _box_inter_areas(boxes1, boxes2):
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = torchvision.ops._utils._upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter, area1, area2

# ------------------------------------------------------------------------------

def _box_ioa(boxes1, boxes2):
  inter, area1, area2 = _box_inter_areas(boxes1, boxes2)
  ioa1 = inter / area1
  ioa2 = inter / area2
  return ioa1, ioa2

# ------------------------------------------------------------------------------

def _box_ioa_numpy(boxes1, boxes2):
  ioa1, ioa2 = _box_ioa(boxes1, boxes2)
  return ioa1.numpy(), ioa2.numpy()

# ------------------------------------------------------------------------------

def _compute_ioa_matrices(preds, targets):
  ioa_matrix, _ = _box_ioa_numpy(
      _boxes_convert(preds),
      _boxes_convert(targets)
      )
  label_match_matrix = (
        targets["class_id"].values.flatten()[:, None]
        == targets["class_id"].values.flatten()[None, :]
    )
  iou_label_match_matrix = ioa_matrix * label_match_matrix
  return ioa_matrix, iou_label_match_matrix

# ------------------------------------------------------------------------------

def get_ind_contained_boxes(preds, ioa_threshold=0.5):
  ioa_matrix, iou_label_match_matrix = _compute_ioa_matrices(preds, preds)
  inds_remove = []
  for i, b in enumerate(iou_label_match_matrix):
      inds = np.flatnonzero(b>ioa_threshold)
      if inds.size>0:
        for ind in inds:
          if (ind not in inds_remove) and (ind != i):
            inds_remove.append(ind)
  return np.unique(inds_remove)

# ------------------------------------------------------------------------------

def remove_contained_boxes(preds, ioa_threshold=0.5):
  preds = preds.sort_values(by="conf", ascending=False)
  ind_remove = get_ind_contained_boxes(preds, ioa_threshold=ioa_threshold)
  return preds.drop(index=preds.iloc[ind_remove].index.tolist())

# ------------------------------------------------------------------------------

def get_imgs_list(preds_dict):
  imgs = preds_dict[MODELS[0]]["img_large"].unique()
  imgs = natsorted(imgs, key=lambda y: y.lower())
  return imgs

# ------------------------------------------------------------------------------

def predict_ensemble(sample,
                     preds_dict,
                     iou_thr_nms=IOU_NMS,
                     iou_thr_wbf=IOU_WBF,
                     skip_box_thr=SKIP_BOX_THRESHOLD,
                     ioa_thr=IOA,
                     weights=None,
                     classes_model=None):

  boxes_list = []
  scores_list = []
  labels_list = []

  for mdl in preds_dict.keys():

    # filter the predictions of the selected sample (IMGXX)
    boxes_df = preds_dict[mdl].query(f"img_large == '{sample}'")

    # filter predictions of the desired classes from this model
    boxes_df = boxes_df.query(f"class_id in {classes_model[mdl]}")

    # NMS: Non-Max Supression (class agnostic) ---> should be out of the for loop
    boxes_df = img_nms(boxes_df, iou_threshold=iou_thr_nms)

    boxes_list.append(torch.clamp(_boxes_convert(boxes_df), min=0, max=1))
    scores_list.append(boxes_df["conf"].values)
    labels_list.append(boxes_df["class_id"].values)

  # WBF: Weighted Boxes Fusion (class specific)
  boxes, scores, labels = weighted_boxes_fusion(
      boxes_list, scores_list, labels_list,
      weights=weights, iou_thr=iou_thr_wbf, skip_box_thr=skip_box_thr
  )

  # arrange remaining prediction boxes in dataframe
  preds = pd.DataFrame(boxes, columns=BOX_COLUMNS)
  preds[BOX_COLUMNS] = _boxes_convert(preds, in_fmt="xyxy", out_fmt="cxcywh")
  preds["class_id"] = labels.astype(int)
  preds["conf"] = scores
  preds["d"] = boxes_df.iloc[0]["d"].astype(int)
  preds["img_large"] = sample
  preds["pred_id"] = preds.index

  # NMS: Non-Max Supression (class agnostic) AFTER ENSEMBLING PREDICTIONS
  preds = img_nms(preds, iou_threshold=iou_thr_nms)

  # CBR: Contained Boxes Removal (class specific)
  preds = remove_contained_boxes(preds, ioa_threshold=ioa_thr)

  return preds

# ------------------------------------------------------------------------------

def get_ensemble_preds_per_img(iou_nms=IOU_NMS,
                               iou_wbf=IOU_WBF,
                               skip_box_thr=SKIP_BOX_THRESHOLD,
                               ioa=IOA,
                               weights=None,
                               classes_model=None):

  # dictionary with keys=modles, values=predictions (of all images)
  preds_dict = {mdl: get_model_preds(mdl) for mdl in MODELS}
  # list of image code names (IMG01, IMG02, ...)
  imgs_list = get_imgs_list(preds_dict)
  # dictionary with keys=images, values=ensemble predictions
  ensemble_preds_dict = {img: predict_ensemble(img,
                                               preds_dict,
                                               iou_nms,
                                               iou_wbf,
                                               skip_box_thr,
                                               ioa,
                                               weights,
                                               classes_model) for img in imgs_list}
  return ensemble_preds_dict                                              

# ==============================================================================
#                                 COUNTING
# ==============================================================================

def count_plaques(df):
  plaque_count = pd.Series(np.full(len(classes), 0, dtype=int), index=classes, name="count")
  class_id = df["class_id"]
  if isinstance(class_id, pd.DataFrame):
    class_id = class_id.squeeze()
  value_counts = class_id.value_counts()
  for cls_id in value_counts.index.astype(int):
    plaque_count[class_from_id[cls_id]] = value_counts[cls_id]

  plaque_count = convert_series_to_col_df(plaque_count)
  return plaque_count

# ------------------------------------------------------------------------------

def calculate_df_counts_ensemble(ensemble_preds_dict):
  imgs_list = list(ensemble_preds_dict.keys())
  df_count_preds = pd.DataFrame(index=imgs_list, columns=classes, dtype=int)

  for img in imgs_list:
    count_preds = count_plaques(ensemble_preds_dict[img])
    df_count_preds.loc[img] = count_preds.T.loc["count"]

  return df_count_preds.astype(int)
    
# ==============================================================================
#                                 EXPORT
# ==============================================================================

def export_counting_result(df_count):
  writer = pd.ExcelWriter(os.path.join(INPUT_DATA_PATH, COUNTING_FNAME), engine="xlsxwriter")
  df_count.to_excel(writer, sheet_name="Sheet1")
  workbook = writer.book
  worksheet = writer.sheets["Sheet1"]
  worksheet.freeze_panes(1, 0)
  format_class = {cls: workbook.add_format({"bg_color": COLORS["class"][cls]}) for cls in classes}
  for c, cls in enumerate(classes):
    worksheet.conditional_format(0, c+1, len(df_count)+1, c+1,
                                 {"type": "cell",
                                   "criteria": ">=",
                                   "value": 0,
                                   "format": format_class[cls]})

  writer.close()

# ------------------------------------------------------------------------------

def export_fname_dictionary(fname_dict):
  fname_dict.to_excel(os.path.join(INPUT_DATA_PATH, FILENAMES_DICT_FNAME), index=False)

# ------------------------------------------------------------------------------

def get_rect(tl, w, h, cls, sw, lw=2, alpha=0.3):
  rect = mpl.patches.Rectangle(
    tl,
    w*sw,
    h*sw,
    linewidth=lw,
    edgecolor=COLORS["class"][cls],
    facecolor="none",
    alpha=alpha
    )
  return rect

# ------------------------------------------------------------------------------

def plot_box(ax, row, sw, rects=[], lw=2, alpha=0.7, va="bottom", ha="left", plot_id=False, plot_conf=False):
  cls = class_from_id[row["class_id"]]
  tl = get_box_tl(row["x"], row["y"], row["w"], row["h"], sw)
  rects.append(get_rect(tl, row["w"], row["h"], cls, sw, lw=lw, alpha=alpha))
  s = cls
  if plot_conf:
    s += f" ({np.round(row['conf'], 2)}) ".replace("0.", ".")
  if plot_id:
    s += f"\n{row['pred_id']}"
  ax.text(
        *tl,
        s,
        color=COLORS["class"][cls],
        alpha=alpha,
        va=va,
        ha=ha,
        **FONT_KW["plot_text_small"])
  return rects, ax

# ------------------------------------------------------------------------------

def plot_sample_full(preds,
                     img_fname,
                     figsize=FIGSIZE,
                     dpi=DPI,
                     plot_pred_id=PLOT_PRED_ID,
                     plot_conf=PLOT_CONF,
                     classes_plot="all",
                     ids_plot="all",
                     alpha_low=ALPHA_LOW,
                     alpha_high=ALPHA_HIGH,
                     save_fig=False,
                     figname="pred.png",
                     close_fig=False):

  img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
  sw, sh, _ = img.shape

  fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
  fig.patch.set_facecolor(COLORS["fig_bg_dark"])
  ax.set_facecolor(COLORS["fig_bg_dark"])

  ax.imshow(img)

  if isinstance(classes_plot, list):
    classes_plot_id = [classes_id[cls] for cls in classes_plot]
    preds = preds[preds["class_id"].isin(classes_plot_id)]
  elif classes_plot != "all":
    class_plot_id = classes_id[classes_plot]
    preds = preds[preds["class_id"] == class_plot_id]

  alphas = pd.Series([alpha_high]*len(preds), index=preds.index)
  if isinstance(ids_plot, (list, pd.Index, pd.Series, np.ndarray)):
    ids_high = ids_plot
    ids_low = preds.index[~preds.index.isin(ids_high)]
    alphas[ids_low] = alpha_low

  rects = []
  for pred_idx, pred_row in preds.iterrows():
    rects, ax = plot_box(ax, pred_row, sw, rects=rects,
                         alpha=alphas[pred_idx],
                         lw=2,va="top", ha="right",
                         plot_id=plot_pred_id, plot_conf=plot_conf)

  pc = mpl.collections.PatchCollection(rects, match_original=True)
  ax.add_collection(pc)

  ax.axis("off")

  if save_fig:
    fig.savefig(figname, dpi=fig.dpi, bbox_inches="tight")

  if close_fig:
    plt.close(fig)

# ------------------------------------------------------------------------------

def export_prediction_plots(input_data_fnames,
                            fname_dict,
                            ensemble_preds_dict,
                            weights=None,
                            classes_model=None,
                            close_fig=CLOSE_FIG):

  for fname in input_data_fnames:
    original_file = os.path.basename(fname)
    large_img_fnames = fname_dict.query(f"original_file == '{original_file}'")["img_name"].values
    for large_img_fname in large_img_fnames:
      figname_load = os.path.join(DATA_PATH_SAVE, add_ext(large_img_fname))
      figname_save = os.path.join(INPUT_DATA_PATH, add_ext(large_img_fname))

      plot_sample_full(ensemble_preds_dict[large_img_fname],
                       figname_load,
                       figsize=FIGSIZE,
                       dpi=DPI,
                       plot_pred_id=PLOT_PRED_ID,
                       plot_conf=PLOT_CONF,
                       classes_plot="all",
                       save_fig=SAVE_PRED_FIGS,
                       figname=figname_save,
                       close_fig=close_fig)


def plot_pred_box(ax, pred_id, box, img, img_w, img_h):
  x, y, w, h = img_w * box[BOX_COLUMNS].astype(float).values
  pts = get_box_coords(x, y, w, h)
  coords = clip_box_coords(pts, img_w, img_h)
  min_x, max_x, min_y, max_y = get_cnt_bounds(coords)
  box_img = img[min_y:max_y, min_x:max_x, :]
  ax.imshow(box_img)
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])
  box_class = class_from_id[box["class_id"]]
  if box_class in COLORS["class"].keys():
    color = COLORS["class"][box_class]
  else:
    color = "black"
  for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_color(color)
    ax.spines[spine].set_linewidth(2)
  title_text = f"Pred. ID {pred_id}\n{box_class}  ({np.round(box['conf'], 3)}) ".replace("0.", ".")
  ax.set_title(title_text, color=color, fontsize=7, fontfamily="serif")
  return ax

# ------------------------------------------------------------------------------

def plot_low_conf_preds_indivisuals(large_img_fname,
                                    preds_low_conf,
                                    figsize=FIGSIZE,
                                    dpi=DPI,
                                    save_fig=SAVE_PRED_FIGS,
                                    close_fig=CLOSE_FIG):

  figname_load = os.path.join(DATA_PATH_SAVE, add_ext(large_img_fname))
  figname_save = os.path.join(INPUT_DATA_PATH, add_ext(large_img_fname + " low confidence predictions (detailed)"))

  img = cv2.cvtColor(cv2.imread(figname_load), cv2.COLOR_BGR2RGB)
  img_w, img_h, _ = img.shape

  n_boxes = preds_low_conf.shape[0]
  nrows = int(np.ceil(np.sqrt(n_boxes)))
  ncols = int(np.ceil(np.sqrt(n_boxes)))
  fig = plt.figure(figsize=figsize, dpi=dpi, layout="tight")
  fig.patch.set_facecolor(COLORS["fig_bg_dark"])

  i = 0
  for pred_id, box in preds_low_conf.iterrows():
    ax = fig.add_subplot(nrows, ncols, i+1)
    ax = plot_pred_box(ax, pred_id, box, img, img_w, img_h)
    i+=1

  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  if save_fig:
    fig.savefig(figname_save, dpi=fig.dpi, bbox_inches="tight")
  if close_fig:
    plt.close(fig)

# ------------------------------------------------------------------------------

def plot_low_conf_preds_in_large_img(large_img_fname,
                                     preds_low_conf,
                                     plot_pred_id=PLOT_PRED_ID,
                                     plot_conf=PLOT_CONF,
                                     classes_plot="all",
                                     ids_plot="all",
                                     figsize=FIGSIZE,
                                     dpi=DPI,
                                     save_fig=SAVE_PRED_FIGS,
                                     close_fig=CLOSE_FIG):
					     
  figname_load = os.path.join(DATA_PATH_SAVE, add_ext(large_img_fname))
  figname_save = os.path.join(INPUT_DATA_PATH, add_ext(large_img_fname + " low confidence predictions"))

  plot_sample_full(preds_low_conf,
                   figname_load,
                   figsize=figsize,
                   dpi=dpi,
                   plot_pred_id=plot_pred_id,
                   plot_conf=plot_conf,
                   classes_plot=classes_plot,
                   ids_plot=ids_plot,
                   save_fig=save_fig,
                   figname=figname_save,
                   close_fig=close_fig)

# ------------------------------------------------------------------------------

def export_low_conf_preds(input_data_fnames,
                          fname_dict,
                          ensemble_preds_dict,
                          classes_plot="all",
                          figsize=FIGSIZE,
                          dpi=DPI,
                          save_fig=SAVE_PRED_FIGS,
                          close_fig=CLOSE_FIG):

  for fname in input_data_fnames:
    original_file = os.path.basename(fname)
    large_img_fnames = fname_dict.query(f"original_file == '{original_file}'")["img_name"].values
    for large_img_fname in large_img_fnames:
      conf_t = min(LOW_CONF_THRESHOLD,
                  ensemble_preds_dict[large_img_fname]["conf"].quantile(LOW_CONF_PERCENTILE))
      preds_low_conf = ensemble_preds_dict[large_img_fname].query(f"conf < {conf_t}")
      ids_low_conf = preds_low_conf.index

      plot_low_conf_preds_indivisuals(large_img_fname,
                                      preds_low_conf,
                                      figsize=figsize,
                                      dpi=dpi,
                                      save_fig=save_fig,
                                      close_fig=close_fig)

      plot_low_conf_preds_in_large_img(large_img_fname,
                                       ensemble_preds_dict[large_img_fname],
                                       plot_pred_id=True,
                                       plot_conf=True,
                                       classes_plot=classes_plot,
                                       ids_plot=ids_low_conf,
                                       figsize=figsize,
                                       dpi=dpi,
                                       save_fig=save_fig,
                                       close_fig=close_fig)

def export(fname_dict,
           total_count,
           input_data_fnames,
           ensemble_preds_dict,
           weights=None,
           classes_model=None,
	   export_low_conf=EXPORT_LOW_CONF,
           figsize=FIGSIZE,
           dpi=DPI,
           save_fig=SAVE_PRED_FIGS,
           close_fig=CLOSE_FIG):
		   
   export_fname_dictionary(fname_dict)
   export_counting_result(total_count)
   export_prediction_plots(input_data_fnames,
			   fname_dict,
			   ensemble_preds_dict,
			   weights,
			   classes_model,
			   close_fig)
   if export_low_conf:
	   export_low_conf_preds(input_data_fnames,
				 fname_dict,
				 ensemble_preds_dict,
				 classes_plot="all",
				 figsize=figsize,
				 dpi=dpi,
				 save_fig=save_fig,
				 close_fig=close_fig)
	                                       
# ==============================================================================
#                                 RUN SCRIPT
# ==============================================================================

print("\n[START]")
print(f"*** plaquer {plaquer_version} ***")

# 1. HANDLING GPU
print("\n1/7 HANDLING GPU ...")
available_gpus = get_available_gpus()
if len(available_gpus) > 0:
  DEVICE = 0
else:
  DEVICE = "cpu"

# 2. LOADING DATA
print("\n2/7 LOADING DATA ...")
input_data_fnames = scan_inputs(INPUT_DATA_PATH)

# 3. PATCHING
print("\n3/7 PATCHING ...")
fname_dict = patching(input_data_fnames, resize=RESIZE, rescale=RESCALE, ubyte=UBYTE)

# 4. PREDICTING
print("\n4/7 PREDICTING ...")
for mdl in MODELS:
  predict_patches(mdl)
  
# 5. STITCHING
print("\n5/7 STITCHING ...")
ensemble_preds_dict = get_ensemble_preds_per_img(IOU_NMS,
                                                 IOU_WBF,
                                                 SKIP_BOX_THRESHOLD,
                                                 IOA,
                                                 weights,
                                                 classes_model)
												 
# 6. COUNTING
print("\n6/7 COUNTING ...")
total_count = calculate_df_counts_ensemble(ensemble_preds_dict)

# 7. EXPORTING
print("\n7/7 EXPORTING ...")
export(fname_dict,
       total_count,
       input_data_fnames,
       ensemble_preds_dict,
       weights,
       classes_model,
       export_low_conf=EXPORT_LOW_CONF,
       figsize=FIGSIZE,
       dpi=DPI,
       save_fig=SAVE_PRED_FIGS,
       close_fig=CLOSE_FIG)

print("\n[END]")
