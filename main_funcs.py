# Functions used more globally
import platform
import sys, os, json, math, time, yaml
from xml.etree.ElementPath import ops
from pathlib import Path
import pandas as pd
import numpy as np
#import plot_funcs as pfun
import utils_funcs as utils
from datetime import datetime
import re
import pickle
import glob
import scipy.io as sio
from tifffile import imread
import imageio.v2 as imageio
import xml.etree.ElementTree as ET
from suite2p.run_s2p import run_s2p
from typing import Optional
import os
import cupy


class analysis:
    
    def __init__(self, animalList=None):
        print('Env: ' + os.environ['CONDA_DEFAULT_ENV'])
        
        if platform.node() == 'macOS-12.6-arm64-arm-64bit':
            # for Windows - Huriye PC
            print("Computer: Huriye MAC")
            self.suite2pOutputPath = 'N/A' 
            self.recordingListPath = "/Users/Huriye/Documents/Code/decision-making-ev/"
            self.rawPath           = 'N/A' # this folder is only avaiable with PC  
            self.rootPath          = "/Users/Huriye/Documents/Code/decision-making-ev/"
        elif platform.node() == 'WIN-AMP016':
            print("Computer: Huriye Windows")
            # for Windows - Huriye PC
            self.suite2pOutputPath = 'C:\\Users\\Huriye\\Documents\\code\\sideBiasLateralisation\\suite2p_output\\' 
            self.recordingListPath = "C:\\Users\\Huriye\\Documents\\code\\sideBiasLateralisation\\"
            self.rawPath           = 'Z:\\' 
            self.rootPath          = "C:\\Users\\Huriye\\Documents\\code\\sideBiasLateralisation\\"
        elif platform.node() == 'WIN-AL012':
            print("Computer: Candela Windows")
            # for Windows - Huriye PC
            self.suite2pOutputPath = '//QNAP-AL001.dpag.ox.ac.uk/CMedina/cingulate_DMS/suite2p_output' 
            self.recordingListPath = "C:/Users/Lak Lab/Documents/Github/sideBiasLateralisation"
            self.rawPath           = 'Z:/' 
            self.rootPath          = "C:/Users/Lak Lab/Documents/Github/sideBiasLateralisation"
        else:
            print('Computer setting is not set.')
        self.analysisPath = os.path.join(self.rootPath, 'analysis') # 'D:\\decision-making-ev\\analysis' # 
        self.figsPath     = os.path.join(self.rootPath, 'figs')
        self.ops_suite2pPath = os.path.join(self.rootPath, 'ops_suite2p.json')
        self.ops_yaml_path = os.path.join(self.rootPath, 'ops_suite2p.yaml')
        #self.DLCconfigPath = os.path.join(self.rootPath, 'pupilExtraction', 'Updated 5 Dot Training Model-Eren CAN-2021-11-21')
        #self.DLCconfigPath = self.DLCconfigPath + '\\config.yaml'
        
        # Create the list 
        info = pd.DataFrame()
        # Recursively search for files ending with 'Block.mat' in all subfolders
        if animalList is None:
            animalList = ['MBL015', 'MBL014']
            print('No animal ID is given, so all animals will be processed: ' + str(animalList))
        badRecordingSessions = ['2023-07-07_1_OFZ008_Block.mat', '2023-07-07_3_OFZ008_Block.mat', # Not good ROIs
                                ]
        acceptableExpDef = {'Grating2AFC_variableStimSize','Grating2AFC_variableStimSize_variabledelay'}
        rows = []
        infoCreation = True

        for animal in animalList: 
            animalPath = os.path.join(self.rawPath, animal)

            # if animalPath exists
            if not os.path.exists(animalPath):
                print(f"PROBLEM IN ANIMAL: {animal} does not exist: {animalPath}")
                infoCreation = False
                continue

            for root, dirs, files in os.walk(animalPath):
                for fileName in files:
                    if fileName in badRecordingSessions or not fileName.endswith('Block.mat'):
                        continue

                    # Load the .mat file
                    file_path = os.path.join(root, fileName)
                    mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
                    block = mat_data.get('block', None)
                    if block is None:
                        print(f"No 'block' struct in {file_path}")
                        continue


                    # Extract fields
                    rigName = getattr(block, 'rigName', None)
                    expDef_full = getattr(block, 'expDef', '')
                    expDef = os.path.splitext(os.path.basename(expDef_full))[0] if expDef_full else None

                    # Check acceptable experiment definitions
                    if expDef not in acceptableExpDef:
                        continue

                    # get performance
                    try:
                        stim_type, performance = get_stim_and_performance(block)
                    except Exception as e:
                        print(f"Most likely expRef does not match {file_path}: {e}")
                        stim_type, performance = None, None
                        continue

                    expDuration = getattr(block, 'duration', None)
                    # Parse filename (assuming format: YYYY-MM-DD_<recID>_<animalID>_Block.mat)
                    parts = fileName.split('_')
                    if len(parts) < 3:
                        print(f"Unexpected filename format: {fileName}")
                        continue
                    recordingDate, recording_id, animal_id = parts[0], parts[1], parts[2]

                    # Convert date string
                    date = datetime.strptime(recordingDate, '%Y-%m-%d')

                    # Mark the imaging session
                    twoP_path = os.path.join(os.path.dirname(root), 'TwoP')
                    tiff_files = glob.glob(os.path.join(twoP_path, '*t-001'))
                    twoP_exist = len(tiff_files) > 0
                    imaging_filename = tiff_files[0] if twoP_exist else None

                    # Collect row
                    row_data = {
                        'animalID': animal_id, 
                        'recordingDate': recordingDate, 
                        'recordingID': recording_id, 
                        'sessionName': fileName[:-10],
                        'twoP': twoP_exist,
                        'path': os.path.dirname(root),
                        'imagingTiffFileNames': imaging_filename,
                        'sessionNameWithPath': file_path,
                        'blockName': fileName[:-10],
                        'experimentDefinition': expDef,
                        'duration': expDuration/60, # in mins
                        'rigName': rigName,
                        'performance': performance,
                        'stimType': stim_type

                    }
                    rows.append(row_data)


        if infoCreation:  # build dataframe once
            info = pd.DataFrame(rows)
            self.recordingList = info

            # Add main filepathname
            self.recordingList['analysispathname'] = pd.Series(dtype='string')
            for ind, recordingDate in enumerate(self.recordingList.recordingDate):
                filepathname = (self.recordingList.path[ind] +
                                '\\'+ self.recordingList.recordingID[ind])
                self.recordingList.loc[ind,'filepathname'] = filepathname  

                analysispathname = (self.analysisPath +
                                    '\\' + self.recordingList.recordingDate[ind] + '_' + 
                                    str(self.recordingList.animalID[ind]) + '_' + 
                                    self.recordingList.recordingID[ind])   
                self.recordingList.loc[ind,'analysispathname'] = os.path.join(analysispathname, '')


def start_matlab_safely(mlroot=r"C:\Program Files\MATLAB\R2025a"):
    import os
    added = []
    # Add MATLAB DLL dirs just long enough to start the engine
    for p in [
        os.path.join(mlroot, "bin", "win64"),
        os.path.join(mlroot, "bin", "win64", "mvm_transport", "mvm_transport"),
    ]:
        try:
            h = os.add_dll_directory(p)   # returns a handle we can close
            added.append(h)
        except Exception:
            pass

    import matlab.engine
    eng = matlab.engine.start_matlab()

    # Now drop those dirs so NumPy/SciPy/OpenCV use their own DLLs
    for h in added:
        try:
            h.close()
        except Exception:
            pass
    return eng

                    
def convert_tiff2avi (imagename, outputsavename, fps=30.0):
    # path = 'Z:\Data\\2022-05-09\\2022-05-09_22107_p-001\\'
    # filename = path + 'test.tif'
    # outputsavename = 'Z:\\Data\\2022-05-09\\2022-05-09_22107_p-001\\output.avi'
    # fps = 30.0

     #Load the stack tiff & set the params
     imageList = pims.TiffStack(imagename)
     nframe, height, width = imageList.shape
     size = width,height
     duration = imageList.shape[0]/fps
     # Create the video & save each frame
     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
     out = cv2.VideoWriter(outputsavename, fourcc, fps,size,0)
     for frame in imageList:
        # frame = cv2.resize(frame, (500,500))
         out.write(frame)
     out.release()
     
def get_file_names_with_strings(pathIn, str_list):
    full_list = os.listdir(pathIn)
    final_list = [nm for ps in str_list for nm in full_list if ps in nm]

    return final_list

def fdr(p_vals):
    #http://www.biostathandbook.com/multiplecomparisons.html
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

def calculateDFF (tiff_folderpath, frameClockfromPAQ):
    s2p_path = tiff_folderpath +'\\suite2p\\plane0\\'
    # from Vape - catcher file: 
    flu_raw, _, _ = utils.s2p_loader(s2p_path, subtract_neuropil=False) 

    flu_raw_subtracted, spks, stat = utils.s2p_loader(s2p_path)
    flu = utils.dfof2(flu_raw_subtracted)

    _, n_frames = tiff_metadata(tiff_folderpath)
    tseries_lens = n_frames

    # deal with the extra frames 
    frameClockfromPAQ = frameClockfromPAQ[:tseries_lens[0]] # get rid of foxy bonus frames

    # correspond to analysed tseries
    paqio_frames = utils.tseries_finder(tseries_lens, frameClockfromPAQ, paq_rate=20000)
    paqio_frames = paqio_frames

    if len(paqio_frames) == sum(tseries_lens):
        print('Dff extraction is completed: ' +tiff_folderpath)
        imagingDataQaulity = True
       # print('All tseries chunks found in frame clock')
    else:
        imagingDataQaulity = False
        print('WARNING: Could not find all tseries chunks in '
              'frame clock, check this')
        print('Total number of frames detected in clock is {}'
               .format(len(paqio_frames)))
        print('These are the lengths of the tseries from '
               'spreadsheet {}'.format(tseries_lens))
        print('The total length of the tseries spreasheets is {}'
               .format(sum(tseries_lens)))
        missing_frames = sum(tseries_lens) - len(paqio_frames)
        print('The missing chunk is {} long'.format(missing_frames))
        try:
            print('A single tseries in the spreadsheet list is '
                  'missing, number {}'.format(tseries_lens.index
                                             (missing_frames) + 1))
        except ValueError:
            print('Missing chunk cannot be attributed to a single '
                   'tseries')
    return {"imagingDataQaulity": imagingDataQaulity,
            "frame-clock": frameClockfromPAQ,
            "paqio_frames":paqio_frames,
            "n_frames":n_frames,
            "flu": flu,
            "spks": spks,
            "stat": stat,
            "flu_raw": flu_raw}

def calculatePupil (filename, frameClockfromPAQ):
    dataPupilCSV = pd.read_csv(filename [0], header = 1)
    dataPupilCSV.head()

    verticalTop_x =np.array(dataPupilCSV['Xmax'] [1:], dtype = float)
    verticalTop_y =np.array(dataPupilCSV['Xmax.1'][1:], dtype = float)
    verticalBottom_x =np.array(dataPupilCSV['Xmin'][1:], dtype = float)
    verticalBottom_y =np.array(dataPupilCSV['Xmin.1'][1:], dtype = float)
    verticallikelihood =np.mean(np.array(dataPupilCSV['Xmax.2'][1:], dtype = float))

    verticalDis = np.array(np.sqrt((verticalTop_x - verticalBottom_x)**2 + (verticalTop_y - verticalBottom_y)**2))

    horizontalTop_x =np.array(dataPupilCSV['Ymax'] [1:], dtype = float)
    horizontalTop_y =np.array(dataPupilCSV['Ymax.1'][1:], dtype = float)
    horizontalBottom_x =np.array(dataPupilCSV['Ymin'][1:], dtype = float)
    horizontalBottom_y =np.array(dataPupilCSV['Ymin.1'][1:], dtype = float)
    horizontallikelihood =np.mean(np.array(dataPupilCSV['Ymax.2'][1:], dtype = float))

    horizontalDis = np.array(np.sqrt((horizontalTop_x - horizontalBottom_x)**2 + (horizontalTop_y - horizontalBottom_y)**2))

    lengthCheck = len(frameClockfromPAQ)==len(horizontalDis)

    return {"verticalTop_x": verticalTop_x,
            "verticalTop_y": verticalTop_y,
            "verticalBottom_x": verticalBottom_x,
            "verticalBottom_y": verticalBottom_y,
            "verticallikelihood": verticallikelihood,
            "verticalDis": verticalDis,
            "horizontalTop_x": horizontalTop_x,
            "horizontalTop_y": horizontalTop_y,
            "horizontalBottom_x": horizontalBottom_x,
            "horizontalBottom_y": horizontalBottom_y,
            "horizontallikelihood": horizontallikelihood,
            "horizontalDis": horizontalDis,
            "lengthCheck": lengthCheck,
            "frameClockfromPAQ": frameClockfromPAQ}

def get_stim_and_performance(block):
    """
    Exact match to MATLAB logic for high-contrast performance - instead of getBehav  code

        cDiff = contrastRight - contrastLeft
        goodTrials = repeatNumber==1 & choice!='NoGo'   (choice: -1/1/0 => Left/Right/NoGo)
        outcomeTime = nanmean([rewardTime, punishSoundOnsetTime], axis=1)  # computed but not used
        highCPerformance = 100 * mean(feedback(good & abs(cDiff)==max(abs(cDiff))) == 'Rewarded')

    Returns
    -------
    stim_type : str
    performance : float (percent), NaN if not computable
    """
    # ---- helpers ----
    def _get(obj, name, default=None): 
        return getattr(obj, name, default)

    def _arr(x):
        if x is None: 
            return None
        a = np.asarray(x)
        return a.squeeze()

    # Stimulus type from block.expDef
    exp_def_full = _get(block, 'expDef', '')
    if isinstance(exp_def_full, bytes):
        exp_def_full = exp_def_full.decode(errors='ignore')
    stim_type = "Unknown"
    if exp_def_full:
        base = os.path.basename(str(exp_def_full).replace('\\', os.sep).replace('/', os.sep))
        stim_type = base[:-2] if base.endswith('.m') else base

    events = _get(block, 'events', None)
    outputs = _get(block, 'outputs', None)
    if events is None:
        return stim_type, np.nan

    endTrialTimes = _arr(_get(events, 'endTrialTimes', None))
    if endTrialTimes is None or endTrialTimes.size == 0:
        return stim_type, np.nan
    N = int(endTrialTimes.size)

    # --- required fields for exact computation ---
    cL = _arr(_get(events, 'contrastLeftValues', None))
    cR = _arr(_get(events, 'contrastRightValues', None))
    rep = _arr(_get(events, 'repeatNumValues', None))
    choice = _arr(_get(events, 'responseValues', None))          # -1/1/0
    feedback = _arr(_get(events, 'feedbackValues', None))        # 0/1 or -1/1

    # If contrasts are missing, we cannot do the "max |cDiff|" logic -> NaN (exact behavior can’t be reproduced)
    if cL is None or cR is None or rep is None or choice is None or feedback is None:
        return stim_type, np.nan

    # trim to N trials
    def _trim(a):
        return a[:N] if a is not None else None

    cL, cR = _trim(cL), _trim(cR)
    rep, choice, feedback = _trim(rep), _trim(choice), _trim(feedback)

    # Build table (like MATLAB table)
    b = pd.DataFrame({
        "contrastLeft":  cL.astype(float, copy=False),
        "contrastRight": cR.astype(float, copy=False),
        "repeatNumber":  rep.astype(float, copy=False),
        "choice":        choice.astype(float, copy=False),
        "feedback_raw":  feedback.astype(float, copy=False),
    })

    # Map feedback to strings 'Unrewarded'/'Rewarded' (exact semantics)
    # MATLAB uses categorical([0 1], {'Unrewarded','Rewarded'}) or [-1 1] -> map >0 to Rewarded
    b["feedback"] = np.where(b["feedback_raw"] > 0, "Rewarded", "Unrewarded")

    # cDiff and goodTrials (exact)
    b["cDiff"] = b["contrastRight"] - b["contrastLeft"]
    goodTrials = (b["repeatNumber"] == 1) & (b["choice"] != 0)

    # outcomeTime row-wise nanmean of [rewardTime, punishSoundOnsetTime] (computed but not used)
    rewardTime = _arr(_get(events, 'rewardTimes', None))  # some profiles
    if rewardTime is None and outputs is not None:
        rewardTime = _arr(_get(outputs, 'rewardTimes', None))  # _noTimeline profile
    punishTime = _arr(_get(events, 'punishSoundOnsetTime', None))
    rewardTime = _trim(rewardTime) if rewardTime is not None else None
    punishTime = _trim(punishTime) if punishTime is not None else None
    if rewardTime is not None and punishTime is not None and len(rewardTime) == N and len(punishTime) == N:
        rt = pd.Series(rewardTime, dtype='float64')
        pt = pd.Series(punishTime, dtype='float64')
        b["outcomeTime"] = pd.concat([rt, pt], axis=1).mean(axis=1, skipna=True)
    else:
        b["outcomeTime"] = np.nan  # keep column for parity with MATLAB, not used in perf

    # highCPerformance: restrict to goodTrials & max |cDiff|
    if not np.isfinite(b["cDiff"]).any():
        return stim_type, np.nan

    max_abs_cdiff = np.nanmax(np.abs(b["cDiff"].values))
    mask_highC = goodTrials & (np.abs(b["cDiff"]) == max_abs_cdiff)

    if mask_highC.any():
        perf = 100.0 * (b.loc[mask_highC, "feedback"].eq("Rewarded")).mean()
    else:
        perf = np.nan

    return stim_type, float(perf)

def save_png_with_contrast(ref_file, out_png, also_save_16bit=False, p_lo=1, p_hi=99.5):
    img = imread(ref_file)

    # If 3D stack, make a mean projection for display
    if img.ndim == 3:
        img = img.mean(axis=0)

    # Robust min/max from percentiles
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    if hi <= lo:  # fallback if image is nearly constant
        lo, hi = float(img.min()), float(img.max())

    # Normalize to 0..1 then 8-bit
    if hi > lo:
        img_norm = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)

    img8 = (img_norm * 255).astype(np.uint8)

    # write display PNG
    out_png = os.path.splitext(out_png)[0] + ".png"
    imageio.imwrite(out_png, img8)
    #print(f"Saved display PNG with contrast stretch: {out_png}")

    # optional: also save a full-range 16-bit PNG (for analysis)
    if also_save_16bit:
        img16 = np.clip(img, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        out_png16 = os.path.splitext(out_png)[0] + "_16bit.png"
        imageio.imwrite(out_png16, img16)
       # print(f"✅ Saved raw-range 16-bit PNG: {out_png16}")

def parse_pv_env(env_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(env_path)
    root = tree.getroot()

    def find_value(key):
        el = root.find(f".//PVStateValue[@key='{key}']")
        return el.get('value') if el is not None else None

    def find_enum(key):
        el = root.find(f".//PVStateValue[@key='{key}']")
        if el is not None and el.find("EnumIndex") is not None:
            return el.find("EnumIndex").get("value")
        return None

    # direct keys
    objective = find_value('objectiveLens') or find_value('objectiveLensMag')
    optical_zoom = find_value('opticalZoom')
    pixels_per_line = find_value('pixelsPerLine')
    frame_period = find_value('framePeriod')

    # scan orientation
    scan_rot = find_value('scanRotation')      # degrees
    scan_flip = find_enum('scanDirection')     # 0=normal, 1=flipped (often along Y)
    scan_center_y = find_value('scanCenterY')  # µm offset on Y
    scan_size_y = find_value('scanSizeY')      # field height µm

    # calibration FOV under PVObjectiveLensController
    calib = root.find(".//PVObjectiveLensController/PVObjective[@name='_x0031_6X']/Calibration")
    if calib is None:
        calib = root.find(".//PVObjectiveLensController//Calibration")
    fov_w = calib.get('fovWidth') if calib is not None else None
    fov_h = calib.get('fovHeight') if calib is not None else None

    # derived
    px_size_um = float(fov_w)/int(pixels_per_line) if fov_w and pixels_per_line else None
    fs = 1.0/float(frame_period) if frame_period else None

    return {
        "objective": objective,
        "optical_zoom": float(optical_zoom) if optical_zoom else None,
        "fov_width_um": float(fov_w) if fov_w else None,
        "fov_height_um": float(fov_h) if fov_h else None,
        "pixels_per_line": int(pixels_per_line) if pixels_per_line else None,
        "frame_period_s": float(frame_period) if frame_period else None,
        "fs_Hz": fs,
        "pixel_size_um": px_size_um,
        # orientation
        "scan_rotation_deg": float(scan_rot) if scan_rot else 0.0,
        "scan_flip_Y": (scan_flip == "1"),  # True if Y flipped
        "scan_center_Y_um": float(scan_center_y) if scan_center_y else None,
        "scan_size_Y_um": float(scan_size_y) if scan_size_y else None
    }

def add_trial_side_info(behData: pd.DataFrame, tiff_path: str) -> pd.DataFrame:
    """
    Adds trial-level columns for stimulus side, recording side, and choice bias.

    Parameters
    ----------
    behData : pd.DataFrame
        Must have at least columns ['correctResponse', 'choice'].
    tiff_path : str
        Path to the TIFF directory (where a .env file also exists).

    Returns
    -------
    behData : pd.DataFrame
        With new columns:
            - recordingSideStim   : hemisphere being imaged (constant for all trials)
            - biasStim            : stimulus side this trial (left/right)
            - biasChoice          : choice side this trial (left/right)
            - recordingSideChoice : choice relative to imaging hemisphere (ipsi/contra/no_choice)
    """

    # find PrairieView .env
    filenameENV = glob.glob(os.path.join(tiff_path, "*.env"))
    if not filenameENV:
        raise FileNotFoundError(f"No .env file found in {tiff_path}")
    recording_side = which_side_from_env(filenameENV[0])

    if recording_side not in ("left", "right"):
        print(recording_side)
        raise ValueError(f"recording_side must be 'left' or 'right', got {recording_side}")

    behData = behData.copy()

    # constant: hemisphere imaged
    behData["recordingSideStim"] = recording_side

    # trial-dependent stimulus side
    behData["biasStim"] = behData["correctResponse"].str.lower()

    # trial-dependent choice side
    behData["biasChoice"] = behData["choice"].str.lower()

    # choice relative to hemisphere
    def classify_choice(choice: str):
        if pd.isna(choice) or choice == "nogo":
            return "no_choice"
        if choice == recording_side:
            return "ipsi"
        else:
            return "contra"

    behData["recordingSideChoice"] = behData["biasChoice"].apply(classify_choice)

    return behData

def which_side_from_env(env_path: dict) -> str:
    tree = ET.parse(env_path)
    root = tree.getroot()

    y_stage = None
    for val in root.findall(".//PVStateValue[@key='positionCurrent']/SubindexedValues[@index='YAxis']/SubindexedValue"):
        y_stage = float(val.attrib["value"])

    if y_stage is None:
        raise ValueError("Could not find Y stage position in env file")

    return "left" if y_stage < 0 else "right"

def _expand_inf(x):
    # allow "inf"/"-inf" strings in JSON
    if isinstance(x, str):
        if x.lower() == "inf": return float("inf")
        if x.lower() == "-inf": return float("-inf")
    if isinstance(x, list):
        return [_expand_inf(v) for v in x]
    if isinstance(x, dict):
        return {k: _expand_inf(v) for k, v in x.items()}
    return x

def _load_ops_json(ops_json_path: str) -> dict:
    with open(ops_json_path, "r") as f:
        ops = json.load(f)
    ops = _expand_inf(ops)
    # expand ~ in paths if present
    if "fast_disk" in ops and isinstance(ops["fast_disk"], str):
        ops["fast_disk"] = os.path.expanduser(ops["fast_disk"])
    return ops

def _load_ops_yaml(ops_yaml_path: str) -> dict:
    with open(ops_yaml_path, "r") as f:
        ops = yaml.safe_load(f)
    ops = _expand_inf(ops)
    # expand ~ in paths if present
    if "fast_disk" in ops and isinstance(ops["fast_disk"], str):
        ops["fast_disk"] = os.path.expanduser(ops["fast_disk"])
    return ops

def find_tiff_file(tiff_path: str, channel: str = "Ch2") -> str:
    """
    Given a directory (tiff_path), find the first TIFF file inside it.
    Optionally filter by channel string (default: 'Ch2').
    """
    # search for .tif and .tiff files
    pattern1 = os.path.join(tiff_path, f"*{channel}.tif")
    pattern2 = os.path.join(tiff_path, f"*{channel}.tiff")
    files = glob.glob(pattern1) + glob.glob(pattern2)
    
    if not files:
        raise FileNotFoundError(f"No {channel} TIFF files found in {tiff_path}")
    
    files.sort()  # deterministic ordering
    return files[0]   # return the first match

def suite2p_extraction(
    tiff_path: str,
    ops_yaml_path: str,
    imagingDetails: Optional[dict] = None,
    genotype: Optional[str] = None,
    channel_st: str = "Ch3",
):
    """
    Run Suite2p on a single TIFF using ops loaded from JSON, with optional overrides from imagingDetails.

    Parameters
    ----------
    tiff_path : str          # full path to *Ch2.tif(f)
    save_path : str          # output dir (usually .../TwoP/.../suite2p)
    ops_json_path : str      # path to ops_Suite2p.json
    imagingDetails : dict    # e.g. {'fs': 15.0, 'optical_zoom': 1.0, 'pixel_size_um': 0.79, ...}
    fast_disk : str|None     # override ops['fast_disk'] if provided
    base_soma_um : float     # soma diameter in microns to convert to pixels
    """

    tiff_file = find_tiff_file(tiff_path, channel=channel_st)
    print(tiff_file)
    if not os.path.isfile(tiff_file):
        raise FileNotFoundError(f"TIFF not found: {tiff_file} - Check Channel name ({channel_st})")


    # 1) load ops from YAML
    ops = _load_ops_yaml(ops_yaml_path)

    # 2) set output and temp-disk locations
    ops["save_path0"] = tiff_path

    # 3) override from imagingDetails
    diameter_um = 7.9
    if imagingDetails:
        zoom = imagingDetails.get("optical_zoom")          # Hz
        pixelsize_um  = imagingDetails.get("pixel_size_um")
        ops["fs"] = imagingDetails.get("fs_Hz") 
        print(ops["fs"])
        ops['diameter'] = int(np.round(diameter_um / pixelsize_um))


    # dia_px = max(4, int(round(base_soma_um / float(px_um))))
    # print(f"Using diameter {dia_px} pixels (base {base_soma_um} um, px size {px_um:.3f} um)")
    # ops["diameter"] = 11# dia_px

    if genotype== '8s':
        ops["tau"] = 0.5 # 1.26 is good for GCaMP6s
    
    # 4) build db and run
    db = {
        "data_path": tiff_path,
        "tiff_list": [tiff_file],
        "save_path": tiff_path
    }
    print(tiff_path)

    t0 = time.time()
    print("CuPy version:", cupy.__version__)
    print("CUDA device:", cupy.cuda.runtime.getDeviceProperties(0)['name'])

    print("Ops useGPU:", ops["useGPU"])
    res_ops = run_s2p(ops=ops, db=db)
    print(f"✓ Done in {time.time()-t0:.1f}s → {Path(tiff_path, 'plane0').as_posix()}")
    return res_ops

class Tee(object):
    def __init__(self, logfile, mode="w"):
        self.file = open(logfile, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

def update_info(info):
    """
    Update info.recordingList flags (PAQextracted, CSVcreated, suite2Pcreated, dffcreated, etc.)
    without reprocessing data. Only checks if expected files/folders exist.
    
    Parameters
    ----------
    info : object
        Must have `recordingList` DataFrame with at least:
        ['path', 'analysispathname', 'sessionName', 'imagingTiffFileNames'].
    
    Returns
    -------
    info : object
        Updated info object with missing flags filled (0 = missing, 1 = exists).
    """
    # Ensure required columns exist
    for col, dtype in [
    ("PAQextracted", "Int64"),     # nullable integer
    ("PAQdataFound", "Int64"),
    ("CSVcreated", "Int64"),
    ("CSVpath", "string"),         # <-- explicitly string
    ("suite2Pcreated", "Int64"),
    ("dffcreated", "Int64"),
    ("imagingDataExtracted", "Int64"),
    ]:
        if col not in info.recordingList.columns:
            info.recordingList[col] = pd.Series(dtype=dtype)

    # Loop through recordings
    for ind in range(len(info.recordingList)):

        # ---- Step 1: PAQ ----
        if pd.isna(info.recordingList.loc[ind, 'PAQextracted']):
            filenamePAQextracted = glob.glob(os.path.join(info.recordingList.path[ind], "**", "*paq_imaging_frames.txt"), recursive=True)
            info.recordingList.loc[ind, 'PAQextracted'] = 1 if len(filenamePAQextracted) > 0 else 0

        if pd.isna(info.recordingList.loc[ind, 'PAQdataFound']):
            filenamePAQdata = glob.glob(os.path.join(info.recordingList.analysispathname[ind], "paq-data.pkl"))
            info.recordingList.loc[ind, 'PAQdataFound'] = 1 if len(filenamePAQdata) > 0 else 0

        # ---- Step 2: CSV ----
        if pd.isna(info.recordingList.loc[ind, 'CSVcreated']):
            filenameCSV = os.path.join(info.recordingList.analysispathname[ind],
                                       info.recordingList.sessionName[ind] + "_CorrectedeventTimes.csv")
            info.recordingList.loc[ind, 'CSVcreated'] = 1 if os.path.exists(filenameCSV) else 0
            info.recordingList.loc[ind, 'CSVpath'] = filenameCSV

        # ---- Step 3: Suite2p ----
        if pd.isna(info.recordingList.loc[ind, 'suite2Pcreated']):
            tiff_path = info.recordingList.imagingTiffFileNames[ind]
            suite2p_dir = os.path.join(tiff_path, "suite2p")
            info.recordingList.loc[ind, 'suite2Pcreated'] = 1 if os.path.isdir(suite2p_dir) else 0

        # ---- Step 4: imaging.pkl ----
        if pd.isna(info.recordingList.loc[ind, 'dffcreated']):
            filenameDFF = os.path.join(info.recordingList.analysispathname[ind], 'imaging-data.pkl')
            info.recordingList.loc[ind, 'dffcreated'] = 1 if os.path.exists(filenameDFF) else 0
        
        if pd.isna(info.recordingList.loc[ind, 'imagingDataExtracted']):
            filenameimaging = os.path.join(info.recordingList.analysispathname[ind], 'responsive_neurons','imaging-dffTrace.pkl')
            info.recordingList.loc[ind, 'imagingDataExtracted'] = 1 if os.path.exists(filenameimaging) else 0

    return info


def build_trial_types(behData: pd.DataFrame,
                      rewarded: pd.Series,
                      choice: pd.Series,
                      stimSide: pd.Series,
                      stimulus: np.ndarray,
                      recordingSideStim: pd.Series,
                      biasStim: pd.Series,
                      recordingSideChoice: pd.Series,
                      biasChoice: pd.Series) -> tuple[list[str], list[np.ndarray]]:
    """
    Reproduces your exact tTypesName + tTypes logic as arrays of booleans.
    """
    tTypesName = [
        'Rewarded','Unrewarded',
        'Left Choices','Right Choices',
        'Rewarded Left','Rewarded Right',
        'Unrewarded Left','Unrewarded Right',
        'Left','Right',
        '-0.0625','-0.125','-0.25','-0.5','0',
        '0.0625','0.125','0.25','0.5',
        '-0.0625 Rewarded','-0.125 Rewarded','-0.25 Rewarded','-0.5 Rewarded',
        '0.0625 Rewarded','0.125 Rewarded','0.25 Rewarded','0.5 Rewarded','0 Rewarded',
        '-0.0625 Unrewarded','-0.125 Unrewarded','-0.25 Unrewarded','-0.5 Unrewarded',
        '0.0625 Unrewarded','0.125 Unrewarded','0.25 Unrewarded','0.5 Unrewarded','0 Unrewarded',
        '-0.0625 Rewarded Hemi Ipsi','-0.125 Rewarded Hemi Ipsi','-0.25 Rewarded Hemi Ipsi','-0.5 Rewarded Hemi Ipsi',
        '0.0625 Rewarded Hemi Ipsi','0.125 Rewarded Hemi Ipsi','0.25 Rewarded Hemi Ipsi','0.5 Rewarded Hemi Ipsi','0 Rewarded Hemi Ipsi',
        '-0.0625 Rewarded Hemi Contra','-0.125 Rewarded Hemi Contra','-0.25 Rewarded Hemi Contra','-0.5 Rewarded Hemi Contra',
        '0.0625 Rewarded Hemi Contra','0.125 Rewarded Hemi Contra','0.25 Rewarded Hemi Contra','0.5 Rewarded Hemi Contra','0 Rewarded Hemi Contra',
        '-0.0625 Rewarded Bias Ipsi','-0.125 Rewarded Bias Ipsi','-0.25 Rewarded Bias Ipsi','-0.5 Rewarded Bias Ipsi',
        '0.0625 Rewarded Bias Ipsi','0.125 Rewarded Bias Ipsi','0.25 Rewarded Bias Ipsi','0.5 Rewarded Bias Ipsi','0 Rewarded Bias Ipsi',
        '-0.0625 Rewarded Bias Contra','-0.125 Rewarded Bias Contra','-0.25 Rewarded Bias Contra','-0.5 Rewarded Bias Contra',
        '0.0625 Rewarded Bias Contra','0.125 Rewarded Bias Contra','0.25 Rewarded Bias Contra','0.5 Rewarded Bias Contra','0 Rewarded Bias Contra',
        '-0.0625 Rewarded Hemi Ipsi Bias Ipsi','-0.125 Rewarded Hemi Ipsi Bias Ipsi','-0.25 Rewarded Hemi Ipsi Bias Ipsi','-0.5 Rewarded Hemi Ipsi Bias Ipsi',
        '0.0625 Rewarded Hemi Ipsi Bias Ipsi','0.125 Rewarded Hemi Ipsi Bias Ipsi','0.25 Rewarded Hemi Ipsi Bias Ipsi','0.5 Rewarded Hemi Ipsi Bias Ipsi','0 Rewarded Hemi Ipsi Bias Ipsi',
        '-0.0625 Rewarded Hemi Ipsi Bias Contra','-0.125 Rewarded Hemi Ipsi Bias Contra','-0.25 Rewarded Hemi Ipsi Bias Contra','-0.5 Rewarded Hemi Ipsi Bias Contra',
        '0.0625 Rewarded Hemi Ipsi Bias Contra','0.125 Rewarded Hemi Ipsi Bias Contra','0.25 Rewarded Hemi Ipsi Bias Contra','0.5 Rewarded Hemi Ipsi Bias Contra','0 Rewarded Hemi Ipsi Bias Contra',
        '-0.0625 Rewarded Hemi Contra Bias Ipsi','-0.125 Rewarded Hemi Contra Bias Ipsi','-0.25 Rewarded Hemi Contra Bias Ipsi','-0.5 Rewarded Hemi Contra Bias Ipsi',
        '0.0625 Rewarded Hemi Contra Bias Ipsi','0.125 Rewarded Hemi Contra Bias Ipsi','0.25 Rewarded Hemi Contra Bias Ipsi','0.5 Rewarded Hemi Contra Bias Ipsi','0 Rewarded Hemi Contra Bias Ipsi',
        '-0.0625 Rewarded Hemi Contra Bias Contra','-0.125 Rewarded Hemi Contra Bias Contra','-0.25 Rewarded Hemi Contra Bias Contra','-0.5 Rewarded Hemi Contra Bias Contra',
        '0.0625 Rewarded Hemi Contra Bias Contra','0.125 Rewarded Hemi Contra Bias Contra','0.25 Rewarded Hemi Contra Bias Contra','0.5 Rewarded Hemi Contra Bias Contra','0 Rewarded Hemi Contra Bias Contra',
        'Stim Hemi Ipsi','Stim Hemi Contra',
        'Stim Bias Ipsi','Stim Bias Contra',
        'Choice Bias Ipsi','Choice Bias Contra',
        'Choice Hemi Ipsi','Choice Hemi Contra'
    ]

    # Build boolean arrays
    rewarded_bool = rewarded.to_numpy() == True
    choice_l = (choice == 'Left').to_numpy()
    choice_r = (choice == 'Right').to_numpy()
    stim_l   = (stimSide == 'Left').to_numpy()
    stim_r   = (stimSide == 'Right').to_numpy()

    stim_eq  = (stimulus == 0)
    stim_vals = {
        '-0.0625': (stimulus == -0.0625),
        '-0.125':  (stimulus == -0.125),
        '-0.25':   (stimulus == -0.25),
        '-0.5':    (stimulus == -0.5),
        '0.0625':  (stimulus == 0.0625),
        '0.125':   (stimulus == 0.125),
        '0.25':    (stimulus == 0.25),
        '0.5':     (stimulus == 0.5),
        '0':       (stimulus == 0)
    }

    hemi_ipsi  = (recordingSideStim == 'ipsi').to_numpy()
    hemi_contra= (recordingSideStim == 'contra').to_numpy()
    bias_is    = (biasStim == 'bias').to_numpy()
    bias_nb    = (biasStim == 'no bias').to_numpy()
    ch_bias_i  = (biasChoice == 'bias').to_numpy()
    ch_bias_c  = (biasChoice == 'no bias').to_numpy()
    ch_hemi_i  = (recordingSideChoice == 'ipsi').to_numpy()
    ch_hemi_c  = (recordingSideChoice == 'contra').to_numpy()

    tTypes: list[np.ndarray] = [
        rewarded_bool, ~rewarded_bool,
        choice_l, choice_r,
        rewarded_bool & choice_l, rewarded_bool & choice_r,
        (~rewarded_bool) & choice_l, (~rewarded_bool) & choice_r,
        stim_l, stim_r,
        stim_vals['-0.0625'], stim_vals['-0.125'], stim_vals['-0.25'], stim_vals['-0.5'], stim_vals['0'],
        stim_vals['0.0625'],  stim_vals['0.125'],  stim_vals['0.25'],  stim_vals['0.5'],
        rewarded_bool & stim_vals['-0.0625'],
        rewarded_bool & stim_vals['-0.125'],
        rewarded_bool & stim_vals['-0.25'],
        rewarded_bool & stim_vals['-0.5'],
        rewarded_bool & stim_vals['0.0625'],
        rewarded_bool & stim_vals['0.125'],
        rewarded_bool & stim_vals['0.25'],
        rewarded_bool & stim_vals['0.5'],
        rewarded_bool & stim_vals['0'],
        (~rewarded_bool) & stim_vals['-0.0625'],
        (~rewarded_bool) & stim_vals['-0.125'],
        (~rewarded_bool) & stim_vals['-0.25'],
        (~rewarded_bool) & stim_vals['-0.5'],
        (~rewarded_bool) & stim_vals['0.0625'],
        (~rewarded_bool) & stim_vals['0.125'],
        (~rewarded_bool) & stim_vals['0.25'],
        (~rewarded_bool) & stim_vals['0.5'],
        (~rewarded_bool) & stim_vals['0'],

        hemi_ipsi  & stim_vals['-0.0625'] & rewarded_bool,
        hemi_ipsi  & stim_vals['-0.125']  & rewarded_bool,
        hemi_ipsi  & stim_vals['-0.25']   & rewarded_bool,
        hemi_ipsi  & stim_vals['-0.5']    & rewarded_bool,
        hemi_ipsi  & stim_vals['0.0625']  & rewarded_bool,
        hemi_ipsi  & stim_vals['0.125']   & rewarded_bool,
        hemi_ipsi  & stim_vals['0.25']    & rewarded_bool,
        hemi_ipsi  & stim_vals['0.5']     & rewarded_bool,
        hemi_ipsi  & stim_vals['0']       & rewarded_bool,

        hemi_contra & stim_vals['-0.0625'] & rewarded_bool,
        hemi_contra & stim_vals['-0.125']  & rewarded_bool,
        hemi_contra & stim_vals['-0.25']   & rewarded_bool,
        hemi_contra & stim_vals['-0.5']    & rewarded_bool,
        hemi_contra & stim_vals['0.0625']  & rewarded_bool,
        hemi_contra & stim_vals['0.125']   & rewarded_bool,
        hemi_contra & stim_vals['0.25']    & rewarded_bool,
        hemi_contra & stim_vals['0.5']     & rewarded_bool,
        hemi_contra & stim_vals['0']       & rewarded_bool,

        bias_is & stim_vals['-0.0625'] & rewarded_bool,
        bias_is & stim_vals['-0.125']  & rewarded_bool,
        bias_is & stim_vals['-0.25']   & rewarded_bool,
        bias_is & stim_vals['-0.5']    & rewarded_bool,
        bias_is & stim_vals['0.0625']  & rewarded_bool,
        bias_is & stim_vals['0.125']   & rewarded_bool,
        bias_is & stim_vals['0.25']    & rewarded_bool,
        bias_is & stim_vals['0.5']     & rewarded_bool,
        bias_is & stim_vals['0']       & rewarded_bool,

        bias_nb & stim_vals['-0.0625'] & rewarded_bool,
        bias_nb & stim_vals['-0.125']  & rewarded_bool,
        bias_nb & stim_vals['-0.25']   & rewarded_bool,
        bias_nb & stim_vals['-0.5']    & rewarded_bool,
        bias_nb & stim_vals['0.0625']  & rewarded_bool,
        bias_nb & stim_vals['0.125']   & rewarded_bool,
        bias_nb & stim_vals['0.25']    & rewarded_bool,
        bias_nb & stim_vals['0.5']     & rewarded_bool,
        bias_nb & stim_vals['0']       & rewarded_bool,

        hemi_ipsi   & stim_vals['-0.0625'] & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['-0.125']  & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['-0.25']   & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['-0.5']    & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['0.0625']  & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['0.125']   & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['0.25']    & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['0.5']     & rewarded_bool & bias_is,
        hemi_ipsi   & stim_vals['0']       & rewarded_bool & bias_is,

        hemi_ipsi   & stim_vals['-0.0625'] & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['-0.125']  & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['-0.25']   & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['-0.5']    & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['0.0625']  & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['0.125']   & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['0.25']    & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['0.5']     & rewarded_bool & bias_nb,
        hemi_ipsi   & stim_vals['0']       & rewarded_bool & bias_nb,

        hemi_contra & stim_vals['-0.0625'] & rewarded_bool & bias_is,
        hemi_contra & stim_vals['-0.125']  & rewarded_bool & bias_is,
        hemi_contra & stim_vals['-0.25']   & rewarded_bool & bias_is,
        hemi_contra & stim_vals['-0.5']    & rewarded_bool & bias_is,
        hemi_contra & stim_vals['0.0625']  & rewarded_bool & bias_is,
        hemi_contra & stim_vals['0.125']   & rewarded_bool & bias_is,
        hemi_contra & stim_vals['0.25']    & rewarded_bool & bias_is,
        hemi_contra & stim_vals['0.5']     & rewarded_bool & bias_is,
        hemi_contra & stim_vals['0']       & rewarded_bool & bias_is,

        hemi_contra & stim_vals['-0.0625'] & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['-0.125']  & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['-0.25']   & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['-0.5']    & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['0.0625']  & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['0.125']   & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['0.25']    & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['0.5']     & rewarded_bool & bias_nb,
        hemi_contra & stim_vals['0']       & rewarded_bool & bias_nb,

        hemi_ipsi, hemi_contra,
        bias_is,  bias_nb,
        ch_bias_i, ch_bias_c,
        ch_hemi_i, ch_hemi_c
    ]

    return tTypesName, tTypes







