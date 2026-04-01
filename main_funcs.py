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
import LakLabAnalysis.Utility.utils_funcs as lutils
import psychofit.psychofit as psy
from scipy.stats import linregress
from collections import defaultdict


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
        self.analysisPath =  'Y:\\sideBiasLateralisation\\analysis' # os.path.join(self.rootPath, 'analysis')
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
                        session_type, stim_type, performance = get_stim_and_performance(block)
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
                        'stimType': stim_type,
                        'sessionType': session_type

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

    # # session_type
    n_both_zero = ((b["contrastRight"] == 0) & (b["contrastLeft"] == 0)).sum()
    n_either_zero = ((b["contrastRight"] == 0) | (b["contrastLeft"] == 0)).sum()
    n_both_nonzero = ((b["contrastRight"] != 0) & (b["contrastLeft"] != 0)).sum()
          
    if (len(b['cDiff'].unique()) <= 5)  and (n_either_zero > 0):
        session_type = "LimitedContrasts" # min 2 contrasts per side + zero-contrast trials
    elif  (len(b['cDiff'].unique()) >5 ) and (n_either_zero > 0) and (n_both_nonzero == 0):
        session_type = "SingleFullContrasts"  # SINGLE STIM: many contrasts per side + zero-contrast trials
    elif  (len(b['cDiff'].unique()) >5 ) and (n_either_zero > 0) and (n_both_nonzero >10):
        session_type = "TwoFullContrasts" 
    else:
        session_type = "Training"

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

    return session_type, stim_type, float(perf)

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
    def get_axis_value(axis):
        el = root.find(f".//PVStateValue[@key='positionCurrent']/SubindexedValues[@index='{axis}']/SubindexedValue")
        return float(el.get('value')) if el is not None else None

    # direct keys
    objective = find_value('objectiveLens') or find_value('objectiveLensMag')
    optical_zoom = find_value('opticalZoom')
    pixels_per_line = find_value('pixelsPerLine')
    frame_period = find_value('framePeriod')

    # scan orientation
    scan_rot = find_value('scanRotation')      # degrees
    scan_flip = find_enum('scanDirection')     # 0=normal, 1=flipped (often along Y)
    yAxis = get_axis_value('YAxis')   # stage Y position, used to infer recording side in which_side_from_env
    xAxis = get_axis_value('XAxis')   # stage X position, not used currently but could be informative
    zAxis = get_axis_value('ZAxis')   # stage Z position, not used currently but could be informative

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
        "YAxis": float(yAxis) if yAxis else None,
        "XAxis": float(xAxis) if xAxis else None,
        "ZAxis": float(zAxis) if zAxis else None,
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
    ops["fs"] = imagingDetails.get("fs_Hz")

    # 3) override from imagingDetails
    # diameter_um = 
    # if imagingDetails:
    #     zoom = imagingDetails.get("optical_zoom")          # Hz
    #     pixelsize_um  = imagingDetails.get("pixel_size_um")
    #     ops["fs"] = imagingDetails.get("fs_Hz") 
    #     print(ops["fs"])
    #     ops['diameter'] = int(np.round(diameter_um / pixelsize_um))


    # dia_px = max(4, int(round(base_soma_um / float(px_um))))
    #print(f"Using diameter {dia_px} pixels (base {base_soma_um} um, px size {px_um:.3f} um)")
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
                      tiff_path,
                      idx_keep = None):

    rewarded    =  behData['rewardTime'].notna() #
    choice      = behData['choice']
    stimSide    = behData['correctResponse']
    stimulus = np.where(behData['contrastLeft'] != 0, -behData['contrastLeft'], behData['contrastRight'])
    behData_aug = add_trial_side_info(behData, tiff_path)
    
    recordingSideStim = behData_aug['recordingSideStim']
    biasStim = behData_aug['biasStim']
    recordingSideChoice = behData_aug['recordingSideChoice']
    biasChoice = behData_aug['biasChoice']

    tTypesName = [
        'All Trials',
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
        rewarded_bool | ~rewarded_bool,
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

    # Exclude trials that do not fit criteria!  
    keep_mask = behData.index.isin(idx_keep) if idx_keep is not None else np.ones(len(behData), dtype=bool)
    tTypes = [t & keep_mask for t in tTypes]

    return tTypesName, tTypes
    
def get_imagingExtractionParams():
    fRate_imaging = 30
    pre_stim_sec = 2 # sec
    total_time = 8
    analysisWindow_time = 1 # sec
    analysisWindow_min = 0.3 # sec

    return fRate_imaging, pre_stim_sec, total_time, analysisWindow_time, analysisWindow_min

def get_startFramePerTrial(analysispathname, sessionName,rawpathname, event_type, exclude = None, behOut = None):
    if exclude is None:
        beh_df, idx_keep, _ = clean_session_behavior(analysispathname, sessionName,rawpathname)
    else:
        _,idx_keep, beh_df = clean_session_behavior(analysispathname, sessionName,rawpathname)
    fRate_imaging, pre_stim_sec, total_time, analysisWindow_time, analysisWindow_min = get_imagingExtractionParams()

    n_trials = len(beh_df)

    if event_type == 'stimulus':
        # assuming these are in seconds
        analysisWindow = beh_df['choiceStartTime'].to_numpy() - beh_df['stimulusOnsetTime'].to_numpy()
        analysisWindow = np.where(
                np.isnan(analysisWindow) | (analysisWindow <= 0),
                analysisWindow_min,
                analysisWindow)
        start_frame = np.full(n_trials, int(pre_stim_sec * fRate_imaging), dtype=int)
        end_frame = start_frame + (analysisWindow * fRate_imaging).astype(int)

    elif event_type == 'reward':
        analysisWindow = np.full(n_trials, analysisWindow_time)
        start_frame = np.full(n_trials, int(pre_stim_sec * fRate_imaging), dtype=int)
        end_frame = start_frame + (analysisWindow * fRate_imaging).astype(int)

    else:
        raise ValueError("event_type must be 'stimulus' or 'reward'")
    if behOut is None:
        return start_frame, end_frame
    else:
        return start_frame, end_frame, idx_keep, beh_df

def clean_session_behavior(analysispathname,sessionName, rawpathname):
    ' Criterias to exclude trials'
    ' EXCLUDE 1:  trials with response time greater than 2 seconds'
    ' EXCLUDE 2:  trials where choice start time is before go cue time'
    csv_path = os.path.join(analysispathname, f"{sessionName}_CorrectedeventTimes.csv")
    beh_df = pd.read_csv(csv_path)

    ### EXCLUDE trials with missing imaging frames (e.g. due to PAQ issues)
    filenameTXT = os.path.join(rawpathname,'twoP') +'\*_imaging_frames.txt'
    filenameTXT= [f for f in glob.glob(filenameTXT)]    
    frame_clock = pd.read_csv(filenameTXT[0],  header= None)

    visTimes    = beh_df['stimulusOnsetTime'] + beh_df['trialOffsets']
    _, excludedIndex = utils.stim_start_frame_Dual2Psetup(frame_clock, visTimes, excIndOutput = True)
    beh_df = beh_df.drop(index=excludedIndex)

    # EXCLUSION CRITERIA 1: Calculate response time > 2 seconds
    beh_df['responseTime'] = beh_df['choiceCompleteTime'] - beh_df['stimulusOnsetTime']
    mask_longRT = (beh_df['choiceCompleteTime'] - beh_df['stimulusOnsetTime']) > 2# EXCLUDE 1:  trials with response time greater than 2 seconds
    idx_longRT = beh_df.index[mask_longRT]
    # EXCLUSION CRITERIA 2: Calculate choice start time before go cue time
    mask_preGoCue = (beh_df['choiceStartTime']  - beh_df['goCueTime'] ) < -0.2 # EXCLUDE 2:  trials where choice start time is before go cue time
    idx_preGoCue = beh_df.index[mask_preGoCue]
    combined_idx = idx_longRT.union(idx_preGoCue)
    idx_keep = beh_df.index.difference(combined_idx)
    # Get cleaned dataframe
    beh_df_clean = beh_df.drop(index=combined_idx)
    #print(f"{len(idx_keep)} trials: {len(idx_longRT)} long RT, {len(idx_preGoCue)} with pre-go-cue from {len(beh_df)}.")

    return beh_df, idx_keep, beh_df_clean

def construct_info_path(info, blockName, folder=None, file=None, load='path'):
    """ ST 05/2025: 
    """
    assert isinstance(folder, str), 'Argument: folder must be strings'
    assert blockName in info.recordingList.blockName.values, f"{blockName} not found in recordingList"
    ind = lutils.dfIndFromValue(blockName, info.recordingList.blockName)[0]
    animal = info.recordingList.animalID[ind]
    
    if folder in info.recordingList.columns:
        folder_path = info.recordingList[folder][ind]
    # Case 2: suite2p_path
    elif folder=='s2p' or folder=='suite2p':
        folder_path = os.path.join(info.suite2pOutputPath, animal, blockName,'plane0')
    # Case 3: cellpose path
    elif folder=='cellpose':
        folder_path = os.path.join(info.recordingList.analysisPath[ind], 'cellpose')
    # Case 4: cellpose/plane* path (with s2p outputs)
    elif folder=='cellpose_plane':
        folder_path = os.path.join(info.recordingList.analysisPath[ind], 'cellpose','plane0')

    if file is not None:
        possible_files = [f for f in glob.glob(folder_path + f'/*{file}*')]
        if len(possible_files)==1:
            final_path = possible_files[0]
        elif len(possible_files)==0: 
            print(f"No file was found using {folder} > {file} for {blockName}, errors possible.")
            final_path = os.path.join(folder_path, file)
        elif len(possible_files)>1: 
            final_path = possible_files[0]
            print(f'Multiple files found, returning the first: {final_path}')
    else: final_path = folder_path

    if load=='path': return final_path.replace('\\', '/')
    elif load == 'item': 
        if 'ops' in file or 'seg.npy' in file:
            return np.load(final_path, allow_pickle=True).item()
        elif 'stat' in file or file.endswith('.npy'):
            return np.load(final_path, allow_pickle=True)
        else: 
            print(f"Item contingency has not been coded; beware errors")

def compute_response_time(df: pd.DataFrame) -> pd.Series:
    """
    Response time definition:
      RT = choiceCompleteTime - choiceStartTime
    Returns a float series (seconds).
    """
    rt = df["choiceCompleteTime"] - df["choiceStartTime"]
    return rt.astype(float)

def find_motivation_cutoff_trial(df: pd.DataFrame, rt_col: str = "responseTime", factor: float = 4.0):
    """
    Find first trial index where RT > factor * mean(previous RTs).
    Returns:
      end_trial_number (int or None),
      end_trial_row_index (int or None)

    'previous average' interpreted as cumulative mean of all *prior* valid RT trials.
    """
    rt = df[rt_col].to_numpy(dtype=float)
    trial_nums = df["trialNumber"].to_numpy()

    prev_rts = []
    for i in range(len(rt)):
        cur = rt[i]
        if np.isnan(cur):
            continue

        if len(prev_rts) >= 1:
            prev_mean = np.nanmean(prev_rts)
            # guard against weird prev_mean = 0
            if prev_mean > 0 and cur > factor * prev_mean:
                return int(trial_nums[i]), int(i)

        prev_rts.append(cur)

    return None, None

def compute_zero_stim_bias(df: pd.DataFrame):
    """
    Compute P(Right | zero contrast).

    Zero contrast trials:
        contrastLeft == 0 AND contrastRight == 0

    Returns:
        p_right (float or NaN),
        n_zero_trials (int),
        n_right (int)
    """
    # select zero-contrast trials
    mask = (
        (df["contrastLeft"] == 0) &
        (df["contrastRight"] == 0)
    )

    d = df.loc[mask]

    n_zero = len(d)
    if n_zero == 0:
        return np.nan, 0, 0

    n_right = int((d["choice"] == "Right").sum())
    bias = n_right / n_zero
    bias = bias-0.5

    return bias

def get_mean_dff_by_contrast_diff_df(
    recordingList,
    event_type="stimulus",
    time_window=(0.1, 0.8),
    baseline_window=(-0.2, 0.0),
    subfolder="responsive_neurons",
    use_zscored=True,
    allowed_cdiffs=(-0.5, -0.25, -0.125, 0.125, 0.25, 0.5),
    biasType = 'bias'):

    fRate = 30
    pre_stim_sec = 2
    total_time = 8
    n_frames = int(total_time * fRate)
    time_axis = np.linspace(-pre_stim_sec, total_time - pre_stim_sec, n_frames)

    start_frame = int((time_window[0] + pre_stim_sec) * fRate)
    end_frame   = int((time_window[1] + pre_stim_sec) * fRate)

    b0 = int(np.argmin(np.abs(time_axis - baseline_window[0])))
    b1 = int(np.argmin(np.abs(time_axis - baseline_window[1])))
    if b1 == b0:
        b1 += 1

    rows = []

    for ind in range(len(recordingList.recordingDate)):
        if recordingList.imagingDataExtracted[ind] != 1:
            continue

        animal_id = recordingList.animalID[ind]
        session = recordingList.sessionName[ind]
        recordingDate = recordingList.recordingDate[ind]
        pathname = recordingList.analysispathname[ind]
        subfolder_path = os.path.join(pathname, subfolder)
        bias = recordingList[biasType][ind]
        performance = recordingList.performance[ind]

        pkl_name = "imaging-dffTrace_mean_zscored.pkl" if use_zscored else "imaging-dffTrace_mean.pkl"
        pkl_path = os.path.join(subfolder_path, pkl_name)
        if not os.path.exists(pkl_path):
            continue

        try:
            with open(pkl_path, "rb") as f:
                dff_reward, dff_stim, dff_choice = pickle.load(f)

            if event_type.lower() == "stimulus":
                dff_dict = dff_stim
            elif event_type.lower() == "choice":
                dff_dict = dff_choice
            elif event_type.lower() == "reward":
                dff_dict = dff_reward
            else:
                raise ValueError("event_type must be 'stimulus', 'choice', or 'reward'")

            for key, arr in dff_dict.items():
                if arr is None:
                    continue
                if arr.shape[1] <= end_frame:
                    continue

                # Correct parsing for your structure:
                # key example: "-0.25 Rewarded" -> cDiff = -0.25
                try:
                    cDiff = float(str(key).split()[0])
                except Exception:
                    continue

                # keep only requested cDiffs
                if allowed_cdiffs is not None:
                    if not np.isclose(cDiff, allowed_cdiffs).any():
                        continue

                # baseline subtraction (using mean trace over cells)
                mean_trace = np.nanmean(arr, axis=0)  # (time,)
                baseline = np.nanmean(mean_trace[b0:b1])
                arr_bs = arr - baseline

                mean_dff = float(np.nanmean(arr_bs[:, start_frame:end_frame]))

                rows.append({
                    "animal": animal_id,
                    "session": session,
                    "recordingDate": recordingDate,
                    "cDiff": cDiff,
                    "mean_dff": mean_dff,
                    "bias": bias,
                    "performance": performance,
                })

        except Exception as e:
            print(f"Error in session {session}: {e}")
            continue

    return pd.DataFrame(rows)

def calculateContrastAverage(df):

    EXCLUDE = np.array([-0.0625, 0.0625], dtype=float)
    contrast_stats = {}
    good = (df['repeatNumber'] == 1) & (df['choice'] != 'NoGo')
    c_diff = df['contrastRight'] - df['contrastLeft']

    for contrast in sorted(c_diff.unique()):
        if np.any(np.isclose(float(contrast), EXCLUDE, atol=1e-12)):
            continue
        trials = good & np.isclose(c_diff, contrast)
        n_trials = trials.sum()

        if n_trials > 0:
            p_right = (df.loc[trials, 'choice'] == 'Right').mean()
            contrast_stats[float(contrast)] = {
                "p_right": float(p_right),
                "n_trials": int(n_trials) }

    return contrast_stats

def fit_psy_AllContrasts( df: pd.DataFrame,
    use_percent: bool = True,
    params: dict | None = None ):
    if params is None:
        params = {
            'parmin': np.array([-40., 5., 0., 0.]),
            'parmax': np.array([40., 30., 0.15, 0.15]),
            'parstart': np.array([0., 10., 0.02, 0.02]),
            'nfits': 30 }
        
    contrast_stats = calculateContrastAverage(df)

    stim_list = []
    n_trials_list = []
    p_right_list = []

    for contrast, stats in contrast_stats.items():
        # ---- exclude 0 contrast ----
        if np.isclose(float(contrast), 0.0):
            continue
        stim_val = float(contrast) * 100.0 if use_percent else float(contrast)

        stim_list.append(stim_val)
        n_trials_list.append(int(stats["n_trials"]))
        p_right_list.append(float(stats["p_right"]))

    g = pd.DataFrame({
        "stim": stim_list,
        "n_trials": n_trials_list,
        "p_right": p_right_list }).sort_values("stim").reset_index(drop=True)

    if len(g) < 3:
        raise ValueError(f"Need at least 3 stimulus levels to fit; got {len(g)}.")

    # build data matrix expected by psy library
    xx = g["stim"].to_numpy(dtype=float)
    nn = g["n_trials"].to_numpy(dtype=float)
    pp = g["p_right"].to_numpy(dtype=float)

    data = np.vstack((xx, nn, pp))
    pars, L = psy.mle_fit_psycho(data, "erf_psycho_2gammas", **params)

    return pars, L, data, g

def computeSideAssociation(df: pd.DataFrame):
    """ Implements Liebana et al. (2025) right/left association metric
    using mFun.calculateContrastAverage(df). """

    contrast_stats = calculateContrastAverage(df)

    contrasts = []
    probs = []

    for contrast, stats in contrast_stats.items():
        contrasts.append(float(contrast))
        probs.append(float(stats["p_right"]))

    contrasts = np.array(contrasts)
    probs = np.array(probs)

    # sort contrasts
    order = np.argsort(contrasts)
    contrasts = contrasts[order]
    probs = probs[order]

    # P(Right | 0)
    zero_mask = np.isclose(contrasts, 0)
    p0 = probs[zero_mask][0] if np.any(zero_mask) else np.nan

    # P(Right | Right stim)
    pos_mask = contrasts > 0
    pR = np.mean(probs[pos_mask]) if np.any(pos_mask) else np.nan

    # P(Right | Left stim)
    neg_mask = contrasts < 0
    pL = np.mean(probs[neg_mask]) if np.any(neg_mask) else np.nan

    slope_R = (pR - p0) if not np.isnan(pR) and not np.isnan(p0) else np.nan
    slope_L = (pL - p0) if not np.isnan(pL) and not np.isnan(p0) else np.nan

    delta_slope = np.abs(slope_R) - np.abs(slope_L)

    if np.isnan(delta_slope):
        label = "Undefined"
    elif delta_slope > 0:
        label = "Right-associating"
    elif delta_slope < 0:
        label = "Left-associating"
    else:
        label = "Balanced"

    return {
        "p0": p0,
        "pR": pR,
        "pL": pL,
        "slope_R": slope_R,
        "slope_L": slope_L,
        "R_minus_L_slope": delta_slope,
        "stimSensitivity": (slope_R + slope_L)/2,
        "stimSensitivityBias": (slope_R - slope_L)/(slope_R + slope_L),
        "label": label
    }

def computeStimulusBias(df: pd.DataFrame):
    """ Compute stimulus bias based on slope difference between
    right and left contrasts. """
    contrast_stats = calculateContrastAverage(df)

    contrasts = []
    probs = []
    for contrast, stats in contrast_stats.items():
        contrasts.append(float(contrast))
        probs.append(float(stats["p_right"]))

    contrasts = np.array(contrasts)
    probs = np.array(probs)
    # sort contrasts
    order = np.argsort(contrasts)
    contrasts = contrasts[order]
    probs = probs[order]
    # masks
    left_mask = contrasts < 0
    right_mask = contrasts > 0

    slopeL = np.nan
    slopeR = np.nan

    if np.sum(left_mask) >= 2:
        slopeL = linregress(contrasts[left_mask], probs[left_mask]).slope

    if np.sum(right_mask) >= 2:
        slopeR = linregress(contrasts[right_mask], probs[right_mask]).slope

    stimulusBias = slopeR - slopeL

    return slopeL, slopeR, stimulusBias

def compute_time_resolved_lateralisation(
    recordingList,
    event_type="stimulus",              # "stimulus" | "choice" | "reward"
    subfolder="responsive_neurons",
    use_zscored=True,
    # which signed contrast-differences to treat as Right-strong vs Left-strong
    pos_cdiffs=(0.125, 0.25, 0.5),
    neg_cdiffs=(-0.125, -0.25, -0.5),
    # imaging time axis definition
    fRate=30,
    pre_sec=2.0,
    total_sec=8.0,
    # baseline subtraction
    baseline_window=(-0.2, 0.0),        # seconds in the same axis (relative to event)
    # optionally restrict to rewarded trials if your dict is “... Rewarded”
    # (we don’t filter by reward here; we use whatever keys exist)
):
    """
    Computes time-resolved lateralisation per session:

        L(t) = mean_trace_pos(t) - mean_trace_neg(t)

    Where mean_trace_pos is the average (across selected +cDiff conditions)
    and mean_trace_neg is the average (across selected -cDiff conditions).

    Assumes dffTrace_mean dict keys start with signed numeric cDiff, e.g.:
        "-0.25 Rewarded", "0.5 Rewarded", etc.
    """

    # build time axis
    n_frames = int(total_sec * fRate)
    time_axis = np.linspace(-pre_sec, total_sec - pre_sec, n_frames)

    # baseline indices
    b0 = int(np.argmin(np.abs(time_axis - baseline_window[0])))
    b1 = int(np.argmin(np.abs(time_axis - baseline_window[1])))
    if b1 == b0:
        b1 += 1

    pkl_name = "imaging-dffTrace_mean_zscored.pkl" if use_zscored else "imaging-dffTrace_mean.pkl"

    rows = []

    for ind in range(len(recordingList.recordingDate)):

        if getattr(recordingList, "imagingDataExtracted", None) is not None:
            if recordingList.imagingDataExtracted[ind] != 1:
                continue

        animal = recordingList.animalID[ind]
        session = recordingList.sessionName[ind]
        recDate = recordingList.recordingDate[ind]
        pathname = recordingList.analysispathname[ind]
        pkl_path = os.path.join(pathname, subfolder, pkl_name)

        if not os.path.exists(pkl_path):
            continue

        try:
            with open(pkl_path, "rb") as f:
                dff_reward, dff_stim, dff_choice = pickle.load(f)

            if event_type.lower() == "stimulus":
                dff_dict = dff_stim
            elif event_type.lower() == "choice":
                dff_dict = dff_choice
            elif event_type.lower() == "reward":
                dff_dict = dff_reward
            else:
                raise ValueError("event_type must be 'stimulus', 'choice', or 'reward'")

            pos_traces = []
            neg_traces = []

            # collect mean traces for each requested signed cDiff
            for key, arr in dff_dict.items():
                if arr is None:
                    continue
                # arr is typically (nCells, nFrames)
                if arr.shape[1] != n_frames:
                    # if your stored traces have different length, skip gracefully
                    continue

                # parse signed cDiff from key
                try:
                    cDiff = float(str(key).split()[0])
                except Exception:
                    continue

                mean_trace = np.nanmean(arr, axis=0)  # average across cells

                # baseline subtract using mean_trace baseline
                baseline = np.nanmean(mean_trace[b0:b1])
                mean_trace_bs = mean_trace - baseline

                # assign
                if np.isclose(cDiff, pos_cdiffs).any():
                    pos_traces.append(mean_trace_bs)
                elif np.isclose(cDiff, neg_cdiffs).any():
                    neg_traces.append(mean_trace_bs)

            # need at least 1 trace on each side
            if (len(pos_traces) == 0) or (len(neg_traces) == 0):
                continue

            pos_mean = np.nanmean(np.vstack(pos_traces), axis=0)
            neg_mean = np.nanmean(np.vstack(neg_traces), axis=0)

            lat_trace = pos_mean - neg_mean  # Right-strong minus Left-strong

            rows.append({
                "animalID": animal,
                "sessionName": session,
                "recordingDate": recDate,
                "event_type": event_type,
                "n_pos_conditions": len(pos_traces),
                "n_neg_conditions": len(neg_traces),
                "pos_mean_trace": pos_mean,     # numpy array
                "neg_mean_trace": neg_mean,     # numpy array
                "lat_trace": lat_trace,         # numpy array
            })

        except Exception as e:
            print(f"[WARN] failed session {session}: {e}")
            continue

    df_lat = pd.DataFrame(rows)
    return df_lat, time_axis

def plot_time_resolved_lateralisation(df_lat: pd.DataFrame, time_axis, title=None, xlim=None):
    """
    Plots mean ± SEM of lat_trace across sessions in df_lat.
    """
    if df_lat.empty:
        print("No sessions to plot.")
        return

    mat = np.vstack(df_lat["lat_trace"].to_numpy())  # nSessions x nFrames
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat[:, 0])))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_axis, mean, linewidth=2)
    ax.fill_between(time_axis, mean - sem, mean + sem, alpha=0.2)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lateralisation (Right-strong − Left-strong)")
    ax.set_title(title or "Time-resolved lateralisation")
    if xlim is not None:
        ax.set_xlim(xlim)

def extract_prechoice_lat(df_lat, time_axis, win=(0.2, 0.8)):
    mask = (time_axis >= win[0]) & (time_axis <= win[1])
    df = df_lat.copy()
    return df.assign(
        lat_prechoice=df["lat_trace"].apply(lambda x: np.nanmean(x[mask]))
    )
