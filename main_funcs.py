# Functions used more globally
import platform
import os
import pandas as pd
import numpy as np
import tifffile
#import plot_funcs as pfun
import utils_funcs as utils
from datetime import datetime
import re
import pickle
import glob
from statsmodels.stats.multitest import fdrcorrection
#if os.environ['CONDA_DEFAULT_ENV'] == 'decision-making-ev':
print('Env: ' + os.environ['CONDA_DEFAULT_ENV'])


class analysis:
    
    def __init__(self):
        
        if platform.node() == 'macOS-12.6-arm64-arm-64bit':
            # for Windows - Huriye PC
            print("Computer: Huriye MAC")
            self.suite2pOutputPath = 'N/A' 
            self.recordingListPath = "/Users/Huriye/Documents/Code/decision-making-ev/"
            self.rawPath           = 'N/A' # this folder is only avaiable with PC  
            self.rootPath          = "/Users/Huriye/Documents/Code/decision-making-ev/"
        elif  platform.node() == 'WIN-AL015':        
            print("Computer: Orsi Windows")   # for Windows - Orsi PC
            self.suite2pOutputPath = 'X:\\Data\\decision-making-ev\\suite2p-output\\' 
            self.recordingListPath = "C:\\Users\\Lak Lab\\Documents\\Code\\decision-making-ev\\"
            self.rawPath           = 'W:\\' # this folder is only avaiable with PC  
            self.rootPath          = "C:\\Users\\Lak Lab\\Documents\\Code\\decision-making-ev\\"
        elif platform.node() == 'WIN-AMP016':
            print("Computer: Huriye Windows")
            # for Windows - Huriye PC
            self.suite2pOutputPath = 'X:\\Data\\decision-making-ev\\suite2p_output\\' 
            self.recordingListPath = "C:\\Users\\Huriye\\Documents\\code\\decision-making-ev\\"
            self.rawPath           = 'Z:\\' # "D:\\decision-making-ev\\Data\\" #
            self.rootPath          = "C:\\Users\\Huriye\\Documents\\code\\decision-making-ev\\"
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
        #self.DLCconfigPath = os.path.join(self.rootPath, 'pupilExtraction', 'Updated 5 Dot Training Model-Eren CAN-2021-11-21')
        #self.DLCconfigPath = self.DLCconfigPath + '\\config.yaml'
        
        # Create the list 
        info = pd.DataFrame()
        # Recursively search for files ending with 'Block.mat' in all subfolders
        animalList = ['MBL015', 'MBL014']
        badRecordingSessions = ['2023-07-07_1_OFZ008_Block.mat', '2023-07-07_3_OFZ008_Block.mat', # Not good ROIs
                                '2023-07-11_1_OFZ008_Block.mat', '2023-07-13_2_OFZ008_Block.mat', # Not good ROIs
                                '2023-06-13_1_OFZ009_Block.mat', '2023-07-03_1_OFZ009_Block.mat',
                                '2023-06-20_1_OFZ010_Block.mat', '2023-06-23_1_OFZ010_Block.mat',
                                '2023-05-30_1_OFZ010_Block.mat', '2023-06-15_1_OFZ010_Block.mat',
                                '2023-05-31_1_OFZ011_Block.mat','2023-06-20_1_OFZ011_Block.mat',
                                '2023-07_07_1_OFZ011_Block.mat', '2023-07-25_1_OFZ011_Block.mat',
                                '2023-07-16_2_OFZ011_Block.mat', '2023-07-16_3_OFZ011_Block.mat',
                                '2023-07-16_1_OFZ011_Block.mat',
                                '2023-07-15_1_OFZ011_Block.mat', '2023-07-17_1_OFZ011_Block.mat',# w/imaging - corruption in transfer
                                '2023-06-11_2_OFZ011_Block.mat', '2023-06-15_1_OFZ011_Block.mat',# w/imaging - stimulus artifact
                                '2023-07-27_1_OFZ008_Block.mat', '2023-07-28_1_OFZ008_Block.mat', 
                                '2023-07-24_1_OFZ011_Block.mat',# w/imaging - shifts in recording
                                '2023-07-15_1_OFZ008_Block.mat', '2023-07-18_1_OFZ008_Block.mat',
                                '2023-07-21_1_OFZ008_Block.mat', # w/imaging - light artifact
                                '2023-07-16_2_OFZ008_Block.mat','2023-07-16_3_OFZ008_Block.mat',
                                 ]
                                                            
        for animal in animalList:
            animalPath = self.rawPath + animal +'\\'
            for root, dirs, files in os.walk(animalPath):
                for fileName in files:
                    if fileName.endswith('Block.mat'):
                        # Extract the animal ID from the file name
                        recordingDate = fileName.split('_')[0]  # Assuming the animal ID is the part before the first underscore
                        recording_id  = fileName.split('_')[1]  # Assuming the animal ID is the part before the first underscore
                        animal_id     = fileName.split('_')[2]  # Assuming the animal ID is the part before the first underscore

                        # Mark the learning recordings
                        date = datetime.strptime(recordingDate, '%Y-%m-%d')
                        if (animal_id in ['OFZ009', 'OFZ010', 'OFZ011']) & (date < datetime(2023, 7, 10)):
                            learning = True
                        else:
                            learning = False

                        #Mark the imaging session
                        twoP_path = os.path.join(root[:-2],'TwoP')
                        twoP_exist = glob.glob (twoP_path + '/*t-001')
                        twoP_exist = len(twoP_exist)>0
                        if twoP_exist == True:
                            if fileName in badRecordingSessions:
                                roi = 0
                            elif (date >datetime(2023, 7, 3)) & (date < datetime(2023, 7, 10))  & (animal_id =='OFZ011'):
                                 roi = 7 # seperate these
                            elif  (date < datetime(2023, 7, 10)):
                                roi = 1
                            elif (date ==datetime(2023, 7, 20)) & (animal_id =='OFZ011'):
                                 roi = 10 # wrong labelling  
                            # elif (date ==datetime(2023, 7, 20)) & (animal_id =='OFZ011'):
                            #     roi = 15
                            # elif (date ==datetime(2023, 7, 21)) & (animal_id =='OFZ011'):
                            #     roi = 15
                            else:
                                imaging_filename = [f for f in glob.glob(twoP_path +'\\*t-001')]
                                imaging_filename = imaging_filename[0]
                                
                        else: # This line does not appear in the code
                            roi= 0
                        

                        # Create a dictionary representing a row with the animal ID and file path
                        row_data = {'animalID': animal_id, 
                                    'recordingDate': recordingDate, 
                                    'recordingID': recording_id, 
                                    'sessionName': fileName[:-10],
                                    'learningData': learning,
                                    'twoP':twoP_exist,
                                    'path': root[:-2],
                                    'sessionNameWithPath': os.path.join(root, fileName),
                                    'blockName': fileName[:-10],
                                    'imagingTiffFileNames': glob.glob(twoP_path +'\\*t-001')[0] if glob.glob(twoP_path +'\\*t-001') else ''}
                        # Append the row to the DataFrame
                        info  = pd.concat([info, pd.DataFrame([row_data])], ignore_index=True)     
        self.recordingList = info
        info.head()
        # Add main filepathname
        self.recordingList['analysispathname'] = np.nan
        for ind, recordingDate in enumerate(self.recordingList.recordingDate):
            filepathname = (self.recordingList.path[ind] +
                            '\\'+ self.recordingList.recordingID[ind])
            self.recordingList.loc[ind,'filepathname'] = filepathname  

            analysispathname = (self.analysisPath +
                                '\\' + self.recordingList.recordingDate[ind] + '_' + 
                                str(self.recordingList.animalID[ind]) + '_' + 
                                self.recordingList.recordingID[ind])   
            self.recordingList.loc[ind,'analysispathname'] = analysispathname +'\\'
            if not os.path.exists(analysispathname): 
                os.makedirs(analysispathname)
                    
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

# load suite2p data and compute dff
#raw, spks, stat = utils.s2p_loader(os.path.join(s2p_path, 'plane{}'.format(plane)))
#dff = utils.dfof2(raw) #compute dF/F - baseline is mean of whole trace here

# deal withframe clock
#tot_frames = dff.shape[1] * tot_planes
#frame_clock = utils.paq_data(paq, 'frame_clock', threshold_ttl = True)
#frame_clock = frame_clock[:tot_frames] # get rid of foxy bonus frames
#frame_clock = frame_clock[plane::tot_planes] # just take clocks from the frame you care about

def tiff_metadata(folderTIFF):

    ''' takes input of list of tiff folders and returns 
        number of frames in the first of each tiff folder '''
    
    # First check if tiff file is good and correct
    tiff_list = []
    tseries_nframes = []
    tiffs = utils.get_tiffs(folderTIFF)
    if not tiffs:
        raise print('cannot find tiff in '
                                    'folder {}'.format(tseries_nframes))
    elif len(tiffs) == 1:
        assert tiffs[0][-7:] == 'Ch3.tif', 'channel not understood '\
                                            'for tiff {}'.format(tiffs)
        tiff_list.append(tiffs[0])
    elif len(tiffs) == 2:  # two channels recorded (red is too dim)
        print('There are more than one tiff file - check: '+ folderTIFF)

    with tifffile.TiffFile(tiffs[0]) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value

    x_px = tif_tags['ImageWidth']
    y_px = tif_tags['ImageLength']
    image_dims = [x_px, y_px]

    n_frames = re.search('(?<=\[)(.*?)(?=\,)', 
                         tif_tags['ImageDescription'])

    n_frames = int(n_frames.group(0))
    tseries_nframes.append(n_frames)

    return image_dims, tseries_nframes

def getIndexForInterestedcellsID ( s_recDate, s_animalID, s_recID, s_cellID ):
    infoPath = 'C:\\Users\\Huriye\\Documents\\code\\clapfcstimulation\\analysis\\infoForAnalysis-readyForSelectingInterestedCells.pkl'    
    animalID, stimuliFamilarity, dataQuality,recData, recID, cellID, pvalsBoth, pvalsVis, pvalsOpto,dff_meanVisValue, dff_meanBothValue, dff_meanOptoValue, pupilID = pd.read_pickle(infoPath) 
    ind = np.where((np.array(animalID) == s_animalID) & (np.array(recID) == s_recID) & (np.array(cellID) == s_cellID) & (np.array(recData) == s_recDate))
    return ind

def selectInterestedcells ( aGroup, stimType, responsive = True, plotValues = False, pupil = True ):
    infoPath = 'C:\\Users\\Huriye\\Documents\\code\\clapfcstimulation\\analysis\\infoForAnalysis-readyForSelectingInterestedCells.pkl'    
    animalID, stimuliFamilarity, dataQuality,recData, recID, cellID, pvalsBoth, pvalsVis, pvalsOpto,dff_meanVisValue, dff_meanBothValue, dff_meanOptoValue, pupilID = pd.read_pickle(infoPath) 
    
    infoPath = 'C:\\Users\\Huriye\\Documents\\code\\clapfcstimulation\\analysis\\infoForAnalysis-readyForPlotting.pkl'
    dff_traceBoth, dff_traceVis, dff_traceOpto, dff_meanBoth1sec, dff_meanVis1sec, dff_meanOpto1sec = pd.read_pickle(infoPath) 

    CTAP_animals = [21104, 21107, 21108, 21109,22101,22102,22103,22104,22105,22106,22107,22108]
    NAAP_animals   = [21101, 21102, 21103, 21105, 21106]  
    control_animals = [23040, 23036, 23037]

    if aGroup == 'CTAP':
        # exclude inhibitory animals
        s = set(CTAP_animals)
        selectedAnimals = np.array([i in s for i in animalID])
    elif aGroup == 'NAAP':
        s = set(NAAP_animals)
        selectedAnimals = np.array([i in s for i in animalID])
    elif aGroup == 'Control':
        s = set(control_animals)
        selectedAnimals = np.array([i in s for i in animalID])

    # exclude trained stimuli
    if stimType == 'Trained':
        includeType = [2,3]
        s = set(includeType)
        selectedFamilarity = np.array([i in s for i in stimuliFamilarity])
        # exclude 22102 and 22108 as they did not learn
        s = [22101, 22103, 22104, 22105, 22106, 22107]
        selectedAnimals = np.array([i in s for i in animalID])
        selectedFamilarity = selectedFamilarity & selectedAnimals
    elif stimType  == 'Naive':
        includeType = [0, 1]
        s = set(includeType)
        selectedFamilarity = np.array([i in s for i in stimuliFamilarity])
    elif stimType  == 'Pupil-control-coveredMicroscope':
        includeType = [6]
        s = set(includeType)
        selectedFamilarity = np.array([i in s for i in stimuliFamilarity])
    elif stimType  == 'Pupil-control-not-coveredMicroscope':
        includeType = [7]
        s = set(includeType)
        selectedFamilarity = np.array([i in s for i in stimuliFamilarity])
    

        # select only pupil
    if pupil:
        includeType = [1]
        s = set(includeType)
        selectedPupil = np.array([i in s for i in pupilID])
    else:
        includeType = [0, 1]
        s = set(includeType)
        selectedPupil = np.array([i in s for i in pupilID])

    # # exclude not good quality data
    # includeType =[0, 1] 
    # s = set(includeType)
    # selectedQuality = np.array([i in s for i in dataQuality])

    selectedExpGroup = selectedAnimals & selectedFamilarity & selectedPupil #& selectedQuality

    # exclude non responsive units
    #p_fdr = 0.0001
    #p_standard = 0.05
    #responsiveOpto = (np.array(pvalsOpto) <= p_fdr)
    #responsiveNoOpto  = (np.array(pvalsOpto) > p_standard)
    if responsive==False:
        selectedCellIndex = selectedExpGroup
        responsiveNoSensory =[]
    else:

        temp = fdrcorrection(pvalsVis, alpha=0.05/3, method='i', is_sorted=False)
        responsiveVis  = temp[0]
        responsiveNoVis  = (np.array(pvalsVis) > 0.05)
    
        temp = fdrcorrection(pvalsOpto, alpha=0.05/3, method='i', is_sorted=False)
        responsiveOpto = temp[0]
        responsiveNoOpto = (np.array(pvalsOpto) > 0.05)
        
        temp = fdrcorrection(pvalsBoth, alpha=0.05/3, method='i', is_sorted=False)
        responsiveBoth = temp[0]
        responsiveNoBoth  = (np.array(pvalsBoth) > 0.05)

        responsiveVis  = selectedExpGroup & responsiveVis   # np.logical_and(selectedExpGroup,responsiveVis)
        responsiveOpto = selectedExpGroup & responsiveOpto  # np.logical_and(selectedExpGroup,responsiveOpto)
        responsiveBoth = selectedExpGroup & responsiveBoth  # np.logical_and(selectedExpGroup,responsiveBoth) 

        responsiveNoVis  = selectedExpGroup & responsiveNoVis   # np.logical_and(selectedExpGroup,responsiveVis)
        responsiveNoOpto = selectedExpGroup & responsiveNoOpto  # np.logical_and(selectedExpGroup,responsiveOpto)
        responsiveNoBoth = selectedExpGroup & responsiveNoBoth  # np.logical_and(selectedExpGroup,responsiveBoth) 

        responsiveOnlyVis    = responsiveVis & responsiveNoOpto & responsiveNoBoth
        responsiveOnlyOpto   = responsiveOpto & responsiveNoVis & responsiveNoBoth
        responsiveOnlyBoth   = responsiveBoth & responsiveNoOpto & responsiveNoVis
        responsiveAll = responsiveVis | responsiveOpto | responsiveBoth
        nonResponsiveAll = responsiveNoVis | responsiveNoOpto | responsiveNoBoth 

        if plotValues:
            print('All cell number:'+ str(np.sum(selectedExpGroup)))
            # All responsive cells
            responsiveAll = responsiveVis | responsiveOpto | responsiveBoth
            responsiveVisOpto = responsiveVis & responsiveOpto
            responsiveVisOptoBoth = responsiveVis & responsiveOpto & responsiveBoth
            print('Any responsive cell number:'+ str(np.sum(responsiveAll)))
            print('Visual AND opto responsive cell number:'+ str(np.sum(responsiveVisOpto)))
            print('Visual AND opto AND BOTH responsive cell number:'+ str(np.sum(responsiveVisOptoBoth)))

            # visual cue responsive cells
            responsiveOnlyVis   = responsiveVis & ~responsiveOpto
            responsiveOnlyVis   = np.logical_and(responsiveOnlyVis,~responsiveBoth)
            excDff = (np.array(dff_meanVisValue)> 0)
            inhDff = (np.array(dff_meanVisValue)<0)
            excOnly = excDff & responsiveOnlyVis
            inhOnly = inhDff & responsiveOnlyVis
            print('Visual cue - all visual responsive cells: '+ str(np.sum(responsiveVis)))
            print('Visual cue - only visual responsive: '+ str(np.sum(responsiveOnlyVis)))
            print('Visual cue - EXC opto responsive: '+ str(np.sum(excOnly)/np.sum(responsiveOnlyVis)))
            print('Visual cue - INH opto responsive: '+ str(np.sum(inhOnly)/np.sum(responsiveOnlyVis)))

            # pto cue responsive cells
            print('Opto stimulation - all opto responsive cells: '+ str(np.sum(responsiveOpto)))
            responsiveOnlyOpto   = responsiveOpto & responsiveNoVis
            responsiveOnlyOpto   = responsiveOnlyOpto & responsiveNoBoth
            print('Opto stimulation - only opto responsive: '+ str(np.sum(responsiveOnlyOpto)))
            excDff = (np.array(dff_meanBothValue) > 0)
            inhDff = (np.array(dff_meanBothValue) < 0)
            excOnly = excDff & responsiveOnlyOpto
            inhOnly = inhDff & responsiveOnlyOpto
            print('Opto stimulation - EXC opto responsive: '+ str(np.sum(excOnly)/np.sum(responsiveOnlyOpto)))
            print('Opto stimulation - INH opto responsive: '+ str(np.sum(inhOnly)/np.sum(responsiveOnlyOpto)))

            # Both cue responsive cells
            print('Both - all both responsive cells:'+ str(np.sum(responsiveBoth)))
            responsiveOnlyBoth  = np.logical_and(responsiveBoth,~responsiveVis)
            responsiveOnlyBoth   = np.logical_and(responsiveOnlyBoth,~responsiveOpto)
            print('Both - only both responsive: '+ str(np.sum(responsiveOnlyBoth)))
            excDff = (np.array(dff_meanBothValue) > 0)
            inhDff = (np.array(dff_meanBothValue) < 0)
            excOnly = excDff & responsiveOnlyBoth
            inhOnly = inhDff & responsiveOnlyBoth
            print('Both - EXC opto responsive: '+ str(np.sum(excOnly)/np.sum(responsiveOnlyBoth)))
            print('Both - INH opto responsive: '+ str(np.sum(inhOnly)/np.sum(responsiveOnlyBoth)))

            # # Sensory responsive cells
            # responsiveSensory  = responsiveVis | responsiveBoth
            # responsiveSensory  = responsiveSensory & ~responsiveOnlyOpto
            # print('Sensory responsive cell number:'+ str(np.sum(responsiveSensory)))
            # excDff = (np.array(dff_meanBothValue) > 0)
            # inhDff = (np.array(dff_meanBothValue) < 0)
            # excOnly = excDff & responsiveSensory
            # inhOnly = inhDff & responsiveSensory
            # print('responsiveSensory - EXC opto responsive: '+ str(np.sum(excOnly)/np.sum(responsiveSensory)))
            # print('responsiveSensory - INH opto responsive: '+ str(np.sum(inhOnly)/np.sum(responsiveSensory)))

            # responsiveNoSensory  = responsiveOnlyOpto #responsiveOpto & responsiveNoVis & responsiveNoBoth
            # print('NO sensory responsive cell number:'+ str(np.sum(responsiveNoSensory)))
            # excDff = (np.array(dff_meanBothValue) > 0)
            # inhDff = (np.array(dff_meanBothValue) < 0)
            # excOnly = excDff & responsiveNoSensory
            # inhOnly = inhDff & responsiveNoSensory
            # print('responsiveNoSensory - EXC opto responsive: '+ str(np.sum(excOnly)/np.sum(responsiveNoSensory)))
            # print('responsiveNoSensory - INH opto responsive: '+ str(np.sum(inhOnly)/np.sum(responsiveNoSensory)))
    
    if responsive =='Visual':
        selectedCellIndex = responsiveOnlyVis
    elif responsive =='Opto':
        selectedCellIndex = responsiveOnlyOpto
    elif responsive =='Both':
        selectedCellIndex = responsiveOnlyBoth
    elif responsive =='All':
        selectedCellIndex = responsiveAll
    elif responsive=='None':
        selectedCellIndex = nonResponsiveAll

    return selectedCellIndex

def norm_to_zero_one(row):
    min_val = np.nanmin(row)
    max_val = np.nanmax(row)
    normalized_row = (row - min_val) / (max_val - min_val)
    return normalized_row

def plot_dff_mean_traces (pathname, cellID, tTypes, axis):
        ## Parameters
    fRate = 1000/30
    responsiveness_test_duration = 1000.0 #in ms 
    simulationDur_ms = 350.0 # in ms
    simulationDur = int(np.ceil(simulationDur_ms/fRate))
    pre_frames    = 2000.0# in ms
    pre_frames    = int(np.ceil(pre_frames/fRate))
    post_frames   = 6000.0 # in ms
    post_frames   = int(np.ceil(post_frames/fRate))
    analysisWindowDur = 1500 # in ms
    analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))
    shutterLength     = int(np.round(simulationDur_ms/fRate))
    #tTypes = [ 'onlyVis', 'Both', 'onlyOpto']

    ########## Organise stimuli times 
    paqData = pd.read_pickle (pathname+'paq-data.pkl')
    paqRate = paqData['rate']
    # Get the stim start times 
    frame_clock    = utils.paq_data (paqData, 'prairieFrame', threshold_ttl=True, plot=False)
    optoStimTimes  = utils.paq_data (paqData, 'optoLoopback', threshold_ttl=True, plot=False)

    # the frame_clock is slightly longer in paq as there are some up to a sec delay from
    # microscope to PAQI/O software.  
    optoStimTimes = utils.stim_start_frame (paq=paqData, stim_chan_name='optoLoopback',
                                        frame_clock=None,stim_times=None, plane=0, n_planes=1)
    visStimTimes = utils.stim_start_frame (paq=paqData, stim_chan_name='maskerLED',
                                        frame_clock=None,stim_times=None, plane=0, n_planes=1)
    shutterTimes = utils.shutter_start_frame (paq=paqData, stim_chan_name='shutterLoopback',
                                        frame_clock=None,stim_times=None, plane=0, n_planes=1)

    # Lets organise it more for analysis friendly format
    trialStartTimes = np.unique(np.concatenate((optoStimTimes,visStimTimes),0))
    trialTypes = []
    for t in trialStartTimes:
        optoexist =  np.any(optoStimTimes== t)
        visexist  =  np.any( visStimTimes == t)
        if  optoexist  & visexist: 
            trialTypes += ['Visual + Opto']
        elif optoexist &~ visexist:
            trialTypes += ['Opto']
        elif ~optoexist & visexist:
            trialTypes += ['Visual']
        else:
            trialTypes += ['CHECK']
    trialStartTimes = shutterTimes
    #t = [idx for idx, t_type in enumerate(trialTypes) if t_type=='Both']

    ########## Organise calcium imaging traces 
    imData = pd.read_pickle (pathname +'imaging-data.pkl')
    fluR      = imData['flu']
    n_frames  = imData['n_frames']
    flu_raw   = imData['flu_raw']

    # Lets put nans for stimulated frames
    frameTimes = np.full((1,fluR.shape[1] ), False) # create a full false array
    for sT in shutterTimes:
        frameTimes[:,sT:(sT+shutterLength)] = True
    fluR[:, frameTimes[0,:]] = np.nan

    # clean detrended traces
    flu = utils.clean_traces(fluR)

    ### Get dff values for 4 trial types
    dffTrace ={} 
    dffTrace_mean ={}
    dffAfterStim1500ms_median ={}
    for indx, t in enumerate(tTypes) :
        if t =='All':
            trialInd = np.transpose(list(range(len(trialStartTimes))))
        else:
            trialInd = [idx for idx, t_type in enumerate(trialTypes) if t_type==t]
        
        if len(trialInd)>1:
            dffTrace[t]      = utils.flu_splitter(flu, trialStartTimes[trialInd], pre_frames, post_frames) # Cell x time x trial

    #create dff for all cells
    for indx, t in enumerate(tTypes):
        pfun.lineplot_withSEM (data=dffTrace[t][cellID], colorInd = indx, label=t, axis = axis)
                
                          
# def run_suite2p (self, data_path, filename): # Not tested - 05/03/2022 HA
#     from suite2p.run_s2p import run_s2p
#     ops = {
#         # main settings
#         'nplanes' : 1, # each tiff has these many planes in sequence
#         'nchannels' : 1, # each tiff has these many channels per plane
#         'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
#         'diameter': 12, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
#         'tau':  1.26, # this is the main parameter for deconvolution (1.25-1.5 for gcamp6s)
#         'fs': 30.,  # sampling rate (total across planes)
#         # output settings
#         'delete_bin': False, # whether to delete binary file after processing
#         'save_mat': True, # whether to save output as matlab files
#         'combined': True, # combine multiple planes into a single result /single canvas for GUI
#         # parallel settings
#         'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
#         'num_workers_roi': 0, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
#         # registration settings
#         'batch_size': 500, # reduce if running out of RAM
#         'do_registration': True, # whether to register data
#         'nimg_init': 300, # subsampled frames for finding reference image
#         'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
#         'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
#         'reg_tif': False, # whether to save registered tiffs
#         'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
#         # cell detection settings
#         'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
#         'navg_frames_svd': 5000, # max number of binned frames for the SVD
#         'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
#         'max_iterations': 20, # maximum number of iterations to do cell detection
#         'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
#         'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
#         'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
#         'threshold_scaling': 0.8, # adjust the automatically determined threshold by this scalar multiplier
#         'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
#         'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
#         'outer_neuropil_radius': np.inf, # maximum neuropil radius
#         'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
#         # deconvolution settings
#         'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
#         'baseline': 'maximin', # baselining mode
#         'win_baseline': 60., # window for maximin
#         'sig_baseline': 10., # smoothing constant for gaussian filter
#         'neucoeff': .7,  # neuropil coefficient
#     }
#     db = {
#     'data_path': data_path,
#     'tiff_list': data_path + filename, 
#     }
#     opsEnd = run_s2p (ops=ops,db=db)