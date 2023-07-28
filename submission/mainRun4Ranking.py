"translation from MATLAB by Julia Dietlmeier"
"email: julia.dietlmeier@insight-centre.org"
# This is a demo to generate validation results into the submission folder (this step is required only during validation phase!!!)
# MICCAI "CMRxRecon" challenge 2023 
# 2023.05.06 @ fudan university
# Email: wangcy@fudan.edu.cn

# to reduce the computing burden and space, we only evaluate the central 2 slices
# For cine: use the first 3 time frames for ranking!
# For mapping: we need all weighting for ranking!
# crop the middle 1/6 of the original image for ranking
import os
import numpy as np
from run4Ranking import run4Ranking
from scipy.io import loadmat, savemat
# add path
# put your data directory here
basePath = '/media/daa/My Passport/Submission/'# %'Submission/'
mainSavePath = '/media/daa/My Passport/Submission_ranked/'# %'Submission/'
modality = 'Cine'# % options: 'Cine' for task1, 'Mapping' for task2

# do not make changes
AFtype = ['AccFactor04','AccFactor08','AccFactor10']#;
setName = 'ValidationSet/'#; % options: 'ValidationSet/', 'TestSet/'
ModalityName=[]

if modality=='Cine':
    ModalityName.append('cine_lax')
    ModalityName.append('cine_sax')
else:
    ModalityName.append('T1map')
    ModalityName.append('T2map')
    
#%% Generate folder for submission


#"=== SingleCoil ==============================================================="
# single coil
coilInfo = 'SingleCoil/'#;  % options: 'MultiCoil','SingleCoil'

for ind0 in range(0,3):
    mainDataPath = basePath + coilInfo + modality + '/' + setName + AFtype[ind0]
    FileList = os.listdir(mainDataPath)
    NumFile = len(FileList)
    k = 0
    # running all patients
    for ind1 in range(0,NumFile):
        print(ind1)
        #if isequal(FileList(ind1).name(1),'.')# DON'T know what to do with this if-statement
        #    k = k+1;
        #    continue;
        #end
        print(['Progress start for subject ', str(ind1-k)]);
        file_name = FileList[ind1]#.name;
        # modality1
        dataPath = mainDataPath+'/'+file_name+'/'+ModalityName[0]+'.mat'
        
        if os.path.isfile(dataPath)==True:
#------------------------------------------------------------------------------            
            dataRecon1 = loadmat(dataPath)#; % load recon data
            
            img = dataRecon1['img4ranking']#.img4ranking; % put your variable name here
            # to reduce the computing burden and space, we only evaluate the central 2 slices
            # For cine: use the first 3 time frames for ranking!
            # For mapping: we need all weighting for ranking!
            img4ranking = run4Ranking(img, ModalityName[0]);
            savePath = mainSavePath+coilInfo+modality+'/'+setName+AFtype[ind0]
            # mkdir for saving
            if not os.path.exists(savePath+'/'+file_name):
                os.mkdir(savePath+'/'+file_name)
            
            mdic = {"img4ranking": img4ranking}
            savemat(savePath+'/'+file_name+'/'+ModalityName[0]+'.mat', mdic)
        
        # modality2
        dataPath = mainDataPath+'/'+file_name+'/'+ModalityName[1]+'.mat'
        
        if os.path.isfile(dataPath)==True:
#------------------------------------------------------------------------------            
            dataRecon2 = loadmat(dataPath)#; % load recon data
            
            img = dataRecon2['img4ranking']#.img4ranking; % put your variable name here
            # to reduce the computing burden and space, we only evaluate the central 2 slices
            # For cine: use the first 3 time frames for ranking!
            # For mapping: we need all weighting for ranking!
            img4ranking = run4Ranking(img, ModalityName[1]);
            savePath = mainSavePath+coilInfo+modality+'/'+setName+AFtype[ind0]
            # mkdir for saving
            if not os.path.exists(savePath+'/'+file_name):
                os.mkdir(savePath+'/'+file_name)
            
            mdic = {"img4ranking": img4ranking}
            savemat(savePath+'/'+file_name+'/'+ModalityName[1]+'.mat', mdic)

        print(" single coil data generation successful!",str(AFtype[ind0]))

m
# MultiCoil
coilInfo = 'MultiCoil/'#;  % options: 'MultiCoil','SingleCoil'
for ind0 in range(0,3):
    mainDataPath = basePath + coilInfo + modality + '/' + setName + AFtype[ind0]
    FileList = os.listdir(mainDataPath)
    NumFile = len(FileList)
    k = 0
    # running all patients
    for ind1 in range(0,NumFile):
        #if isequal(FileList(ind1).name(1),'.')# DON'T know what to do with this if-statement
        #    k = k+1;
        #    continue;
        #end
        print(['Progress start for subject ', str(ind1-k)]);
        file_name = FileList[ind1]#.name;
        # modality1
        dataPath = mainDataPath+'/'+file_name+'/'+ModalityName[0]+'.mat'
        
        if os.path.isfile(dataPath)==True:
#------------------------------------------------------------------------------            
            dataRecon1 = loadmat(dataPath)#; % load recon data
            
            img = dataRecon1['img4ranking']#.img4ranking; % put your variable name here
            # to reduce the computing burden and space, we only evaluate the central 2 slices
            # For cine: use the first 3 time frames for ranking!
            # For mapping: we need all weighting for ranking!
            img4ranking = run4Ranking(img, ModalityName[0]);
            savePath = mainSavePath+coilInfo+modality+'/'+setName+AFtype[ind0]
            # mkdir for saving
            if not os.path.exists(savePath+'/'+file_name):
                os.mkdir(savePath+'/'+file_name)
            
            mdic = {"img4ranking": img4ranking}
            savemat(savePath+'/'+file_name+'/'+ModalityName[0]+'.mat', mdic)
        
        # modality2
        dataPath = mainDataPath+'/'+file_name+'/'+ModalityName[1]+'.mat'
        
        if os.path.isfile(dataPath)==True:
#------------------------------------------------------------------------------            
            dataRecon2 = loadmat(dataPath)#; % load recon data
            
            img = dataRecon2['img4ranking']#.img4ranking; % put your variable name here
            # to reduce the computing burden and space, we only evaluate the central 2 slices
            # For cine: use the first 3 time frames for ranking!
            # For mapping: we need all weighting for ranking!
            img4ranking = run4Ranking(img, ModalityName[1]);
            savePath = mainSavePath+coilInfo+modality+'/'+setName+AFtype[ind0]
            # mkdir for saving
            if not os.path.exists(savePath+'/'+file_name):
                os.mkdir(savePath+'/'+file_name)
            
            mdic = {"img4ranking": img4ranking}
            savemat(savePath+'/'+file_name+'/'+ModalityName[1]+'.mat', mdic)

        print(" multi coil data generation successful!",str((AFtype[ind0])))