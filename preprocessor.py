import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_DIR='fma_small/fma_small'

def get_tids_from_directory(audio_dir):
    """
    Get track IDs from the mp3s in a directory.
    """
    
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids

def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    """
    
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def create_spectogram(track_id):
    """
    Create spectogram of the given track
    """
    
    filename = get_audio_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T



def create_array(df):
    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            spect = create_spectogram(track_id)
    
            # Normalize for small shape differences
            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(dict_genres[genre])
            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

def splitDataFrameIntoSmaller(df, chunkSize = 1600):
    """
    Splits dataframe to smaller chunks 
    """
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

def shuffle_copies(a, b):
    """
    Create a random permutation of the given dataframes
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

tids = get_tids_from_directory(AUDIO_DIR)
filepath = 'metadata/tracks.csv'
tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
keep_cols = [('set', 'split'),
('set', 'subset'),('track', 'genre_top')]

df_all = tracks[keep_cols]
df_all = df_all[df_all[('set', 'subset')] == 'small']

df_all['track_id'] = df_all.index

dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4, 
               'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8  }


# Prepare datasets
df_train = df_all[df_all[('set', 'split')]=='training']
df_valid = df_all[df_all[('set', 'split')]=='validation']
df_test = df_all[df_all[('set', 'split')]=='test']

X_test, y_test = create_array(df_test)
np.savez('test_arr', X_test, y_test)
X_valid, y_valid = create_array(df_valid)
np.savez('valid_arr', X_valid, y_valid)

listDf = splitDataFrameIntoSmaller(df_train)
df1_train = listDf[0]
df2_train = listDf[1]
df3_train = listDf[2]
df4_train = listDf[3]

X_train1, y_train1 = create_array(df1_train)
np.savez('train1_arr', X_train1, y_train1)
X_train2, y_train2 = create_array(df2_train)
np.savez('train2_arr', X_train2, y_train2)
X_train3, y_train3 = create_array(df3_train)
np.savez('train3_arr', X_train3, y_train3)
X_train4, y_train4 = create_array(df4_train)
np.savez('train4_arr', X_train4, y_train4)

npzfile = np.load('train1_arr.npz')
X_train1 = npzfile['arr_0']
y_train1 = npzfile['arr_1']

npzfile = np.load('train2_arr.npz')
X_train2 = npzfile['arr_0']
y_train2 = npzfile['arr_1']

npzfile = np.load('train3_arr.npz')
X_train3 = npzfile['arr_0']
y_train3 = npzfile['arr_1']

npzfile = np.load('train4_arr.npz')
X_train4 = npzfile['arr_0']
y_train4 = npzfile['arr_1']

npzfile = np.load('valid_arr.npz')
X_valid = npzfile['arr_0']
y_valid = npzfile['arr_1']

X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis = 0)
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis = 0)

y_train = y_train -1
y_valid = y_valid -1
X_train_raw = librosa.core.db_to_power(X_train, ref=1.0)
X_train_log = np.log(X_train_raw)
X_valid_raw = librosa.core.db_to_power(X_valid, ref=1.0)
X_valid_log = np.log(X_valid_raw)

X_train, y_train = shuffle_copies(X_train_log, y_train)
X_valid, y_valid = shuffle_copies(X_valid_log, y_valid)

# Save the preprocessed datasets
np.savez('preprocessed_data/shuffled_train', X_train, y_train)
np.savez('preprocessed_data/shuffled_valid', X_valid, y_valid)