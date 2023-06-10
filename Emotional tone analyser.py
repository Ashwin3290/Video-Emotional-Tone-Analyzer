import cv2
import moviepy.editor as mp
import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip

emotions=['pleasant_surprised', 'neutral', 'happy', 'angry', 'fear', 'disgust', 'sad']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.regularizers import L2
from tensorflow import data as dt
from tensorflow.keras.preprocessing import sequence

def create_model():
  model = Sequential()
  model.add(LSTM(128,return_sequences=False,input_shape=(40,127)))
  model.add(Dense(32))
  model.add(Dropout(0.5))
  model.add(Dense(32))
  model.add(Dropout(0.5))
  model.add(Dense(7,activation='softmax'))
  model.compile(optimizer="Adam", loss='CategoricalCrossentropy', metrics=["accuracy"])

  return model

model=create_model()
model.load_weights('best_weights.hdf5')

def clasify_emotions(mfcc):

    pred_emotions=[]
    data=dt.Dataset.from_tensor_slices(mfcc)
    prediction=model.predict(data.batch(1))
    for i in prediction:
        pred_emotions.append(emotions[np.argmax(i)])
    return pred_emotions

def extract_mfcc(filename):
    target_length= 65000
    mfccs=[]
    timestamps=[]

    try:
        audio, sr = librosa.load(filename)

    except (librosa.LibrosaError, OSError) as e:
        print(f"Error encountered while processing file: {filename}")
        return None
    
    audio_length=len(audio)
    number_chunks=int(audio_length/target_length)
    time_of_chunks=audio_length/(sr*number_chunks)
    print(f"Number of chunks {number_chunks},time of chunks {time_of_chunks}")

    current_spot=0
    for i in range(0,audio_length,target_length):
        if  audio_length < target_length:
            new_audio = np.pad(audio, (0, target_length - audio_length), 'constant')
        else:
            new_audio = audio[i:i+target_length]
            current_spot+=target_length
            audio_length-=target_length
            timestamps.append(i/sr)
        
        mfcc = librosa.feature.mfcc(y=new_audio, sr=sr,n_mfcc=40)
        mfccs.append(mfcc)
    
    print("MFCCs extracted")
    return mfccs[:-1], timestamps


def annotate_video_with_emotions(video_path, timestamps, emotions, output_path):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    current_emotion = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for timestamp, emotion in zip(timestamps, emotions):
            if current_time >= timestamp:
                current_emotion = emotion

        if current_emotion is not None:
            text = f"Emotion: {current_emotion}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_width, text_height = text_size[0], text_size[1]
            text_x = int((frame.shape[1] - text_width) / 2)
            text_y = int((frame.shape[0] - text_height) / 2)
            cv2.rectangle(frame, (text_x - 5, text_y - 5), (text_x + text_width + 5, text_y + text_height + 5),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, text, (text_x, text_y + text_height), font, font_scale, (0, 0, 0), font_thickness,
                        cv2.LINE_AA)

        out.write(frame)

        cv2.imshow("Annotated Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def extract_audio(video_path, output_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)

def replace_video_with_annotated(original_video_path, annotated_video_path, output_path):
    original_clip = VideoFileClip(original_video_path)
    annotated_clip = VideoFileClip(annotated_video_path)

    final_clip = annotated_clip.set_audio(original_clip.audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec="aac")

    original_clip.close()
    annotated_clip.close()

import time

if __name__ == '__main__':
    t=time.time()
    video_path = input("Enter video path: ")
    rot_dir=""
    path=[]
    if "\\" in video_path:
        path=video_path.split("\\")
        root_dir=path[:-1]
        video_path=os.path.join(*path)
    audio_path = 'extracted_audio.wav'
    final_path=os.path.join(*root_dir,path[-1]+"_final.mp4")
    extract_audio(video_path, audio_path)
    print("Audio extracted at ",time.time()-t," seconds")
    mfccs, timestamps = extract_mfcc(audio_path)
    print("MFCCs extracted at ",time.time()-t," seconds")
    emotions = clasify_emotions(mfccs)
    print("Emotions classified at ",time.time()-t," seconds")
    annotate_video_with_emotions(video_path, timestamps, emotions, 'annotated_video.mp4')
    print("Video annotated at ",time.time()-t," seconds")
    replace_video_with_annotated(video_path, 'annotated_video.mp4',final_path)
    print("Video replaced at ",time.time()-t," seconds")
    os.remove('annotated_video.mp4')
    os.remove(audio_path)
    print("Done in ",time.time()-t," seconds")
