
# Emotional Tone Recognition and Annotation

## Introduction
Emotional Tone Recognition and Annotation is a powerful Python script that leverages advanced machine learning techniques to analyze and understand the emotional tone of videos. It employs emotion recognition algorithms to classify the underlying emotional states expressed within the video content. The script further annotates the video frames with the predicted emotions, enhancing the viewing experience and providing valuable insights into the emotional dynamics of the video.

## How It Works
The script follows a series of steps to perform emotional tone recognition and annotation:

1. **Emotion Classification Model**: The script utilizes a pre-trained neural network model to classify emotions. The model has been trained on a diverse dataset and can accurately predict emotional states based on patterns and features extracted from the video content.

2. **Feature Extraction**: The script extracts relevant features from the video frames, capturing visual cues such as facial expressions, body language, and color tones. These features serve as input to the emotion classification model, enabling accurate emotion prediction.

3. **Emotion Prediction**: The extracted features are fed into the emotion classification model, which predicts the emotional tone for each frame of the video. The emotions are classified into a predefined set of categories, such as happiness, sadness, anger, surprise, fear, disgust, and more.

4. **Annotation Overlay**: The script overlays the predicted emotions as subtitles onto the video frames. The subtitles indicate the emotional state at each moment in the video, providing viewers with an enhanced understanding of the emotional dynamics unfolding on the screen.

## How to Use
To use Emotional Tone Recognition and Annotation for your videos, follow these steps:

1. **Install Dependencies**: Ensure that you have Python 3.10 or above installed on your system. Install the necessary dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```


2. **Prepare Video**: Place your video file in the same directory as the script. Ensure that the video file is in a compatible format (e.g., MP4).

3. **Run the Script**: Open a terminal or command prompt, navigate to the script's directory, and execute the following command:
Replace `Emotional tone analyser.py` with the actual name of your Python script.

4. **Monitor Progress**: The script will display the progress of each step, including feature extraction, emotion classification, and annotation overlay. Depending on the length and complexity of the video, the process may take some time.

5. **View the Annotated Video**: Once the script completes, you will find the annotated video in the same directory as the input video. Play the video using any media player to observe the emotional subtitles overlaid on the frames.

Feel free to experiment with different videos and explore the emotional dynamics captured by Emotional Tone Recognition and Annotation. This tool provides a unique perspective on video content, offering insights into the underlying emotions and enhancing the overall viewing experience.

## Requirements
- Python 3.10 or above
- OpenCV (cv2)
- MoviePy
- TensorFlow

---
**Feel free to point out issue or improvements with the code**

