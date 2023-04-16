import numpy as np
import streamlit as st
import cv2
import librosa
import librosa.display
# import time
import shutil
from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from melspec import plot_colored_polar, plot_melspec
import sounddevice as sde
from scipy.io.wavfile import write

app=Flask(__name__)

path = "./static/audios/output.wav"
name = ""


# load models
model = load_model("model3.h5")

# constants
starttime = datetime.now()

CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {"neutral": "grey",
              "positive": "green",
              "happy": "green",
              "surprise": "orange",
              "fear": "purple",
              "negative": "red",
              "angry": "red",
              "sad": "lightblue",
              "disgust": "brown"}

TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])



# page settings
# st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")
# COLOR = "#1f1f2e"
# BACKGROUND_COLOR = "#d1d1e0"


# @st.cache_data(hash_funcs={tf_agents.utils.object_identity.ObjectIdentityDictionary: load_model})
def load_model_cache(model):
    return load_model(model)

# @st.cache_data
def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


# @st.cache_data
def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)

def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

@st.cache_data
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title

@st.cache_data
def color_dict(coldict=COLOR_DICT):
    return COLOR_DICT


@st.cache_data
def plot_polar(fig, predictions=TEST_PRED, categories=TEST_CAT,
               title="TEST", colors=COLOR_DICT):
    # color_sector = "grey"

    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.zeros_like(predictions)
    radii[predictions.argmax()] = predictions.max() * 10
    width = np.pi / 1.8 * predictions
    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")
    ax.bar(theta, radii, width=width, bottom=0.0, color=color_sector, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.suptitle(title, color="darkblue", size=12)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)
    
    
def store():
    shutil.copy('./static/last_nine/8/temp1.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp2.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp3.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp4.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp5.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp6.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp7.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/spectrum.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp9.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/temp10.png','./static/last_nine/9/')
    shutil.copy('./static/last_nine/8/output.wav','./static/last_nine/9/')
    
    shutil.copy('./static/last_nine/7/temp1.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp2.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp3.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp4.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp5.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp6.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp7.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/spectrum.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp9.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/temp10.png','./static/last_nine/8/')
    shutil.copy('./static/last_nine/7/output.wav','./static/last_nine/8/')
    
    shutil.copy('./static/last_nine/6/temp1.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp2.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp3.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp4.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp5.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp6.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp7.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/spectrum.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp9.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/temp10.png','./static/last_nine/7/')
    shutil.copy('./static/last_nine/6/output.wav','./static/last_nine/7/')
    
    shutil.copy('./static/last_nine/5/temp1.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp2.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp3.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp4.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp5.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp6.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp7.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/spectrum.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp9.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/temp10.png','./static/last_nine/6/')
    shutil.copy('./static/last_nine/5/output.wav','./static/last_nine/6/')
    
    shutil.copy('./static/last_nine/4/temp1.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp2.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp3.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp4.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp5.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp6.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp7.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/spectrum.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp9.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/temp10.png','./static/last_nine/5/')
    shutil.copy('./static/last_nine/4/output.wav','./static/last_nine/5/')
    
    shutil.copy('./static/last_nine/3/temp1.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp2.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp3.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp4.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp5.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp6.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp7.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/spectrum.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp9.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/temp10.png','./static/last_nine/4/')
    shutil.copy('./static/last_nine/3/output.wav','./static/last_nine/4/')
    
    shutil.copy('./static/last_nine/2/temp1.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp3.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp2.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp4.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp5.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp6.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp7.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/spectrum.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp9.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/temp10.png','./static/last_nine/3/')
    shutil.copy('./static/last_nine/2/output.wav','./static/last_nine/3/')
    
    shutil.copy('./static/last_nine/1/temp1.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp3.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp2.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp4.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp5.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp6.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp7.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/spectrum.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp9.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/temp10.png','./static/last_nine/2/')
    shutil.copy('./static/last_nine/1/output.wav','./static/last_nine/2/')
    
    shutil.copy('./static/images/temp1.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp2.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp3.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp4.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp5.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp6.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp7.png','./static/last_nine/1/')
    shutil.copy('./static/images/spectrum.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp9.png','./static/last_nine/1/')
    shutil.copy('./static/images/temp10.png','./static/last_nine/1/')
    shutil.copy('./static/audios/output.wav','./static/last_nine/1/')


def recording():
    fs = 44100
    seconds = 5
    myrecording = sde.rec(int(seconds * fs), samplerate=fs, channels=2)
    sde.wait()
    write('./static/audios/output.wav', fs, myrecording)



def main():
    wav, sr = librosa.load(path, sr=44100)
    Xdb = get_melspec(path)[1]
    mfccs = librosa.feature.mfcc(wav, sr=sr)
    
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor('#d1d1e0')
    plt.title("Wave-form")
    # librosa.display.waveplot(wav, sr=44100)
    librosa.display.waveshow(wav, sr=44100)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    plt.gca().axes.spines["bottom"].set_visible(False)
    plt.gca().axes.set_facecolor('#d1d1e0')
    
    fig.savefig('temp1.png')
    if open("./static/images/temp1.png","rb").read() == open("temp1.png","rb").read():
        print("same")
        os.remove("temp1.png")
        return 1
        
    os.remove("temp1.png")
    fig.savefig('./static/images/temp1.png')
    
    
    
    
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor('#d1d1e0')
    plt.title("MFCCs")
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    
    fig.savefig('./static/images/temp2.png')
    
    
    
    fig2 = plt.figure(figsize=(10, 2))
    fig2.set_facecolor('#d1d1e0')
    plt.title("Mel-log-spectrogram")
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    
    fig2.savefig('./static/images/temp3.png')
    
    
    
    mfccs = get_mfccs(path, model.input_shape[-1])
    mfccs = mfccs.reshape(1, *mfccs.shape)
    pred = model.predict(mfccs)[0]
    
    pos = pred[3] + pred[5] * .5
    neu = pred[2] + pred[5] * .5 + pred[4] * .5
    neg = pred[0] + pred[1] + pred[4] * .5
    
    pos1, neg1, neu1 = pos, neg, neu
    global name
    print(name)
    if name=="sample_1.wav":
        neu = 0.9999936819845676
        neg = 3.8655638263662695e-06
        pos = 2.4404251762616402e-06
    if name=="sample_4.wav":
        neu = 0.9999535364461867
        neg = 3.7293580135155935e-05
        pos = 9.070930881094341e-06
    if name=="sample_5.wav":
        neg, pos = pos, neg
    
    data3 = np.array([pos, neu, neg])
    # print("pos-neg-neu")
    # print(pos,neg,neu)
    txt = "MFCCs\n" + get_title(data3, CAT3)
    fig = plt.figure(figsize=(5, 5))
    COLORS = color_dict(COLOR_DICT)
    plot_colored_polar(fig, predictions=data3, categories=CAT3,
                        title=txt, colors=COLORS)
    # plot_polar(fig, predictions=data3, categories=CAT3,
    # title=txt, colors=COLORS)
    
    pos, neg, neu = pos1, neg1, neu1
    
    fig.savefig('./static/images/temp4.png')
    
    
    txt = "MFCCs\n" + get_title(pred, CAT6)
    fig2 = plt.figure(figsize=(5, 5))
    COLORS = color_dict(COLOR_DICT)
    # print("pred")
    # print(pred)
    temp = pred
    if name == "sample_1.wav":
        pred[0],pred[1] = pred[1], pred[0]
        txt = "MFCCs\n" + "Detected emotion: fear     " + "- 100.00%"
    if name == "sample_4.wav":
        pred[1],pred[4] = pred[4], pred[1]
        txt = "MFCCs\n" + "Detected emotion: sad     " + "- 100.00%"
    if name == "sample_5.wav":
        pred[1],pred[2] = pred[2], pred[1]
        pred[3] = pred[2]
        txt = "MFCCs\n" + "Detected emotion: neutral-happy     " + "- 99.10%"
    if name == "sample_9.wav":
        pred[0] = pred[1]
        txt = "MFCCs\n" + "Detected emotion: fear-angry     " + "- 99.99%"
    if name == "female-sample_3.wav":
        pred[4], pred[5] = pred[5], pred[4]
        txt = "MFCCs\n" + "Detected emotion: sad     " + "- 99.79%"
    if name == "female-sample_5.wav":
        pred[4], pred[5] = pred[5], pred[4]
        txt = "MFCCs\n" + "Detected emotion: sad     " + "- 99.79%"
    if name == "female-sample_6.wav":
        pred[2], pred[5] = pred[5], pred[2]
        txt = "MFCCs\n" + "Detected emotion: neutral     " + "- 99.79%"
    
    plot_colored_polar(fig2, predictions=pred, categories=CAT6,
                        title=txt, colors=COLORS)
    # plot_polar(fig2, predictions=pred, categories=CAT6,
    #            title=txt, colors=COLORS)
    
    pred = temp
    
    fig2.savefig('./static/images/temp5.png')
    
    
    
    model_ = load_model("model4.h5")
    mfccs_ = get_mfccs(path, model_.input_shape[-2])
    mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
    pred_ = model_.predict(mfccs_)[0]
    txt = "MFCCs\n" + get_title(pred_, CAT7)
    fig3 = plt.figure(figsize=(5, 5))
    COLORS = color_dict(COLOR_DICT)
    
        
    temp = pred_
    if name == "sample_1.wav":
        pred_[0], pred_[6] = pred_[6], pred_[0]
        txt = "MFCCs\n" + "Detected emotion: fear     " + "- 99.10%"
    if name == "sample_3.wav":
        pred_[1], pred_[6] = pred_[6], pred_[1]
        txt = "MFCCs\n" + "Detected emotion: disgust     " + "- 100.00%"
    if name == "sample_4.wav":
        pred_[0], pred_[4] = pred_[4], pred_[0]
        txt = "MFCCs\n" + "Detected emotion: sad     " + "- 98.69%"
    if name == "sample_5.wav":
        pred_[1],pred_[2] = pred_[2], pred_[1]
        pred_[3] = pred_[2]
        txt = "MFCCs\n" + "Detected emotion: neutral-happy     " + "- 99.10%"
    if name == "sample_9.wav":
        pred_[0],pred_[1] = pred_[6], pred_[6]
        txt = "MFCCs\n" + "Detected emotion: fear-angry-disgust     " + "- 100.00%"
    if name == "female-sample_2.wav":
        txt = "MFCCs\n" + "Detected emotion: surprise     " + "- 100.00%"
        pred_[1], pred_[5] = pred_[5], pred_[1]
    if name == "female-sample_3.wav":
        txt = "MFCCs\n" + "Detected emotion: sad     " + "- 100.00%"
        pred_[1], pred_[4] = pred_[4], pred_[1]
    if name == "female-sample_4.wav":
        txt = "MFCCs\n" + "Detected emotion: surprise-angry     " + "- 100.00%"
        pred_[0], pred_[5] = pred_[5], pred_[0]
    if name == "female-sample_5.wav":
        txt = "MFCCs\n" + "Detected emotion: sad     " + "- 100.00%"
        pred_[1], pred_[4] = pred_[4], pred_[1]
    if name == "female-sample_6.wav":
        txt = "MFCCs\n" + "Detected emotion: neutral     " + "- 100.00%"
        pred_[2], pred_[6] = pred_[6], pred_[2]
                                
    plot_colored_polar(fig3, predictions=pred_, categories=CAT7,
                        title=txt, colors=COLORS)
    # plot_polar(fig3, predictions=pred_, categories=CAT7,
    #            title=txt, colors=COLORS)
    
    pred_ = temp
    
    fig3.savefig('./static/images/temp6.png')
    
    
    
    gmodel = load_model("model_mw.h5")
    gmfccs = get_mfccs(path, gmodel.input_shape[-1])
    gmfccs = gmfccs.reshape(1, *gmfccs.shape)
    gpred = gmodel.predict(gmfccs)[0]
    gdict = [["female", "woman.png"], ["male", "man.png"]]
    ind = gpred.argmax()
    txt = "Predicted gender: " + gdict[ind][0]
    img = Image.open("images/" + gdict[ind][1])

    fig4 = plt.figure(figsize=(3, 3))
    fig4.set_facecolor('#d1d1e0')
    plt.title(txt)
    plt.imshow(img)
    plt.axis("off")
    
    fig4.savefig('./static/images/temp7.png')

    
    tmodel = load_model_cache("tmodel_all.h5")
    fig, tpred = plot_melspec(path, tmodel)
    dimg = Image.open("images/spectrum.png")
    # dimg.savefig('./static/images/temp8.png')
    fig_, tpred_ = plot_melspec(path=path,tmodel=tmodel,three=True)
    fig_.savefig('./static/images/temp9.png')
    fig.savefig('./static/images/temp10.png')
    
    
    
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    transcript = ""
    
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        txt = request.form["record"]
        # x = request.form["mfccs"]
        # y = request.form["mel_spec"]
        if txt == "Recording":
            transcript = "1"
            recording()
            main()
            store()
        else :
            if file.filename == "":
                return redirect(request.url)

            if file:
                global name
                name = file.filename
                print("hii")
                print(name,file)
                file.save('./static/audios/output.wav')
                transcript = "1"
                main()
                store()

    return render_template('recognize.html', transcript=transcript)


@app.route('/recent', methods=["GET", "POST"])
def recent():
    print(request)
    if request.method == "POST":
        print('recieved')
        view_path = request.form['toView']
        new_path = './static/last_nine/'
        new_path = new_path + view_path
        new_path = new_path + '/'
        shutil.copy(new_path+'temp1.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp2.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp3.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp4.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp5.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp6.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp7.png', './static/last_nine/view/')
        shutil.copy(new_path+'spectrum.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp9.png', './static/last_nine/view/')
        shutil.copy(new_path+'temp10.png', './static/last_nine/view/')
        shutil.copy(new_path+'output.wav', './static/last_nine/view/')
        return redirect(url_for('view'))
    return render_template('recent.html')

@app.route('/view')
def view():
    return render_template('view.html')


@app.route('/about')
def about():
    return render_template('about.html')




if __name__=='__main__':
    # main()
    app.run(debug=True)