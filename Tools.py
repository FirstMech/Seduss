from demo import make_animation
from skimage import img_as_ubyte
from demo import load_checkpoints
from IPython.display import display, Javascript
from google.colab import output
from google.colab.output import eval_js
import base64
from multiprocessing import Process

import torch

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")


# edit the config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
source_image_path = './assets/avatar.png'
driving_video_path = 'recorded_video.mp4'
output_video_path = './generated.mp4'
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
predict_mode = 'relative' # ['standard', 'relative', 'avd']
find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result
pixel = 256 # for vox, taichi and mgif, the resolution is 256*256

def record_video(duration=5, quality=0.8):
    js = Javascript('''
        async function recordVideo(duration, quality) {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); // Disable audio
            document.body.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Record video for the given duration
            const chunks = [];
            const recorder = new MediaRecorder(stream);
            recorder.ondataavailable = e => chunks.push(e.data);
            recorder.start();

            await new Promise(resolve => setTimeout(resolve, duration * 1000));

            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                const reader = new FileReader();
                reader.onloadend = function() {
                    const base64data = reader.result;
                    google.colab.kernel.invokeFunction('storeVideo', [base64data], {});
                };
                reader.readAsDataURL(blob);
            };

            // Stop recording and stream
            setTimeout(() => {
                recorder.stop();
                stream.getVideoTracks()[0].stop();
                video.remove();
            }, duration * 1000);
        }
    ''')

    display(js)

    def store_video(base64data):
        with open('recorded_video.mp4', 'wb') as f:
            f.write(base64.b64decode(base64data.split(',')[1]))
        print("Video saved as 'recorded_video.mp4'.")

    def store_video_parallel(base64data):
        p = Process(target=store_video, args=(base64data,))
        p.start()
        p.join()

    output.register_callback('storeVideo', store_video_parallel)
    eval_js('recordVideo({}, {})'.format(duration, quality))


def run():

    source_image = imageio.imread(source_image_path)
    reader = imageio.get_reader(driving_video_path)

    source_image = resize(source_image, (pixel, pixel))[..., :3]

    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]
  
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)
    
    predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    
    return source_image, driving_video, predictions
    
