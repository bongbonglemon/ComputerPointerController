# Computer Pointer Controller

Control your mouse pointer by moving your face. My algorithm uses pre-trained deep learning models to estimate your gaze and moves your mouse pointer accordingly. Built with OpenVINO 2020.1 for Udacity's Intel EdgeAI final project.


## Project Set Up and Installation
1. Install OpenVINO 2020.1 here: https://docs.openvinotoolkit.org/latest/index.html
2.  Install dependencies 

    pip install -r requirements.txt

3. Download these models from Intel Open Zoo using *model downloader* from the toolkit:

    sudo ./downloader.py --name MODEL-NAME -o OUTPUT-DIR

List of **MODEL-NAME**:
 - face-detection-adas-binary-0001 
 - head-pose-estimation-adas-0001
 - landmarks-regression-retail-0009 
 - gaze-estimation-adas-0002

Take note of the full paths you have decided to download your models to as you will need them for the demo!

## Demo
In the src directory, run 'python main.py' with the following params

 1. -i  < file path to demo.mp4 in the bin >  ending with /starter/bin/demo.mp4
 2. -f  < file path to the face detection model > ending with /intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
 3. -e  < file path to the landmark detection model > ending with /intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009
 4. -a  < file path to the head pose model > ending with /intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
 5.  -g  < file-path to the gaze model > ending with /intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-000

## Documentation

If you want to use webcam add -v cam 

## Benchmarks

**FP32**
Time taken to load face detection model (in seconds): 0.34676194190979004

Time taken to load landmark detection model (in seconds): 0.10683393478393555

Time taken to load head pose estimation model (in seconds): 0.12396597862243652

Time taken to load gaze estimation model (in seconds): 0.16040396690368652

Time take for face detection inference (in seconds): 0.0394740104675293

Time take for landmark detection inference (in seconds): 3.0994415283203125e-06

Time take for head pose estimation inference (in seconds): 0.0028836727142333984

Time take for gaze estimation inference (in seconds): 0.00426483154296875

**FP16**

Face detection model does have FP16

Time taken to load landmark detection model (in seconds): 0.15234684944152832

Time taken to load head pose estimation model (in seconds): 0.19573688507080078

Time taken to load gaze estimation model (in seconds): 0.20962214469909668

Time take for face detection inference (in seconds): 0.03774118423461914

Time take for landmark detection inference (in seconds): 2.1457672119140625e-06

Time take for head pose estimation inference (in seconds): 0.002727031707763672

Time take for gaze estimation inference (in seconds): 0.004296302795410156

## Results

FP16 models took a slightly longer time to load than FP32 models but was slightly faster in inference.  Since FP16 takes up less memory, the model does not have to access the memory as much. Computations are also sped up since there's less arithmetic bandwidth.
 
### Edge Cases

When the subject's head is turned almost 90 degrees, the subject's eyes will be very close to the edge in the crop of the face which means I cannot use the same crop size for extracting the eyes in these images. So for these edge case, I just used a smaller crop size for the eyes.
