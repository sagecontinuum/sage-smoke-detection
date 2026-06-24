# sage-smoke-detection
The edge compute environment consist of the sensor data acquisition system (HPWREN cameras [3] or Wild Sage Nodes) and the smoke detection plugin. The plugin uses a novel deep learning architecture called SmokeyNet [4] which uses spatiotemporal information from camera imagery for real-time wildfire smoke detection. When trained on the FIgLib dataset, SmokeyNet outperforms comparable baselines and rivals human performance. The trainning data comes from the Fire Ignition Library [5] which is an open source image library that consist of historical fires that were labeled as smoke or no smoke 40 minutes before and after the fire started and became visible from the HPWREN cameras. See [references](ecr-meta/ecr-science-description.md)

The rest of the README provides instructions to run the model on different environments.

To re-train the edge model with your own data, please see [trainning README](training/README.md).

The [Dockerfile](Dockerfile) already downloads the most recent versions of both the binary classifier model and Smokeynet (during the build stage) that can be ran on an edge device or locally and for user-convenience to test the models.
## Instructions

## Step 1: Build Docker image for plugin

Build image with buildx:
```
docker build -t sagecontinuum/sage-smoke-detection:0.1.0 --platform linux/amd64,linux/arm64 .
```

Build image without buildx:
```
docker build -t sagecontinuum/sage-smoke-detection:0.1.0 .
```

## Step 2: Run Docker container locally or on an edge device
There are three possible camera inputs and two smoke detector models to configure the plugin and to run the Docker container through command line arguments.

To get help with how to set the command line arguments:
```
docker run sagecontinuum/sage-smoke-detection:0.1.0 --help
```
Output:
```
usage: main.py [-h] [-st smoke_threshold] [-c camera_endpoint]
               [-ct camera_type] [-hcid hpwren_camera_id]
               [-hsid hpwren_site_id] [-delay smokeynet_delay]
               [-sdt sage_data_topic] [-mf model_file_name] [-mt model_type]

Smoke Detector Plugin

optional arguments:
  -h, --help            show this help message and exit
  -st smoke_threshold, --smoke_threshold smoke_threshold
                        Threshold for model inference (only used for binary
                        classifier) (default: 0.9)
  -c camera_endpoint, --camera camera_endpoint
                        Camera endpoint connected to the edge device.
                        (default: None)
  -ct camera_type, --camera-type camera_type
                        Camera type (default: mp4)
  -hcid hpwren_camera_id, --hpwren-camera-id hpwren_camera_id
                        Camera ID for HPWREN. Optional if HPWREN camera API
                        endpoint is being used. (default: 0)
  -hsid hpwren_site_id, --hpwren-site-id hpwren_site_id
                        Site ID for HPWREN. Optional if HPWREN camera API
                        endpoint is being used. (default: 0)
  -delay smokeynet_delay, --smokeynet-delay smokeynet_delay
                        SmokeyNet time delay to get the next image from Camera
                        (seconds). Default is set to 60 secs due to HPWREN
                        FigLib Trainning Data (default: 60.0)
  -sdt sage_data_topic, --sage-data-topic sage_data_topic
                        Sage data topic (default: env.smoke.)
  -mf model_file_name, --model-file-name model_file_name
                        Model file name (default: model.onnx)
  -mt model_type, --model-type model_type
                        Edge model type (default: smokeynet)
```

Camera inputs:
- (Default) Pre-recorded video of a fire (taken from FigLib): `--camera-type mp4`
- HPWREN camera API: `--camera-type hpwren`
- Camera connected to an edge device and passed in as a command line argument: `--camera-type device`
    - Also must set the an endpoint that is reacheable on the node. RTSP example: `--camera rtsp://sage:rtspstream@127.0.0.1/axis-media/media.amp`

Smoke Detector Models:
- Smokeynet (set as default): `--model-file-name model.onnx` and `--model-type smokeynet`
- Binary classifier model: `--model-file-name model.tflite` and `--model-type binary-classifier`

For setting the sage data topic (`--sage-data-topic env.smoke.`):
- The name of the topic to push to [Sage Data Repository](https://sagecontinuum.org/docs/about/architecture) and become publicly accessible to users through the [Data API](https://sagecontinuum.org/docs/tutorials/accessing-data#data-api)

Lastly, there is one environment variables that could be set for running the container in debug mode or when not running on a Sage Node [Sage Platform](https://sagecontinuum.org/docs/about/overview):
- PYWAGGLE_LOG_DIR: temporary directory to output the [pywaggle](https://github.com/waggle-sensor/pywaggle) log files for debugging purposes. This is the same format used by the [Data API](https://sagecontinuum.org/docs/tutorials/accessing-data#data-api).

Run model:
```
docker run  -v ${PWD}/pywaggle-logs:/src/pywaggle-logs --env PYWAGGLE_LOG_DIR=pywaggle-logs sagecontinuum/sage-smoke-detection:0.1.0
```

Note that setting `-v ${PWD}/pywaggle-logs:/src/pywaggle-logs --env PYWAGGLE_LOG_DIR=pywaggle-logs` allows to show the written files by pywaggle when debugging.

For the case that it is not needed, simply run the container without the volume mount and the env var:
```
docker run sagecontinuum/sage-smoke-detection:0.1.0
```

Output when plugin is configured to run HPWREN camera API as a camera input:
```
Starting smoke detection inferencing
Get image from HPWREN Camera
Image url: http://hpwren.ucsd.edu/cameras/L/tje-1-mobo-c.jpg
Description:  Unknown direction Color Original
Perform an inference based on trainned model
Publish
.
.
.
Get image from HPWREN Camera
Image url: http://hpwren.ucsd.edu/cameras/L/tje-1-mobo-c.jpg
Description:  Unknown direction Color Original
Perform an inference based on trainned model
Publish
```

Example output of the plugin when the pre-recorded MP4 is used:
```
Starting smoke detection inferencing
Get image from 20190610-Pauma-bh-w-mobo-c.mp4
Image url: 20190610-Pauma-bh-w-mobo-c.mp4
Description: Pre-recorded video
Perform an inference based on trainned model
Publish
.
.
.
Get image from 20190610-Pauma-bh-w-mobo-c.mp4
Image url: 20190610-Pauma-bh-w-mobo-c.mp4
Description: Pre-recorded video
Perform an inference based on trainned model
Publish
```

Example output of the plugin when the bottom camera on the Wild Sage Node is used:
```
Coming Soon
```