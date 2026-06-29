import argparse
import configure
import logging
import os,sys
import publisher

parser = argparse.ArgumentParser(description='Smoke Detector Plugin',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-st',
                        '--smoke_threshold',
                        metavar='smoke_threshold',
                        type=float,
                        default=0.9,
                        help='Threshold for model inference'
                    )

parser.add_argument('-c',
                        '--camera',
                        metavar='camera_endpoint',
                        type=str,
                        required=False,
                        help='Camera endpoint connected to the edge device.'
                    )

parser.add_argument('-ct',
                        '--camera-type',
                        metavar='camera_type',
                        type=str,
                        default='mp4',
                        choices=['mp4', 'device', 'hpwren'],
                        help='Camera type'
                    )

parser.add_argument('-hcid',
                        '--hpwren-camera-id',
                        metavar='hpwren_camera_id',
                        type=int,
                        default=0,
                        help='Camera ID for HPWREN. Optional if HPWREN camera API endpoint is being used.'
                    )

parser.add_argument('-hsid',
                        '--hpwren-site-id',
                        metavar='hpwren_site_id',
                        type=int,
                        default=0,
                        help='Site ID for HPWREN. Optional if HPWREN camera API endpoint is being used.'
                    )

parser.add_argument('-delay',
                        '--smokeynet-delay',
                        metavar='smokeynet_delay',
                        type=float,
                        default=60.0,
                        help='SmokeyNet delay to get the next image from Camera. (seconds) for real-time video and (frames) for recorded video. Default is set to 60 due to HPWREN FigLib Trainning Data'
                    )

parser.add_argument('-sdt',
                        '--sage-data-topic',
                        metavar='sage_data_topic',
                        type=str,
                        default='env.smoke.',
                        help='Sage data topic'
                    )

parser.add_argument('-mf',
                        '--model-file-name',
                        metavar='model_file_name',
                        type=str,
                        default='model.onnx',
                        help='Model file name'
                    )

parser.add_argument('-mt',
                        '--model-type',
                        metavar='model_type',
                        type=str,
                        default='smokeynet',
                        choices=['smokeynet'],
                        help='Edge model type'
                    )

args = parser.parse_args()

smoke_threshold=args.smoke_threshold
camera_endpoint=args.camera
hpwren_site_id = args.hpwren_site_id
hpwren_camera_id = args.hpwren_camera_id
smokeynet_delay = args.smokeynet_delay

sage_data_topic = args.sage_data_topic
model_file = args.model_file_name
model_abs_path = os.path.abspath(model_file)
model_type = args.model_type
camera_type = args.camera_type

FORMAT = "[%(asctime)s %(filename)s:%(lineno)s]%(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="%Y/%m/%d %H:%M:%S",
)

if camera_type == 'mp4':
    camera_device = configure.RecordedMP4()
elif camera_type == 'device':
    if camera_endpoint is None:
        print(f'No camera device specified. Exiting...')
        exit(1)
    camera_device = configure.CameraDevice(camera_endpoint)
elif camera_type == 'hpwren':
    camera_device = configure.Hpwren(hpwren_camera_id,hpwren_site_id)
else:
    logging.error(f'Error: not supported case for CAMERA_TYPE: {camera_type}.')
    sys.exit()

camera_meta = camera_device.get_metadata()
camera_src = camera_meta['camera_src']
server_name = camera_meta['server_name']
image_url = camera_meta['image_url']
description = camera_meta['description']

logging.info(f'Starting smoke detection inferencing')
logging.info(f'Get image from {server_name}')
logging.info(f'Image url: {image_url}')
logging.info(f'Description: {description}')
logging.info(f'Using {model_type}')

logging.info('Perform an inference based on trainned model')
execute = configure.ExecuteBase(model_abs_path,model_type,camera_device,smokeynet_delay)
execute.run(smoke_threshold)

logging.info('Publish')
publisher_waggle = publisher.PublisherWaggle(model_type,execute)
publisher_waggle.publish(sage_data_topic,smoke_threshold,camera_src)
