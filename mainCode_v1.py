import cv2
import json
from watson_developer_cloud import VisualRecognitionV3
import datetime
import ibm_boto3
from ibm_botocore.client import Config, ClientError
import numpy as np
import sys
import ibmiotf.application
import ibmiotf.device
import random
import time
import sys
from glob import glob
import itertools as it

from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey

#IBM Watson Device Credentials 
organization = "oe4tp9"
deviceType = "iotDeviceType"
deviceId= "myDevice1"
authMethod="token"
authToken="gZVeccyM-4oV0Qf5H_"

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='SGKrtcybX-vr71npQCXNuH97XWOsDeOn-tBDUHJojh0j')

def myCommandCallback(cmd):
        print("Command received: %s" % cmd.data)
        print(cmd.data['command'])

try:
	deviceOptions = {"org": organization, "type": deviceType, "id": deviceId, "auth-method": authMethod, "auth-token": authToken}
	deviceCli = ibmiotf.device.Client(deviceOptions)
	#..............................................
	
except Exception as e:
	print("Caught exception connecting device: %s" % str(e))
	sys.exit()
	
# Connect and send a datapoint "hello" with value "world" into the cloud as an event of type "greeting" 10 times
deviceCli.connect()
body_classifier=cv2.CascadeClassifier("haarcascade_fullbody.xml")

#ReadVideo/Image
video=cv2.VideoCapture("video.mp4") #(0)

COS_ENDPOINT = "https://s3.jp-tok.cloud-object-storage.appdomain.cloud"
COS_API_KEY_ID = "s374zVcUCL-zMTdp6AJWJdcJctI0wRKnlcY3AHgBXn9E"
COS_AUTH_ENDPOINT = "https://iam.cloud.ibm.com/identity/token"
COS_RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/5307d6d6c4614d03bb91cfdacac6e18c:0d14fcf6-463d-4886-ac2a-11212d24d62d::"

client = Cloudant("5b1fb4fb-0a77-4eb8-8a7a-2fd49fbcab7c-bluemix", "fc03e8f1a465a14e2cfc4bf86feb9ab1c15018bf6468f65a756b39c4bc60d2c2", url="https://5b1fb4fb-0a77-4eb8-8a7a-2fd49fbcab7c-bluemix:fc03e8f1a465a14e2cfc4bf86feb9ab1c15018bf6468f65a756b39c4bc60d2c2@5b1fb4fb-0a77-4eb8-8a7a-2fd49fbcab7c-bluemix.cloudantnosqldb.appdomain.cloud")
client.connect()
database_name = "security_images"

# Create resource
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh
    
def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        
def multi_part_upload(bucket_name, item_name, file_path):
    try:
        print("Starting file transfer for {0} to bucket: {1}\n".format(item_name, bucket_name))
        # set 5 MB chunks
        part_size = 1024 * 1024 * 5

        # set threadhold to 15 MB
        file_threshold = 1024 * 1024 * 15

        # set the transfer threshold and chunk size
        transfer_config = ibm_boto3.s3.transfer.TransferConfig(
            multipart_threshold=file_threshold,
            multipart_chunksize=part_size
        )

        # the upload_fileobj method will automatically execute a multi-part upload
        # in 5 MB chunks for all files over 15 MB
        with open(file_path, "rb") as file_data:
            cos.Object(bucket_name, item_name).upload_fileobj(
                Fileobj=file_data,
                Config=transfer_config
            )

        print("Transfer for {0} Complete!\n".format(item_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to complete multi-part upload: {0}".format(e))
               
while True:
    #capture the first frame
    _, frame=video.read()
    cv2.imwrite('frame.jpg',frame)
    #
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    
    #Setting File Name for Image
    picname=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    picname=picname+".jpg"
    pic=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    default = [cv2.samples.findFile('frame.jpg')] if len(sys.argv[1:]) == 0 else []
    for fn in it.chain(*map(glob, default + sys.argv[1:])):
        print(fn, ' - ',)
        try:
            img = cv2.imread(fn)
            if img is None:
                print('Failed to load image file:', fn)
                continue
        except:
            print('loading error')
            continue

        found, _w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        #draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found)))
        #cv2.imshow('Body Detection', img)
    
    cv2.imwrite("LocalTempDB/"+picname,img)
    
    with open("LocalTempDB/"+picname, 'rb') as images_file:
        classes = visual_recognition.classify(
            images_file,
            threshold='0.4',
        classifier_ids='DefaultCustomModel_2072297414').get_result()
    #print(json.dumps(classes, indent=2))
    imagesList = classes["images"]       #classes : DICT
    #classifiersDict = imagesList[0]      #imagesList : LIST
    #c = classifiersDict["classifiers"]   #classifiersDict : DICT
    #d = c[0]                    #c : LIST
                                #d : DICT
    print("==============================")
    print(type(imagesList))
    #print(d["classes"])
    print("==============================")
    
    my_database = client.create_database(database_name)        
    multi_part_upload("temp-bucket-vgg",picname,"LocalTempDB/"+picname)
    if my_database.exists():
        print("'{database_name}' successfully created.")
        json_document = {
                "_id": pic,
                "link":COS_ENDPOINT+"/temp-bucket-vgg/"+picname,
                "code":2
                }
        new_document = my_database.create_document(json_document)
        if new_document.exists():
            print("Document '{new_document}' successfully created.")
    time.sleep(1)
    #print data
    def myOnPublishCallback():
        print ("Published data to IBM Watson")

    success = deviceCli.publishEvent("Data", "json", json_document, qos=0, on_publish=myOnPublishCallback)
    if not success:
        print("Not connected to IoTF")
    time.sleep(1)
    deviceCli.commandCallback = myCommandCallback
    #waitKey(1)- for every 1 millisecond new frame will be captured
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
#release the camera
video.release()
#destroy all windows
cv2.destroyAllWindows()
deviceCli.disconnect()
    



