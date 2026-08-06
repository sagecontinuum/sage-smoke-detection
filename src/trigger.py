import os
import requests

_AUTHURL = 'https://hotshot.sdsc.edu/pylaski/auth'
_PROXYURL = 'https://wifire-api-proxy.nrp-nautilus.io'
_HOOKURL = 'https://wifire-webhook.nrp-nautilus.io/webhook'

class Trigger:

    def __init__(self, authURL=_AUTHURL, proxyURL=_PROXYURL):
        self.authURL = authURL
        self.proxyURL = proxyURL
        self.statuses = []


    def predictSmoke(self,
                     model_type,
                     execute,
                     ):
        do_trigger = False

        if model_type == 'smokeynet':
            image_preds, _, _ = execute.inference_results
            do_trigger = any(image_preds)

        elif model_type == 'binary-classifier':
            raise NotImplementedError("Binary Classifier not implemented yet")

        else:
            raise ValueError(f"Invalid model type {model_type}")

        return do_trigger


    def trigger(self, site_lat_long):
        authToken = self.getAuthToken()
        headers = {
            'content-type': "application/json",
            'authorization': 'Bearer ' + authToken['token'],
        }
        for counter, latLong in enumerate(self.getEnsembleLatLongList(site_lat_long)):
            params = self.getDefaultParams()
            params['ignition']['point'] = latLong
            params['webhook']['request_id'] = str(counter)
            status = self.launchFarsiteModel(headers, params)
            self.statuses.append(status)


    def getAuthToken(self):
        userName = os.getenv('AUTHUSERNAME')
        password = os.getenv('AUTHPASSWORD')

        if not userName:
            raise EnvironmentError(f"Failed because AUTHUSERNAME is not set.")
        if not password:
            raise EnvironmentError(f"Failed because AUTHPASSWORD is not set.")
        
        authURL1 = self.authURL + '?user=' + userName + '&password=' + password

        r = requests.request('GET', authURL1)
        authToken = r.json()
        return authToken


    def getEnsembleLatLongList(self, site_lat_long, count=1, dx=0.01):
        returnList = []
        for i in range(count):
            returnList.append([
                site_lat_long[0] + dx*i,
                site_lat_long[1] + dx*i,
            ])
        return returnList


    def launchFarsiteModel(self, headers, params):
        r = requests.request('POST', self.proxyURL, headers=headers, json=params)
        return r.text


    def getDefaultParams(self):
        defaultParams = {
            "hours": 2,
            "ember": 80,
            "ignition": {
                "point": [32.848132, -116.805901]
            },
            "weather": {
                "repeat_values": {
                    "wind_direction": 180,
                    "wind_speed": 5,
                    "relative_humidity": 5,
                    "temperature": 70
                }
            },
            "fuel_moisture": {
                "one_hr": 3,
                "ten_hr": 4,
                "hundred_hr": 6,
                "live_herb": 5,
                "live_woody": 60
            },
            "webhook": {
                "url": _HOOKURL,
                "request_id": "0"
            }
        }
        return defaultParams

