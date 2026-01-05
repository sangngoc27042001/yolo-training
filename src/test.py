from roboflow import Roboflow
rf = Roboflow(api_key="dV9Gb4PSK0SwRh4ljXab")
project = rf.workspace("ehjun-7e7tb").project("phone-qywmy")
version = project.version(1)
dataset = version.download("yolov11")
                