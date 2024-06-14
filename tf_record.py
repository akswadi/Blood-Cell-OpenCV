from roboflow import Roboflow
rf = Roboflow(api_key="oZaEzhFkJIWcPCpAmVsH")
project = rf.workspace("arin-swadi-0xhey").project("blood-fdwzc")
version = project.version(2)
dataset = version.download("tfrecord")
