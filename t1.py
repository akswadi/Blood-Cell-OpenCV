import os
from roboflow import Roboflow
rf = Roboflow(api_key="M87FC8yaCJLcfz5vxrPK")
project = rf.workspace("arin-s").project("wbc-and-rcb")
model = project.version(1).model
# Define input and output folders
input_folder = "Images"
output_folder = "Results_WBC_ML"
os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        
        input_image_path = os.path.join(input_folder, filename)
        
        
        output_prediction_path = os.path.join(output_folder, f"prediction_{filename}")
        prediction = model.predict(input_image_path, confidence=21, overlap=52)
        prediction.save(output_prediction_path)
        
        print(f"Prediction saved for {filename}")

print("Prediction process completed.")