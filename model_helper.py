import numpy as np
import joblib
import json
from keras.models import load_model  # For loading Keras models

import joblib
def load_model_and_preprocessors(path,model_name):
    preprocessor_path=path+"/preprocessor/"
    model_path=path+"/neural_nets/{}_model.h5".format(model_name)
    model = load_model(model_path)
    scaler=joblib.load(preprocessor_path+"scaler.pkl")
    with open(preprocessor_path+"model_paths.json", "r") as file:
        label_dict = json.load(file) 
    for name,file_name in label_dict.items():
        label_dict[name]=joblib.load(preprocessor_path+file_name)
    mlb=joblib.load(preprocessor_path+"mlb_"+model_name+".pkl")

    #mlb = joblib.load(model_path + "_mlb.pkl")
    #scaler = joblib.load(model_path + "_scaler.pkl")
    return model, mlb, scaler,label_dict




def get_top_recommendations(predictions, mlb, top_n=3):
    recommended_items = []
    for pred in predictions:
        top_indices = np.argsort(pred)[-top_n:][::-1]  # Get indices of top N values
        top_items = [mlb.classes_[i] for i in top_indices if pred[i] > 0.5]  # Include only if above threshold
        recommended_items.append(top_items)
    return recommended_items
def predict_with_model(model, scaler, mlb, input_data,label_encoder, top_n=3):
    scaler_data=np.asarray([input_data[0][:4]])
    input_data_scaled = scaler.transform(scaler_data)
    label_data=input_data[0][4:]
    label_data_encoded=[]
    count=0
    for key,value in label_encoder.items():
        data_point=np.asarray([label_data[count]])
        label_data_encoded.append(value.transform(data_point))
        count=count+1
    label_data_encoded=[i[0] for i in label_data_encoded]
    ids=list(input_data_scaled[0])
    ids.extend(label_data_encoded)
    final_data=np.asarray(ids)
    final_data=final_data.reshape(1,10)
    print("****")
    print(final_data)
    predictions = model.predict(final_data)
    return get_top_recommendations(predictions, mlb, top_n)