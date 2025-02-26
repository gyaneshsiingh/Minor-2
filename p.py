
import streamlit as st
import pandas as pd
import os
import numpy as np
import ffmpeg
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import uuid
import tempfile

results_file = "results.csv"
model_file = "model.pkl"
weight_file = "weights.pkl"
model = None
weights = None
threshold = 30 


point1 = (846, 577)
point2 = (850, 577)
point3 = (845, 619)
point4 = (860, 633)

x_values = [point1[0], point2[0], point3[0], point4[0]]
y_values = [point1[1], point2[1], point3[1], point4[1]]
x_min, x_max = min(x_values), max(x_values)
y_min, y_max = min(y_values), max(y_values)

def train_model():
    global model
    df = pd.read_csv('nm RGB.csv')
    x = df.drop(columns=['nm'], axis=1)
    y = df['nm']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

def load_model():
    global model
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        train_model()

def load_weights():
    global weights
    if os.path.exists(weight_file):
        with open(weight_file, 'rb') as f:
            weights = pickle.load(f)
    else:
        weights = np.ones((4, 4))

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    (
        ffmpeg.input(video_path)
        .output(f"{output_folder}/frame_%04d.jpg", r=60, format='image2', vcodec='mjpeg')
        .run(quiet=True, overwrite_output=True)
    )

def get_average_color_from_frames(folder_path):
    total_color = np.array([0, 0, 0], dtype=np.float64)
    frame_count = 0
    rolling_avg = None

    load_weights()
    total_weight = np.sum(weights)
    csv_data = []
    
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".jpg"):
            image = Image.open(os.path.join(folder_path, file))
            image = image.crop((x_min, y_min, x_max, y_max))
            width, height = image.size
            grid_size_x = max(2, width // 50)
            grid_size_y = max(2, height // 50)
            square_width = width // grid_size_x
            square_height = height // grid_size_y

            weighted_color = np.array([0, 0, 0], dtype=np.float64)
            for i in range(grid_size_x):
                for j in range(grid_size_y):
                    weight = weights[i % weights.shape[0]][j % weights.shape[1]]
                    if weight == 0:
                        continue
                    left = i * square_width
                    upper = j * square_height
                    right = min(left + square_width, width)
                    lower = min(upper + square_height, height)
                    small_region = image.crop((left, upper, right, lower))
                    avg_color = np.array(small_region.resize((1, 1)).getpixel((0, 0)))
                    weighted_color += avg_color * weight
            
            frame_avg_color = weighted_color / total_weight
            if rolling_avg is not None:
                diff = np.linalg.norm(frame_avg_color - rolling_avg)
                if diff > threshold:
                    continue  
            rolling_avg = frame_avg_color 
            total_color += frame_avg_color
            frame_count += 1
            csv_data.append([frame_avg_color[0], frame_avg_color[1], frame_avg_color[2]])
    
    df = pd.DataFrame(csv_data, columns=["Red", "Green", "Blue"])
    df.to_csv("frame_colors.csv", index=False)
    return (total_color / frame_count).astype(int) if frame_count > 0 else None

def predict_nm_from_rgb(rgb_color):
    predicted_nm = model.predict([rgb_color])
    return predicted_nm[0]

def save_result(unique_id, rgb_color, nm_value):
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("ID,Red,Green,Blue,Predicted Wavelength\n")
    with open(results_file, 'a') as f:
        f.write(f"{unique_id},{rgb_color[0]},{rgb_color[1]},{rgb_color[2]},{nm_value}\n")

def update_weights(region, significance):
    global weights
    weights[region] += significance
    weights /= np.max(weights)
    with open(weight_file, 'wb') as f:
        pickle.dump(weights, f)

def view_results():
    st.title("View Results")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        st.write("Details of results:")
        st.dataframe(df)
    else:
        st.warning("No results found.")

def main():
    st.set_page_config(page_title="Predict Wavelength from Video", page_icon=":movie_camera:", layout="centered")
    load_model()
    menu = ["Predict Wavelength from Video", "View Previous Results"]
    choice = st.sidebar.selectbox("Select Option", menu)
    
    if choice == "Predict Wavelength from Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            st.write("Processing your video...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            
            output_folder = "extracted_frames"
            extract_frames(temp_file_path, output_folder)
            average_color = get_average_color_from_frames(output_folder)
            os.remove(temp_file_path)
            
            if average_color is None:
                st.error("No frames extracted. Unable to process video.")
            else:
                nm_value = predict_nm_from_rgb(average_color)
                unique_id = str(uuid.uuid4())
                save_result(unique_id, average_color, nm_value)
                
                st.subheader("Predicted Results:")
                st.write(f"ID: {unique_id}")
                st.write(f"Predicted Wavelength (nm): {nm_value}")
                st.write(f"RGB Color: {average_color}")
    
    elif choice == "View Previous Results":
        view_results()

if __name__ == "__main__":
    main()
