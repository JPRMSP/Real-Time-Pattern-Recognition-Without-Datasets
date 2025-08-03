import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------------
# Feature extraction function
# -------------------------------
def extract_features(img):
    img = cv2.resize(img, (50, 50))  # Normalize size
    
    # Statistical features
    mean_val = np.mean(img)
    var_val = np.var(img)
    
    # Entropy
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    hist_norm = hist_norm[np.nonzero(hist_norm)]
    entropy = -np.sum(hist_norm*np.log2(hist_norm))
    
    # Edge count
    edges = cv2.Canny(img, 100, 200)
    edge_count = np.sum(edges > 0)
    
    return [mean_val, var_val, entropy, edge_count]

# -------------------------------
# App settings
# -------------------------------
st.set_page_config(page_title="Real-Time Pattern Recognition", layout="wide")
st.title("🎨 Real-Time Pattern Recognition Without Datasets")
st.write("Draw symbols, train the system in real-time, and classify without any pre-trained models or datasets.")

# -------------------------------
# Global variables
# -------------------------------
if "training_data" not in st.session_state:
    st.session_state.training_data = []  # List of (features, label)

# -------------------------------
# Drawing Canvas
# -------------------------------
st.subheader("Draw Your Pattern")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=5,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

# -------------------------------
# Training sample saving
# -------------------------------
st.subheader("Training Mode")
label_input = st.text_input("Enter label for your drawing")

if st.button("Save as Training Sample"):
    if canvas_result.image_data is not None and label_input.strip():
        img = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
        features = extract_features(img)
        st.session_state.training_data.append((features, label_input.strip()))
        st.success(f"✅ Saved sample for class: {label_input.strip()}")

# -------------------------------
# Classification
# -------------------------------
st.subheader("Classification Mode")
if st.button("Classify Drawing"):
    if canvas_result.image_data is not None and st.session_state.training_data:
        img = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)
        test_features = extract_features(img)
        
        min_dist = float("inf")
        predicted_label = None
        for features, label in st.session_state.training_data:
            dist = np.linalg.norm(np.array(test_features) - np.array(features))
            if dist < min_dist:
                min_dist = dist
                predicted_label = label
        
        st.info(f"Predicted Class: **{predicted_label}**")
    else:
        st.warning("No training samples available. Please add some first.")

# -------------------------------
# Clustering (Unsupervised Learning)
# -------------------------------
st.subheader("Unsupervised Clustering Mode")
if st.button("Run Clustering"):
    if st.session_state.training_data:
        X = [f for f, _ in st.session_state.training_data]
        
        if len(X) < 2:
            st.warning("⚠️ Need at least 2 samples to perform clustering.")
        else:
            n_clusters = min(2, len(X))  # Auto adjust cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            st.write("Cluster Assignments:", kmeans.labels_)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X)
            plt.figure()
            plt.scatter(reduced[:,0], reduced[:,1], c=kmeans.labels_, cmap='rainbow')
            plt.title("Feature Space Clustering")
            st.pyplot(plt)
    else:
        st.warning("No samples to cluster. Please add some first.")

# -------------------------------
# Show saved training samples
# -------------------------------
if st.checkbox("Show Stored Training Samples"):
    if st.session_state.training_data:
        st.write("Number of samples:", len(st.session_state.training_data))
        for i, (features, label) in enumerate(st.session_state.training_data, 1):
            st.write(f"{i}. Label: {label}, Features: {np.round(features, 2)}")
    else:
        st.info("No training samples yet.")
