import streamlit as st
from PIL import Image, ExifTags
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

# Set page config
st.set_page_config(
    page_title="Accident Detection",
    page_icon="🚨",
    layout="centered"
)

def get_image_location(img):
    """Extract GPS coordinates from image EXIF data"""
    try:
        exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() 
               if k in ExifTags.TAGS and isinstance(v, (bytes, str, int, float))}
        
        if 'GPSInfo' in exif:
            gps_info = exif['GPSInfo']
            gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
            
            def convert(coord):
                degrees = coord[0]
                minutes = coord[1] / 60.0
                seconds = coord[2] / 3600.0
                return degrees + minutes + seconds
            
            lat = convert(gps.get('GPSLatitude', (0, 0, 0)))
            lon = convert(gps.get('GPSLongitude', (0, 0, 0)))
            
            if gps.get('GPSLatitudeRef') == 'S':
                lat = -lat
            if gps.get('GPSLongitudeRef') == 'W':
                lon = -lon
            
            return (lat, lon), f"Image Location: {lat:.4f}, {lon:.4f}"
    except Exception as e:
        st.error(f"Error extracting GPS data: {str(e)}")
    return None, "No location data in image"

@st.cache_resource
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Using device: {device}")
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").to(device)
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        st.write("Model loaded successfully!")
        return model, processor, device
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None

def classify_scene(predictions, confidences):
    """Classify scene into Accident, Normal Traffic, or Non-Accident"""
    accident_phrases = [
        "crash", "collision", "accident", "damaged", 
        "overturned", "emergency", "fire", "injury",
        "wreck", "smoke", "flames", "responders",
        "pileup", "hit", "spilling", "burning"
    ]
    
    normal_traffic_phrases = [
        "normal traffic", "moving vehicles", "clear road",
        "flowing traffic", "orderly traffic", "no incidents"
    ]
    
    accident_score = 0
    normal_score = 0
    
    for pred, conf in zip(predictions, confidences):
        pred_lower = pred.lower()
        accident_score += sum(
            conf/100 for phrase in accident_phrases 
            if phrase in pred_lower
        )
        normal_score += sum(
            conf/100 for phrase in normal_traffic_phrases
            if phrase in pred_lower
        )
    
    if accident_score >= 1.5:
        return "Accident", accident_score, normal_score
    elif normal_score >= 2.0:
        return "Normal Traffic", accident_score, normal_score
    elif accident_score > normal_score:
        return "Possible Accident", accident_score, normal_score
    else:
        return "Non-Accident Scene", accident_score, normal_score

def main():
    st.title("🚦 Traffic Scene Classifier")
    st.markdown("Upload an image to detect accidents or normal traffic scenes or non accident")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            return

        coords, location_str = get_image_location(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.caption(location_str)

        with col2:
            if st.button("Analyze Scene", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        model, processor, device = load_model()
                        if model is None or processor is None:
                            return
                        
                        text_labels = [
                            # Accident scenarios
                            "severe car crash with debris",
                            "vehicle collision with damaged cars",
                            "traffic accident with injuries",
                            "overturned truck on highway",
                            "burning vehicle with flames",  
                            "Bad car crash with broken pieces everywhere",
                            "Two cars hit each other and got smashed",
                            "Car accident where people got hurt",
                            "Truck flipped over on the road",
                            "Car on fire with big flames",
                            # Normal traffic scenarios
                            "Smooth traffic on highway",
                            "Clear road with cars moving", 
                            "Neat traffic at crossing",
                            "Easy traffic, no problems",
                            "Calm street with parked cars",
                            "Cars moving smoothly on road",
                            "No traffic jams, all clear"
                        ]
                        
                        inputs = processor(
                            text=text_labels,
                            images=img,
                            return_tensors="pt",
                            padding=True
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

                        top5_idx = np.argsort(probs[0])[-5:][::-1]
                        top5_confidences = probs[0][top5_idx] * 100
                        top5_predictions = [text_labels[i] for i in top5_idx]

                        classification, accident_score, normal_score = classify_scene(
                            top5_predictions, top5_confidences
                        )

                        st.subheader("Analysis Results")
                        
                        if classification == "Accident":
                            st.error(f"🚨 ACCIDENT DETECTED")
                            st.write(f"Confidence: {accident_score:.2f}")
                            st.markdown("**Top indicators:**")
                            for i, (pred, conf) in enumerate(zip(top5_predictions, top5_confidences)):
                                st.write(f"{i+1}. {pred} ({conf:.1f}%)")
                        
                        elif classification == "Normal Traffic":
                            st.success(f"✅ NORMAL TRAFFIC")
                            st.write(f"Confidence: {normal_score:.2f}")
                            st.markdown("**Scene indicators:**")
                            for i, (pred, conf) in enumerate(zip(top5_predictions[:3], top5_confidences[:3])):
                                st.write(f"{i+1}. {pred} ({conf:.1f}%)")
                        
                        else:
                            st.info(f"ℹ️ {classification.upper()}")
                            st.write(f"Accident score: {accident_score:.2f} | Normal score: {normal_score:.2f}")
                            st.markdown("**Key observations:**")
                            for i, (pred, conf) in enumerate(zip(top5_predictions[:3], top5_confidences[:3])):
                                st.write(f"{i+1}. {pred} ({conf:.1f}%)")

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()