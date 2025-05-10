import streamlit as st
import os
import tempfile
from modules import eye_contact, behavior_detect, gesture_delay, social_reciprocity

st.title("👶 AI-Based Early Autism Detector")
st.write("Upload a video of a toddler during interaction. The system will analyze behavior to assess autism risk.")

uploaded_file = st.file_uploader("Upload a video (.mp4)", type=["mp4", "mov"])

if uploaded_file:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🧠 Analyze Video"):
        with st.spinner("Analyzing behavior..."):

            # Run analysis modules
            eye_score = eye_contact.analyze(video_path)
            rep_score = behavior_detect.analyze(video_path)
            gesture_score = gesture_delay.analyze(video_path)
            social_score = social_reciprocity.analyze(video_path)

            # Risk Flags
            eye_flag     = 1 if eye_score < 0.2 else 0
            rep_flag     = 1 if rep_score > 0 else 0
            gesture_flag = 1 if gesture_score < 0.01 else 0
            social_flag  = 1 if social_score < 0.3 else 0

            # Weighted risk score
            weights = {'eye': 0.3, 'rep': 0.25, 'gesture': 0.2, 'social': 0.25}
            risk_score = (
                eye_flag     * weights['eye'] +
                rep_flag     * weights['rep'] +
                gesture_flag * weights['gesture'] +
                social_flag  * weights['social']
            )

        # Display Results
        st.subheader("🧾 Results")

        st.markdown(f"**Eye Contact Score:** `{eye_score:.2f}` {'🟥 LOW' if eye_flag else '🟩 OK'}")
        st.markdown(f"**Repetitive Movements:** `{rep_score:.2f}` {'🟥 HIGH' if rep_flag else '🟩 LOW'}")
        st.markdown(f"**Gesture Frequency:** `{gesture_score:.3f}` {'🟥 LOW' if gesture_flag else '🟩 OK'}")
        st.markdown(f"**Social Response:** `{social_score:.2f}` {'🟥 LOW' if social_flag else '🟩 OK'}")

        st.markdown("---")
        st.subheader("🧠 Final Risk Assessment")

        if risk_score >= 0.5:
            st.error(f"⚠️ High Autism Risk Detected (Score: {risk_score:.2f})")
        else:
            st.success(f"✅ Low Autism Risk (Score: {risk_score:.2f})")
