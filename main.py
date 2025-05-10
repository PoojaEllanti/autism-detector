from modules import eye_contact, behavior_detect, gesture_delay, social_reciprocity

video_path = "test_video.mp4"

# --- Run each module ---
eye_score = eye_contact.analyze(video_path)
rep_score = behavior_detect.analyze(video_path)
gesture_score = gesture_delay.analyze(video_path)
social_score = social_reciprocity.analyze(video_path)

# --- Print individual results ---
print("\n========== Autism Risk Module Outputs ==========")
print(f"Eye Contact Score         : {eye_score:.2f}  → {'LOW' if eye_score < 0.2 else 'OK'}")
print(f"Repetitive Movement Score : {rep_score:.2f}  → {'HIGH' if rep_score > 0 else 'LOW'}")
print(f"Gesture Rate              : {gesture_score:.3f}  → {'LOW' if gesture_score < 0.01 else 'OK'}")
print(f"Social Response Score     : {social_score:.2f}  → {'LOW' if social_score < 0.3 else 'OK'}")

# --- Convert to Risk Flags (1 = risk detected) ---
eye_flag     = 1 if eye_score < 0.2 else 0
rep_flag     = 1 if rep_score > 0 else 0
gesture_flag = 1 if gesture_score < 0.01 else 0
social_flag  = 1 if social_score < 0.3 else 0

# --- Weighted Risk Score ---
# You can adjust weights based on priority
weights = {
    'eye': 0.3,
    'rep': 0.25,
    'gesture': 0.2,
    'social': 0.25
}

risk_score = (
    eye_flag     * weights['eye'] +
    rep_flag     * weights['rep'] +
    gesture_flag * weights['gesture'] +
    social_flag  * weights['social']
)

print("\n========== Final Autism Risk Assessment ==========")
if risk_score >= 0.5:
    print("⚠️  Possible Early Signs of Autism Detected")
else:
    print("✅ Low Autism Risk Based on Observed Behavior")
print(f"Overall Risk Score: {risk_score:.2f}")
