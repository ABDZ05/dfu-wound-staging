"""Model + analysis core for the DFU wound-staging prototype (same logic as the Kaggle notebook)."""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFilter, ImageEnhance

device = torch.device("cpu")

# MobileNetV3-Large with a single-logit head (sigmoid = probability of ulcer), as trained in the notebook
live_model = models.mobilenet_v3_large(weights=None)
live_model.classifier[3] = nn.Linear(live_model.classifier[3].in_features, 1)
live_model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "JSED_MobileNetV3_DFU.pth"), map_location=device))
live_model.eval()

infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_clinical_wound_biomarkers(rgb_img, cam_mask):
    """
    1. Isolates the ulcer bed via AI Grad-CAM activation.
    2. Segments Red (Granulation), Yellow (Slough), and Black (Necrotic Eschar) tissue.
    3. Analyzes the periwound halo (ring around the ulcer) for erythema/inflammation.
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Define Ulcer Bed (high AI activation)
    wound_mask = (cam_mask > 0.38).astype(np.uint8)
    if np.sum(wound_mask) < 25:
        wound_mask = (cam_mask > 0.20).astype(np.uint8)
        
    # Define Periwound Halo (15-pixel ring immediately surrounding the ulcer)
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(wound_mask, kernel, iterations=1)
    periwound_ring = (dilated_mask == 1) & (wound_mask == 0)
    
    w_bool = wound_mask == 1
    total_px = max(np.sum(w_bool), 1)
    area_pct = (total_px / (224.0 * 224.0)) * 100.0
    
    # Tissue Classification inside Ulcer Bed (RYB Clinical System)
    necrotic = w_bool & (v < 70) # Dark/devitalized eschar
    slough   = w_bool & (~necrotic) & (((h >= 12) & (h <= 48) & (s > 30)) | ((s < 60) & (v > 135)))
    granul   = w_bool & (~necrotic) & (~slough)
    
    nec_pct  = (np.sum(necrotic) / total_px) * 100.0
    slo_pct  = (np.sum(slough) / total_px) * 100.0
    gra_pct  = (np.sum(granul) / total_px) * 100.0
    
    # Periwound Erythema Index (Redness saturation in periwound ring vs background skin)
    red_hue = ((h <= 12) | (h >= 165)) & (s > 55) & (v > 60)
    if np.sum(periwound_ring) > 0:
        peri_erythema_score = (np.sum(periwound_ring & red_hue) / np.sum(periwound_ring)) * 100.0
    else:
        peri_erythema_score = 0.0
        
    # Create Visual Tissue Segmentation Map for the Doctor
    # Red = Granulation, Yellow = Slough, Purple/Black = Necrosis, Cyan Ring = Periwound Halo
    seg_vis = rgb_img.copy()
    overlay = rgb_img.copy()
    overlay[granul]   = [220, 40, 40]   # Bright Red: Healthy Granulation
    overlay[slough]   = [245, 215, 30]  # Yellow: Fibrinous Slough
    overlay[necrotic] = [40, 20, 60]    # Dark Purple/Black: Necrotic Eschar
    
    seg_vis = cv2.addWeighted(seg_vis, 0.50, overlay, 0.50, 0)
    # Draw crisp contour around the ulcer margin
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_vis, contours, -1, (0, 255, 255), 2)
    
    return {
        "Area_Pct": area_pct,
        "Granulation_Pct": gra_pct,
        "Slough_Pct": slo_pct,
        "Necrotic_Pct": nec_pct,
        "Periwound_Erythema": peri_erythema_score,
        "Seg_Map": seg_vis
    }

def digital_podiatry_consult(image, site, ischemia, neuropathy, depth, tele_sim):
    if image is None:
        return None, None, {}, "Please upload a foot photograph."
        
    pil_img = Image.fromarray(image).convert("RGB").resize((224, 224))
    if tele_sim:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.8))
        pil_img = ImageEnhance.Brightness(pil_img).enhance(0.55)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(0.55)
        
    img_t = infer_tf(pil_img).unsqueeze(0).to(device)
    img_t.requires_grad = True
    
    feats, grads = [], []
    h1 = live_model.features[-1].register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = live_model.features[-1].register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    live_model.zero_grad()
    out = live_model(img_t)
    prob = torch.sigmoid(out).item()
    out.backward()
    h1.remove(); h2.remove()
    
    w = np.mean(grads[0].cpu().data.numpy()[0], axis=(1, 2))
    f = feats[0].cpu().data.numpy()[0]
    cam = np.maximum(np.sum(w[:, None, None] * f, axis=0), 0)
    cam = cv2.resize(cam, (224, 224))
    if np.max(cam) > 0: cam /= np.max(cam)
    
    orig_np = np.array(pil_img)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    gradcam_img = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
    
    if prob < 0.50:
        return gradcam_img, orig_np, {"Intact Diabetic Skin": float(1-prob), "Active Ulcer": float(prob)}, (
            "🟢 INTACT SKIN (No Active Ulceration Detected)\n"
            "• Action: Routine IWGDF preventative foot screening every 3–6 months, daily self-inspection, and therapeutic footwear."
        )
        
    bio = extract_clinical_wound_biomarkers(orig_np, cam)
    devitalized_pct = bio["Slough_Pct"] + bio["Necrotic_Pct"]
    
    # Calculate Official IWGDF SINBAD Score (0 to 6)
    s_site = 1 if "Midfoot/Hindfoot" in site else 0
    s_isch = 1 if "Reduced/Absent" in ischemia else 0
    s_neur = 1 if "Present" in neuropathy else 0
    s_inf  = 1 if (bio["Periwound_Erythema"] > 25.0 or bio["Slough_Pct"] > 35.0) else 0
    s_area = 1 if bio["Area_Pct"] > 12.0 else 0
    s_dept = 1 if "Deep" in depth else 0
    sinbad_score = s_site + s_isch + s_neur + s_inf + s_area + s_dept
    
    debridement_rec = (
        f"⚠️ INDICATED: {devitalized_pct:.1f}% of the wound bed consists of devitalized slough/eschar. "
        "Perform sharp surgical debridement (unless contraindicated by severe uncorrected ischemia)."
        if devitalized_pct > 25.0 else
        f"✅ HEALTHY BED: {bio['Granulation_Pct']:.1f}% red granulation tissue. Maintain moist wound dressing and avoid aggressive debridement."
    )
    
    abx_rec = (
        f"⚠️ SUSPECTED LOCAL INFECTION (Periwound Erythema Index: {bio['Periwound_Erythema']:.1f}%, Slough: {bio['Slough_Pct']:.1f}%): "
        "Obtain post-debridement deep tissue curettage culture and initiate empirical IDSA/IWGDF antibiotic therapy."
        if s_inf == 1 else
        "✅ NO OVERT OPTICAL CELLULITIS: Do not prescribe routine systemic antibiotics for clinically uninfected ulcers (Antibiotic Stewardship)."
    )
    
    clinical_note = (
        f"📋 AUTOMATED PODIATRIC & ENDOCRINE CONSULTATION REPORT\n"
        f"==========================================================\n"
        f"1. OPTICAL WOUND-BED COMPOSITION (RYB Colorimetric Staging):\n"
        f"   • Healthy Granulation (Red) : {bio['Granulation_Pct']:.1f}%\n"
        f"   • Fibrinous Slough (Yellow) : {bio['Slough_Pct']:.1f}%\n"
        f"   • Necrotic Eschar (Black)   : {bio['Necrotic_Pct']:.1f}%\n"
        f"   • Periwound Erythema Halo   : {bio['Periwound_Erythema']:.1f}%\n\n"
        f"2. IWGDF SINBAD SEVERITY SCORE : {sinbad_score} / 6 "
        f"({'HIGH RISK FOR AMPUTATION — Urgent Multidisciplinary Referral' if sinbad_score >= 3 else 'MODERATE RISK — Outpatient Podiatric Management'})\n\n"
        f"3. POINT-OF-CARE MANAGEMENT ORDERS:\n"
        f"   • Debridement : {debridement_rec}\n"
        f"   • Antibiotics : {abx_rec}\n"
        f"   • Offloading  : Mandatory non-removable knee-high offloading device (Total Contact Cast).\n"
        f"   • Vascular    : {'URGENT ABI / Duplex Ultrasound & Vascular Surgery Consult.' if s_isch == 1 or bio['Necrotic_Pct'] > 15 else 'Pedal perfusion intact; monitor clinically.'}"
    )
    
    tissue_breakdown = {
        "Red Granulation (Healing Tissue)": float(bio["Granulation_Pct"] / 100.0),
        "Yellow Slough (Fibrin/Exudate)": float(bio["Slough_Pct"] / 100.0),
        "Black Eschar (Necrotic/Ischemic)": float(bio["Necrotic_Pct"] / 100.0)
    }
    return gradcam_img, bio["Seg_Map"], tissue_breakdown, clinical_note
