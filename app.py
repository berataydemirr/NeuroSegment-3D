import os
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import gradio as gr
from src.model import get_model

# --- System Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/brats_unet_epoch10.pth"

print(f"System initialized on: {DEVICE}")

# --- Model Loading Logic ---
def load_system_model():
    try:
        model = get_model(DEVICE)
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"Weights loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Running with random weights for UI testing.")
        
        model.eval()
        return model
    except Exception as e:
        print(f"Critical Error loading model: {str(e)}")
        return None

model = load_system_model()

# --- Processing Pipeline ---
def process_scan(file_obj, slice_idx):
    if file_obj is None or model is None:
        return None
    
    # Reading NIfTI file
    try:
        nifti = nib.load(file_obj.name)
        raw_volume = nifti.get_fdata()
    except Exception as e:
        return None

    # Preprocessing (Resize to 96x96x96 for the model)
    input_tensor = torch.tensor(raw_volume).float().unsqueeze(0).unsqueeze(0)
    input_tensor = torch.nn.functional.interpolate(input_tensor, size=(96, 96, 96), mode='trilinear')
    input_tensor = input_tensor.to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        segmentation_mask = (logits.sigmoid() > 0.5).float()
        
    # Extract slices
    vol_img = input_tensor[0, 0].cpu().numpy()
    vol_mask = segmentation_mask[0, 1].cpu().numpy()
    
    # Calculate slice index
    total_slices = vol_img.shape[2]
    current_slice = int(slice_idx * (total_slices - 1) / 100)
    
    # Visualization (Professional Matplotlib Style)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
    plt.subplots_adjust(wspace=0.1, hspace=0)
    
    # Raw MRI
    ax[0].imshow(vol_img[:, :, current_slice], cmap="gray")
    ax[0].set_title(f"T1-Weighted MRI (Slice {current_slice})", fontsize=10, pad=10, color='#374151')
    ax[0].axis("off")
    
    # Segmentation Overlay
    ax[1].imshow(vol_img[:, :, current_slice], cmap="gray")
    ax[1].imshow(vol_mask[:, :, current_slice], cmap="Reds", alpha=0.4)
    ax[1].set_title("AI Segmentation Mask", fontsize=10, pad=10, color='#dc2626')
    ax[1].axis("off")
    
    plt.close(fig)
    return fig

# --- Frontend Engineering (CSS & Layout) ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
body {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
}
/* Header Styling */
#header-box {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
}
#header-title {
    font-size: 1.8rem;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: -0.025em;
}
#header-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.25rem;
}
/* Documentation Section */
.doc-box {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}
.doc-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #334155;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.doc-text {
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.6;
}
/* Control Panel */
.control-panel {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
}
/* Button */
button.primary-btn {
    background-color: #2563eb !important;
    border-radius: 6px !important;
}
.footer {
    margin-top: 4rem;
    text-align: center;
    border-top: 1px solid #e2e8f0;
    padding-top: 2rem;
    color: #94a3b8;
    font-size: 0.8rem;
}
"""

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="#f8fafc",
    block_background_fill="white",
    block_border_width="1px",
    input_background_fill="#f8fafc",
)

# --- Application Layout ---
with gr.Blocks(theme=theme, css=custom_css, title="NeuroSegment Analytics") as demo:
    
    # 1. Header
    with gr.Column(elem_id="header-box"):
        gr.Markdown(
            """
            <div id="header-title">NeuroSegment 3D Analytics</div>
            <div id="header-subtitle">Automated Volumetric Brain Tumor Segmentation System</div>
            """
        )

    # 2. Documentation & Guide (New Section)
    with gr.Row(elem_classes="doc-box"):
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="doc-title">Operational Workflow</div>
                <div class="doc-text">
                1. <b>Data Ingestion:</b> Upload a T1-weighted MRI scan in NIfTI format (.nii or .nii.gz).<br>
                2. <b>Slice Navigation:</b> Utilize the depth slider to select the specific axial slice for analysis.<br>
                3. <b>Execution:</b> Initiate the segmentation pipeline to generate the binary mask overlay.
                </div>
                """
            )
        
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="doc-title">Intended Audience & Utility</div>
                <div class="doc-text">
                <ul>
                    <li><b>Clinical Researchers:</b> For comparative analysis of automated segmentation architectures against manual contouring.</li>
                    <li><b>Deep Learning Engineers:</b> To evaluate model generalization on diverse volumetric datasets.</li>
                    <li><b>Radiology Education:</b> Demonstrating capabilities of 3D Computer Vision in pathological detection.</li>
                </ul>
                </div>
                """
            )

    # 3. Main Workspace
    with gr.Row():
        
        # Left: Controls
        with gr.Column(scale=1, elem_classes="control-panel"):
            gr.Markdown("<div class='doc-title'>Control Parameters</div>")
            
            input_file = gr.File(
                label="Input Sequence (NIfTI)",
                file_count="single",
                height=100
            )
            
            slice_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=50,
                step=1,
                label="Axial Depth Index (%)"
            )
            
            process_btn = gr.Button(
                "Execute Segmentation", 
                variant="primary", 
                elem_classes="primary-btn"
            )
            
            with gr.Accordion("Technical Specifications", open=False):
                gr.Markdown("Architecture: 3D U-Net | Framework: MONAI/PyTorch | Precision: FP32")

        # Right: Visualization
        with gr.Column(scale=3):
            output_plot = gr.Plot(show_label=False, container=False)

    # 4. Footer
    gr.Markdown(
        """
        <div class="footer">
        NeuroSegment Analytics • Research Preview • Not for Clinical Diagnosis
        </div>
        """
    )

    # Logic
    process_btn.click(fn=process_scan, inputs=[input_file, slice_slider], outputs=output_plot)
    slice_slider.change(fn=process_scan, inputs=[input_file, slice_slider], outputs=output_plot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)