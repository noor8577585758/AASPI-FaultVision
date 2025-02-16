from flask import Flask, render_template, request, redirect, url_for
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import segyio
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model
from scipy.stats import entropy
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Input, Concatenate
import os
import plotly.offline as pyo

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

custom_objects = {
    "Model": Model,
    "Conv3D": Conv3D,
    "MaxPooling3D": MaxPooling3D,
    "UpSampling3D": UpSampling3D,
    "Concatenate": Concatenate
}

json_path = "model/model3.json"
weights_path = "model/pretrained_model.hdf5"

with open(json_path, 'r', encoding="utf-8") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
loaded_model.load_weights(weights_path)

def predictWithMask(gx):
    os = 12  # Overlap width
    m1, m2, m3 = gx.shape
    c1 = int(np.round((m1 + os) / (128 - os) + 0.5))
    c2 = int(np.round((m2 + os) / (128 - os) + 0.5))
    c3 = int(np.round((m3 + os) / (128 - os) + 0.5))
    
    p1 = (128 - os) * c1 + os
    p2 = (128 - os) * c2 + os
    p3 = (128 - os) * c3 + os

    gp = np.zeros((p1, p2, p3), dtype=np.single)
    gy = np.zeros((p1, p2, p3), dtype=np.single)
    mk = np.zeros((p1, p2, p3), dtype=np.single)
    gs = np.zeros((1, 128, 128, 128, 1), dtype=np.single)
    
    gp[:m1, :m2, :m3] = gx
    
    for k1 in range(c1):
        for k2 in range(c2):
            for k3 in range(c3):
                b1, e1 = k1 * 128 - k1 * os, k1 * 128 - k1 * os + 128
                b2, e2 = k2 * 128 - k2 * os, k2 * 128 - k2 * os + 128
                b3, e3 = k3 * 128 - k3 * os, k3 * 128 - k3 * os + 128
                
                gs[0, :, :, :, 0] = gp[b1:e1, b2:e2, b3:e3]
                gs = gs - np.min(gs)
                gs = gs / np.max(gs)
                gs = gs * 255
                Y = loaded_model.predict(gs, verbose=1)
                Y = np.array(Y)
                
                gy[b1:e1, b2:e2, b3:e3] += Y[0, :, :, :, 0]
                mk[b1:e1, b2:e2, b3:e3] += 1

    gy = gy / mk
    gy = gy[:m1, :m2, :m3]
    
    return gx, gy

@app.route('/', methods=['GET', 'POST'])
def index():
    fault_entropy = None 
    fault_variance = None  

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            selected_plot = request.form.get("plot_selection", "inline")

            with segyio.open(filepath, "r", ignore_geometry=False) as segyio_file:
                n_ilines = len(segyio_file.ilines)
                n_xlines = len(segyio_file.xlines)
                n_times = segyio_file.trace.raw[0].size
                inline_range = list(segyio_file.ilines)

                print(f"✅ Total Inlines: {n_ilines}")
                print(f"✅ Total Crosslines: {n_xlines}")
                print(f"✅ Total Time Samples: {n_times}")
                print(f"✅ Inline Range: {inline_range[0]} to {inline_range[-1]}")
                print(f"✅ Selected Plot Type: {selected_plot}")  # Debugging statement

            gx = segyio.tools.cube(filepath)
            gx = gx[:, :, :256]  # Limit depth to 256
            gx, gy = predictWithMask(gx)

            m1, m2, m3 = gx.shape
            k1, k2, k3 = min(150, m1 - 1), min(100, m2 - 1), min(50, m3 - 1)
            gx1, gy1 = np.transpose(gx[k1, :, :]), np.transpose(gy[k1, :, :])
            gx2, gy2 = np.transpose(gx[:, k2, :]), np.transpose(gy[:, k2, :])
            gx3, gy3 = np.transpose(gx[:, :, k3]), np.transpose(gy[:, :, k3])

            fault_entropy = entropy(np.histogram(gy.flatten(), bins=10, density=True)[0])
            fault_variance = np.var(gy)

            def create_plotly_side_by_side(original, predicted, title="Seismic vs Faults"):
                def upscale(data, scale=4):
                    return zoom(data, scale, order=3) 
                
                original_upscaled = upscale(original, scale=4)
                predicted_upscaled = upscale(predicted, scale=4)
                fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Original Seismic", "Seismic with Fault Overlay"))
                fig.add_trace(go.Heatmap(z=original_upscaled, colorscale='gray', showscale=False), row=1, col=1)
                fig.add_trace(go.Heatmap(z=original_upscaled, colorscale='gray', showscale=False, opacity=1), row=1, col=2)
                fig.add_trace(go.Heatmap(z=predicted_upscaled, colorscale='Reds', showscale=False, opacity=0.3), row=1, col=2)
                fig.update_layout(title=title, width=1200, height=600)
                return fig

            fig_xline = create_plotly_side_by_side(gx1, gy1, "X-line Slice - Seismic vs Faults")
            fig_iline = create_plotly_side_by_side(gx2, gy2, "Inline Slice - Seismic vs Faults")
            fig_time = create_plotly_side_by_side(gx3, gy3, "Time Slice - Seismic vs Faults")

            plot_xline = pyo.plot(fig_xline, output_type='div')
            plot_iline = pyo.plot(fig_iline, output_type='div')
            plot_time = pyo.plot(fig_time, output_type='div')
            selected_plot = request.form.get("plot_selection", "xline")
            
            if selected_plot == "iline":
                return render_template('index.html', plot=plot_iline, selected_plot=selected_plot, fault_entropy=fault_entropy, fault_variance=fault_variance)
            elif selected_plot == "time":
                return render_template('index.html', plot=plot_time, selected_plot=selected_plot, fault_entropy=fault_entropy, fault_variance=fault_variance)
            else:
                return render_template('index.html', plot=plot_xline, selected_plot="xline", fault_entropy=fault_entropy, fault_variance=fault_variance)
    
    return render_template('index.html', plot=None, selected_plot="xline", fault_entropy=fault_entropy, fault_variance=fault_variance)



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)