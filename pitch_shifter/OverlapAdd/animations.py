from utils import sine

import numpy as np
import matplotlib.pyplot as plt

fs = 44100
t, sig = sine(261)

sig = sig[:4000]

params = {
    'window_size': 512,
    'hop_size': 128,
    'factor': 1.41
}

def update_plot():
    ax1.clear()
    ax2.clear()
    
    ws = params['window_size']
    hs = params['hop_size']
    f = params['factor']
    h_sync = int(hs * f)
    
    n_frames = int(np.floor((len(sig) - ws) / hs)) + 1
    win = np.hanning(ws)
    
    total_len_out = (n_frames - 1) * h_sync + ws
    combined = np.zeros(total_len_out)
    
    ax1.plot(sig, color='lightgray', alpha=0.5)
    colors = plt.cm.viridis(np.linspace(0, 1, n_frames))
    
    for i in range(n_frames):
        start_in = i * hs
        frame = sig[start_in : start_in + ws] * win
        
        start_out = i * h_sync
        combined[start_out : start_out + ws] += frame
        
        if i % 2 == 0: 
            ax1.plot(np.arange(start_in, start_in + ws), sig[start_in:start_in+ws], color=colors[i], alpha=0.7)
            ax2.plot(np.arange(start_out, start_out + ws), frame, color=colors[i], alpha=0.5)

    ax2.plot(combined, color='black', linewidth=1.2, label="Sum")
    
    ax1.set_title(f"Window Size = {ws} (A/D) | Hop In = {hs} (Left/Right)")
    ax2.set_title(f"Hop Out = {h_sync} (Factor {f})")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.draw()

def on_key(event):
    if event.key == 'd':
        params['window_size'] += 32
    elif event.key == 'a':
        params['window_size'] = max(32, params['window_size'] - 32)
    elif event.key == 'right':
        params['hop_size'] += 16
    elif event.key == 'left':
        params['hop_size'] = max(16, params['hop_size'] - 16)
    
    params['hop_size'] = min(params['hop_size'], params['window_size'])
    
    update_plot()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.canvas.mpl_connect('key_press_event', on_key)

print("A/D for Window Size | Left/Right for Hop Size")
update_plot()
plt.show()