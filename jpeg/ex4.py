import numpy as np
import matplotlib.pyplot as plt
import heapq
import ast
import cv2

from scipy.fft import dctn, idctn
from scipy import datasets
from collections import Counter

# UTILS FUNCTIONS

Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def zig_zag_scan(block):
    rows, cols = block.shape
    solution = [[] for _ in range(rows + cols - 1)]

    for i in range(rows):
        for j in range(cols):
            sum_idx = i + j
            if sum_idx % 2 == 0:
                solution[sum_idx].insert(0, block[i, j])
            else:
                solution[sum_idx].append(block[i, j])
    return np.concatenate(solution)

def inverse_zigzag_scan(vector):
    block = np.zeros((8, 8))
    rows, cols = 8, 8
    
    idx_map = []
    for sum_idx in range(rows + cols - 1):
        temp = []
        for i in range(rows):
            for j in range(cols):
                if i + j == sum_idx:
                    temp.append((i, j))
        if sum_idx % 2 == 0:
            idx_map.extend(temp[::-1])
        else:
            idx_map.extend(temp)
    
    for val, (r, c) in zip(vector, idx_map):
        block[r, c] = val
    return block

def huffman_simulate(zigzag_vec):
    dc_val = zigzag_vec[0]
    ac_vals = zigzag_vec[1:]

    rle_pairs = []
    zeros_count = 0
    
    for val in ac_vals:
        if val == 0:
            zeros_count += 1
        else:
            rle_pairs.append((zeros_count, int(val)))
            zeros_count = 0
    rle_pairs.append("EOB")
    
    return dc_val, rle_pairs

def build_huffman_table(all_symbols):
    str_symbols = [str(s) for s in all_symbols]
    freqs = Counter(str_symbols)
    
    # Heap with [count, [symbol,binary_code]]
    heap = [[f, [s, ""]] for s, f in freqs.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # return a dict of {symbol:binary_code}
    return dict(heapq.heappop(heap)[1:])

def find_alpha_for_mse(image, target_mse, tol=1.0, max_iter=16):
    low_alpha = 0.0001 
    high_alpha = 20.0
    
    print(f"Searching for MSE: {target_mse}")
    print(f"Original image size: {image.nbytes / 1024 / 8:.2f}KB")
    
    for _ in range(max_iter):
        mid_alpha = (low_alpha + high_alpha) / 2
        current_Q = Q_jpeg * mid_alpha
        
        bits, table, shape = jpeg_encode(image, current_Q)
        decoded = jpeg_decode(shape, bits, table, current_Q)
        
        current_mse = np.mean((image - decoded)**2)
        print(f"Searching for a better alpha={mid_alpha:.3f} -> MSE={current_mse:<10.3f} | {(len(bits)/8)/1024:.2f}KB")

        if abs(current_mse - target_mse) < tol:
            return decoded, mid_alpha, current_mse
        
        if current_mse < target_mse:
            low_alpha = mid_alpha
        else:
            high_alpha = mid_alpha
            
    return decoded, mid_alpha, current_mse

def jpeg_encode(dataset, Q_matrix):
    h, w = dataset.shape

    all_symbols = []

    # Division into 8x8 Pixel Blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            pixel_block = dataset[i:i+8, j:j+8] - 128
            
            # Forward DCT
            y = dctn(pixel_block, type=2, norm='ortho')
            
            # Quantization
            y_quant = np.round(y / Q_matrix).astype(int)
            
            # Vectorize the matrix with Zig Zag Scan
            zz = zig_zag_scan(y_quant)

            # After the zigzag scan, we will have a 64 value vector. The first value is the mean of the block, while the other values are small coefficients. These can be encoded using the run length encoding. (i.e. rather than storing [0,0,0,0,5], we can store a pair of (zeroes,next_val) -> (4,5)). Then we can apply huffman on the pairs.
            dc, ac_rle = huffman_simulate(zz)
            
            current_block = [("DC", int(dc))] + ac_rle
            all_symbols.extend(current_block)

    huffman_table = build_huffman_table(all_symbols)
    bitstream = "".join([huffman_table[str(s)] for s in all_symbols])

    return bitstream, huffman_table, (h,w)

def jpeg_decode(shape, bitstream, table, Q_matrix):    
    h, w = shape
    reverse_huffman = {v: k for k, v in table.items()}

    decoded_symbols = []
    temp_bits = ""
    for bit in bitstream:
        temp_bits += bit
        if temp_bits in reverse_huffman:
            symbol_str = reverse_huffman[temp_bits]

            if symbol_str != "EOB":
                real_symbol = ast.literal_eval(symbol_str)
                decoded_symbols.append(real_symbol)
            else:
                decoded_symbols.append(symbol_str)
            
            temp_bits = ""

    X_hat = np.zeros((h,w))
    sym_ptr = 0

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            reconstructed_zz = np.zeros(64)
            
            first_sym = decoded_symbols[sym_ptr]
            sym_ptr += 1
            reconstructed_zz[0] = first_sym[1]
            
            current_pos = 1
            while sym_ptr < len(decoded_symbols):
                current_sym = decoded_symbols[sym_ptr]
                sym_ptr += 1
                
                if current_sym == "EOB":
                    break
                
                run, val = current_sym
                current_pos += run
                
                if current_pos < 64:
                    reconstructed_zz[current_pos] = val
                    current_pos += 1
            
            inv_zz = inverse_zigzag_scan(reconstructed_zz)
            dequant = inv_zz * Q_matrix
            X_hat[i:i+8, j:j+8] = idctn(dequant, type=2, norm='ortho') + 128

    return X_hat

def load_video_frames_512(file_path, max_frames=30, start=256):
    cap = cv2.VideoCapture(file_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[start * 4:start * 4 + 512, start:start + 512]
        frames.append(crop.astype(float))        

    cap.release()
    return frames

def compress_video(frames):
    compressed_video_data = []
    total_bits = 0
    
    for i, frame in enumerate(frames):
        bits, table, shape = jpeg_encode(frame, 10 * Q_jpeg)
            
        compressed_video_data.append({
            'bits': bits,
            'table': table,
            'shape': shape,
            'Q': 10 * Q_jpeg
        })
        total_bits += len(bits)
        print(f"Compressed frame {i}")

    return compressed_video_data, total_bits

def decompress_video(compressed_data):
    decompressed_frames = []
    for i, data in enumerate(compressed_data):
        frame = jpeg_decode(data['shape'], data['bits'], data['table'], data['Q'])
        print(f"Decompressed frame {i}")
        decompressed_frames.append(frame)
    return decompressed_frames

def save_frames_to_video(frames_list, output_name='output.mp4', fps=24):
    if not frames_list:
        return

    height, width = frames_list[0].shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, size, isColor=False)

    for frame in frames_list:
        frame_to_save = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame_to_save)

    out.release()
    print(f"Saved video: {output_name}")

video_frames = load_video_frames_512("tree_30fps.mp4", 90)
compressed_v, size_bits = compress_video(video_frames)
reconstructed_video = decompress_video(compressed_v)

save_frames_to_video(reconstructed_video, 'compressed.mp4', fps=30)