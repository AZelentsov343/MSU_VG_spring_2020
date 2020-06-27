# GRADED FUNCTION: deinterlace_video

def deinterlace_video(fields, top_frame_first=False):
    num_frames, height, width, num_channels = fields.shape
    deinterlaced = np.zeros((num_frames, height * 2, width, num_channels), dtype='uint8')

    b = 8

    Tb = 32
    Ts = 20

    def is_cut_block(block, threshold_b):
        block = block.astype('uint8')
        gray = cv2.cvtColor(block.astype('uint8'), cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        summ = np.sum(canny / 255)
        return summ > threshold_b

    def MSE(block1, block2):
        return np.mean((block1.astype(np.int32) - block2) ** 2)

    def safe_edge(edge, max_edge):
        if edge > max_edge:
            edge = max_edge
        return edge

    def ex_search(block, reference_image):
        # reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
        # block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
        height, width, _ = reference_image.shape
        min_mse = np.inf
        bl = reference_image[:8:, :8:, :]
        coords = np.array([0, 0])
        for i in range(height // block.shape[0] + 1 * (height % block.shape[0] > 0)):
            for j in range(width // block.shape[1] + 1 * (width % block.shape[1] > 0)):
                ref_block = reference_image[block.shape[0] * i:safe_edge(block.shape[0] * (i + 1), height),
                            block.shape[1] * j:safe_edge(block.shape[1] * (j + 1), width)]
                if ref_block.shape[0] != block.shape[0] or ref_block.shape[1] != block.shape[1]:
                    continue
                mse = MSE(ref_block, block)
                if mse < min_mse:
                    min_mse = mse
                    # coords = np.array([block_size*i,block_size*j])
                    bl = ref_block
        return bl

    for index, field in enumerate(tqdm(fields)):
        next_field = fields[index + 1].astype('uint8') if index + 1 < len(fields) else None
        prev_field = fields[index - 1].astype('uint8') if index - 1 >= 0 else None
        top_frame_first = !top_frame_first

        if prev_field is not None:
            MVB_full = compute_flow(field, prev_field)
        if next_field is not None:
            MVF_full = compute_flow(field, next_field)

        # bruh_map = np.zeros(field.shape[:2:])
        for i in range(height // 8 + 1 * (height % 8 > 0)):
            for j in range(width // 8 + 1 * (width % 8 > 0)):
                h_begin = b * i
                h_end = min(b * (i + 1), height - 1)
                w_begin = b * j
                w_end = min(b * (j + 1), width - 1)
                block = field[h_begin: h_end:, w_begin:w_end]
                if block.shape[0] == 0 or block.shape[1] == 0:
                    continue

                '''sal = saliency(block)
                #print(i, j, np.mean(sal))
                if np.mean(sal) <= 20:
                    if top_frame_first:
                        deinterlaced[index, h_begin*2:h_end*2:2, w_begin:w_end] = block
                        deinterlaced[index, h_begin*2+1:h_end*2:2, w_begin:w_end] = bob(block, top_frame_first)                   
                    else:
                        deinterlaced[index, h_begin*2+1:h_end*2:2, w_begin:w_end] = block
                        deinterlaced[index, h_begin*2:h_end*2:2, w_begin:w_end] = bob(block, top_frame_first)'''
                if is_cut_block(block, block.shape[0] * block.shape[1] / 4):
                    print(1)
                else:
                    print(2)
                    deinterlaced[index, h_begin * 2:h_end * 2:, w_begin:w_end] = 255
                    '''if next_field is not None and prev_field is not None:
                        MVB = ex_search(block, prev_field)
                        MVF = ex_search(block, next_field)
                        mean = MVF / 2 + MVB / 2
                    elif next_field is not None:
                        mean = ex_search(block, next_field)
                    else:
                        mean = ex_search(block, prev_field)
                    #print(block.shape, mean.shape)
                    deinterlaced[index, h_begin*2:h_end*2:, w_begin:w_end] = merge_blocks(block, mean,
                                                                                          top_frame_first)'''

        return deinterlaced