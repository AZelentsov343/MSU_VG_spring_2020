# GRADED FUNCTION: deinterlace_video

def deinterlace_video(fields, top_frame_first=True):
    num_frames, height, width, num_channels = fields.shape
    deinterlaced = np.zeros((num_frames, height * 2, width, num_channels), dtype='uint8')

    b = 8

    Tb = 32

    def is_cut_block(block, threshold_b):
        block = block.astype('uint8')
        gray = cv2.cvtColor(block.astype('uint8'), cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        summ = np.sum(canny / 255)
        return summ > threshold_b

    for index, field in enumerate(tqdm(fields)):
        next_field = fields[index + 1].astype('uint8') if index + 1 < len(fields) else None
        prev_field = fields[index - 1].astype('uint8') if index - 1 >= 0 else None
        # field = field.astype('int16')

        # blocks = [field[8*i:min(8*(i+1),height-1):, 8*j:min(8*(j+1), width-1)] for i in range(height) for j in range(width)]

        bruh_map = np.zeros(field.shape[:2:])
        for i in range(height // 8):
            for j in range(width // 8):
                block = field[b * i: min(b * (i + 1), height - 1):, b * j:min(b * (j + 1), width - 1)]
                if is_cut_block(block.astype('uint8'), Tb):
                    bruh_map[b * i: min(b * (i + 1), height - 1):, b * j:min(b * (j + 1), width - 1)] = 1
        if next_field is not None:
            MVF = compute_flow(field, next_field)
        if prev_field is not None:
            MVB = compute_flow(field, prev_field)

        for row in range(height * 2):
            residual = (row + index) % 2
            field_row = (row - residual) // 2
            prev_field_row = row // 2

            if (top_frame_first and residual == 0) or (not top_frame_first and residual == 1):
                deinterlaced[index, row] = field[field_row]
            else:
                if field_row < 0:
                    # Важно приводить к uint8
                    deinterlaced[index, row] = (field[0] / 2).astype('uint8')
                elif field_row + 1 >= height:
                    deinterlaced[index, row] = (field[-1] / 2).astype('uint8')
                else:
                    # deinterlaced[index, row] = \
                    # np.rint(field[field_row] / 2 + field[field_row + 1] / 2).astype('uint8')
                    for x in range(width):
                        if (bruh_map[field_row, x] == 0) or prev_field is None or next_field is None:
                            deinterlaced[index, row, x] = \
                                np.rint(field[field_row, x] / 2 + field[field_row + 1, x] / 2).astype('uint8')
                        else:
                            # print(int(x + MVB[prev_field_row, x, 1]))
                            deinterlaced[index, row, x] = (prev_field[int(prev_field_row + MVB[prev_field_row, x, 1]),
                                                                      int(x + MVB[prev_field_row, x, 0])] +
                                                           next_field[int(prev_field_row + MVF[prev_field_row, x, 1]),
                                                                      int(x + MVF[prev_field_row, x, 0])]) / 2

    return deinterlaced


razmakh = 2

# place for spatial
for x in range(width):
    upper = canny[field_row, max(0, x - razmakh):min(x + razmakh + 1, width)] == 255
    lower = canny[field_row + 1, max(0, x - razmakh):min(x + razmakh + 1, width)] == 255

    stup = field[field_row, max(0, x - razmakh):min(x + razmakh + 1, width)]
    stlow = field[field_row + 1, max(0, x - razmakh):min(x + razmakh + 1, width)]

    if np.all(lower == False) or np.all(upper == False):
        d_spat = np.rint(field[field_row, x] / 2 +
                         field[field_row + 1, x] / 2).astype('uint8')
    else:
        try:
            done = False
            for k in range(2 * razmakh + 1):
                if upper[k] and lower[2 * razmakh - k]:
                    d_spat = np.rint(stup[k] / 2 +
                                     stlow[2 * razmakh - k] / 2).astype('uint8')
                    done = True
            if not done:
                d_spat = np.rint(field[field_row, x] / 2 +
                                 field[field_row + 1, x] / 2).astype('uint8')
        except IndexError:
            d_spat = np.rint(field[field_row, x] / 2 +
                             field[field_row + 1, x] / 2).astype('uint8')