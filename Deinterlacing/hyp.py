# GRADED FUNCTION: deinterlace_video
import numpy as np


def deinterlace_video(fields, top_frame_first=True):
    num_frames, height, width, num_channels = fields.shape
    deinterlaced = np.zeros((num_frames, height * 2, width, num_channels), dtype='uint8')
    sig_c = 1
    sig_s = 1

    for index, field in enumerate(tqdm(fields)):
        next_field = fields[index + 1].astype('int16') if index + 1 < len(fields) else None
        next_next_field = fields[index + 2].astype('int16') if index + 2 < len(fields) else None
        prev_field = fields[index - 1].astype('int16') if index - 1 >= 0 else None
        prev_prev_field = fields[index - 2].astype('int16') if index - 2 >= 0 else None
        field = field.astype('int16')
        gray = cv2.cvtColor(field.astype('uint8'), cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        # MVF = compute_flow(field, next_field)
        # MVB = compute_flow(field, prev_field)
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

                    c = field[field_row].astype('int16')
                    e = field[field_row + 1].astype('int16')
                    if prev_field is not None and next_field is not None:
                        b = (prev_field[max(0, prev_field_row - 1)].astype('int16') +
                             next_field[max(0, prev_field_row - 1)].astype('int16')) / 2
                        f = (prev_field[min(prev_field_row + 1, prev_field.shape[0] - 1)].astype('int16') +
                             next_field[min(prev_field_row + 1, prev_field.shape[0] - 1)].astype('int16')) / 2
                        d_temp = (prev_field[prev_field_row].astype('int16') +
                                  next_field[prev_field_row].astype('int16')) / 2
                        dT1 = np.abs(prev_field[prev_field_row].astype('int16') -
                                     next_field[prev_field_row].astype('int16'))
                    elif next_field is not None:
                        b = next_field[max(0, prev_field_row - 1)].astype('int16')
                        f = next_field[min(prev_field_row + 1, next_field.shape[0] - 1)].astype('int16')
                        d_temp = next_field[prev_field_row].astype('int16')
                        dT1 = np.zeros((width, num_channels))
                    else:
                        b = prev_field[max(0, prev_field_row - 1)].astype('int16')
                        f = prev_field[min(prev_field_row + 1, prev_field.shape[0] - 1)].astype('int16')
                        d_temp = prev_field[prev_field_row].astype('int16')
                        dT1 = np.zeros((width, num_channels))

                    if prev_prev_field is not None:
                        dT2 = (np.abs(prev_prev_field[field_row].astype('int16') - c) +
                               np.abs(prev_prev_field[field_row + 1].astype('int16') - e)) / 2
                    else:
                        dT2 = np.zeros((width, num_channels))

                    if next_next_field is not None:
                        dT3 = (np.abs(next_next_field[field_row].astype('int16') - c) +
                               np.abs(next_next_field[field_row + 1].astype('int16') - e)) / 2
                    else:
                        dT3 = np.zeros((width, num_channels))

                    dT = np.maximum(2 * dT1, dT2, dT3)

                    dS_min = np.minimum((d_temp - e), (d_temp - c), np.maximum(b - c, f - e))
                    dS_max = np.maximum((d_temp - e), (d_temp - c), np.minimum(b - c, f - e))
                    dS = np.maximum(dS_min, -dS_max)

                    diff = np.maximum(dS, dT)
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

                        spat = np.abs(d_spat - d_temp[x]) <= diff[x]
                        temp_plus = d_spat - d_temp[x] > diff[x]
                        # temp_minus = d_spat - d_temp[x] < -np.abs(diff[x])

                        for channel in range(num_channels):
                            if spat[channel]:
                                deinterlaced[index, row, x, channel] = np.rint(d_spat[channel]).astype('uint8')
                            elif temp_plus[channel]:
                                deinterlaced[index, row, x, channel] = np.rint(d_temp[x, channel] +
                                                                               diff[x, channel]).astype('uint8')
                            else:
                                deinterlaced[index, row, x, channel] = np.rint(d_temp[x, channel] -
                                                                               diff[x, channel]).astype('uint8')

    return deinterlaced