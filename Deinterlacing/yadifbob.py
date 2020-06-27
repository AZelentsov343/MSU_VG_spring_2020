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

                    diff = np.maximum(dT, dS)

                    # place for spatial
                    d_spat = (field[field_row].astype('int16') + field[field_row + 1].astype('int16')) / 2

                    spat = np.abs(d_spat - d_temp) <= diff
                    temp_plus = d_spat - d_temp > diff
                    temp_minus = d_spat - d_temp < -diff

                    # print(np.where(temp_plus))
                    for channel in range(num_channels):
                        for x in range(width):
                            if spat[x, channel]:
                                deinterlaced[index, row, x, channel] = np.rint(d_spat[x, channel]).astype('uint8')
                            elif temp_plus[x, channel]:
                                deinterlaced[index, row, x, channel] = np.rint(d_temp[x, channel] +
                                                                               diff[x, channel]).astype('uint8')
                            else:
                                deinterlaced[index, row, x, channel] = np.rint(d_temp[x, channel] -
                                                                               diff[x, channel]).astype('uint8')

    return deinterlaced