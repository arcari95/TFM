# def bin_to_float(binary):
#     di = binary.find('.')
#     if di == -1:
#         return int(binary,2)
#     else:
#         whole = binary[:di] or '0'
#         fraction = binary [di+1:] or '0'
#         return int(whole,2) + int(whole,2)/abs(int(whole,2)) * int(fraction,2) / 2** len(fraction)
#
# samples = [
#     '00.1010101011',
#
# ]
#
# for binary in samples:
#     print(binary,'=',bin_to_float(binary))

def binToFloat(b):
    parts = b.split('.')
    s,t = parts
    x = 0 if len(s) == 0 else int(s,2)
    y = 0 if len(t) == 0 else int(t,2)
    return x + y/2**len(t)

print(binToFloat('0.11011011101'))