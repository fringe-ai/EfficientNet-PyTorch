import torch
import struct
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from efficientnet_pytorch import EfficientNet

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path')
    parser.add_argument('-c', '--num_classes', required=True, help='the number of classes in the trained model')
    parser.add_argument('-o', '--out_file', required=True, help='the output weight file path')
    parser.add_argument('--inc', default=3, type=int, help='the number of image channels, default=3')
    parser.add_argument('--model', default='efficientnet-b0', help='the name of the model, such as: efficientnet-b0')
    args = vars(parser.parse_args())
    if not os.path.isfile(args['weights']):
        raise SystemExit('Invalid input file')
    return args

args = parse_args()

num_classes = int(args['num_classes'])
out_file = args['out_file']
inc = args['inc']
model = EfficientNet.from_pretrained(args['model'], num_classes=num_classes, in_channels=inc).to(torch.double)
model.load_state_dict(torch.load(args['weights']))
model.eval()

print(f'writing weights to {out_file}')
f = open(out_file, 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
f.close()
print('Done')
