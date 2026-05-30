from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument('--comp_dict', required=False, default=None,type=str,help="For multi component data give the component dictionary like this : '{'comp1':'c1idx0,c1idx1' , 'comp2' : 'c2idx0, c2idx1', ...}' ")
parser.add_argument('--compare', action='store_true')
args = parser.parse_args()

print(args.compare)
print(args.comp_dict)

comps = json.loads(args.comp_dict)
comp_dict = {}

if args.comp_dict is None:
    
    comp_dict = None

else:
    comps = json.loads(args.comp_dict)
    comp_dict = {}
    for key,value in comps.items():
        comp_dict[key] = (int(value.split(",")[0]),int(value.split(",")[1]))

print(comp_dict)
