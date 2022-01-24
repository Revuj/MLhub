import argparse
import sys
from model_parser import parse_json

def build_parser():
    parser = argparse.ArgumentParser(description='MLHub - PornHub but for ML')
    parser.add_argument('--specs', required=True)
    parser.add_argument('--dockerize', required=False, action='store_true', default=False)
    parser.add_argument('--features', type=str, required=False, default=None)
    parser.add_argument('--labels', type=str, required=False, default=None)
    parser.set_defaults(dockerize=False)
    return parser

def parse_args(parser):
    return parser.parse_args()

def main():
    parser = build_parser()
    args = parse_args(parser)
    parse_json(args.specs, args.dockerize, features=args.features, labels=args.labels)
    sys.exit(0)

if __name__ == '__main__':
    main()
