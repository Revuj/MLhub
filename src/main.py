import argparse
import sys
from model_parser import parse_json

def build_parser():
    parser = argparse.ArgumentParser(description='MLHub - PornHub but for ML')
    parser.add_argument('--specs', required=True)
    parser.add_argument('--dockerize', required=False, default=False)
    return parser

def parse_args(parser):
    return parser.parse_args()

def main():
    parser = build_parser()
    args = parse_args(parser)
    parse_json(args.specs, args.dockerize)
    sys.exit(0)

if __name__ == '__main__':
    main()
